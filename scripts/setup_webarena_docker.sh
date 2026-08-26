#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/setup_webarena_docker.sh --host <hostname-or-ip> [options]

Options:
  --host <hostname-or-ip>  Hostname or IP used by the browser to access WebArena.
  --data-dir <path>        Download/data directory. Default: $HOME/webarena_docker
  --reset                  Remove existing WebArena containers before running.
  --download-only          Download/load Docker images and website data, but do not run containers.
  --skip-config            Run containers but skip Magento/GitLab URL reconfiguration.
  -h, --help               Show this help.

Examples:
  scripts/setup_webarena_docker.sh --host localhost --reset
  scripts/setup_webarena_docker.sh --host 12.34.56.78 --data-dir /data/webarena --reset

Notes:
  This sets up the individual WebArena websites that have public tar/data URLs:
  shopping, shopping_admin, forum, gitlab, wikipedia, and homepage. The map site
  is not included because the upstream README does not publish a simple map
  frontend image tar; use the WebArena AMI or the separate map backend cloud-init
  path.
USAGE
}

WEB_HOST=""
DATA_DIR="${WEB_ARENA_DOCKER_DIR:-$HOME/webarena_docker}"
RESET=0
RUN_CONTAINERS=1
CONFIGURE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      WEB_HOST="${2:-}"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="${2:-}"
      shift 2
      ;;
    --reset)
      RESET=1
      shift
      ;;
    --download-only)
      RUN_CONTAINERS=0
      CONFIGURE=0
      shift
      ;;
    --skip-config)
      CONFIGURE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$WEB_HOST" && "$RUN_CONTAINERS" -eq 1 ]]; then
  echo "ERROR: --host is required unless --download-only is used." >&2
  usage >&2
  exit 2
fi

WEB_HOST="${WEB_HOST#http://}"
WEB_HOST="${WEB_HOST#https://}"
WEB_HOST="${WEB_HOST%/}"

IMAGE_DIR="$DATA_DIR/images"
WIKI_DIR="$DATA_DIR/wiki"
HOMEPAGE_DIR="$DATA_DIR/webarena-homepage"
mkdir -p "$IMAGE_DIR" "$WIKI_DIR"

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  else
    echo "ERROR: docker is not available to this user. Start Docker or run with a docker-enabled user." >&2
    exit 1
  fi
fi

download_file() {
  local url="$1"
  local out="$2"
  local local_size="0"
  local remote_size=""

  if [[ -f "$out" ]]; then
    local_size=$(stat -c%s "$out")
    remote_size=$(curl -fsIL "$url" 2>/dev/null | awk 'BEGIN {IGNORECASE=1} /^content-length:/ {gsub("\r", "", $2); size=$2} END {print size}' || true)
    if [[ -n "$remote_size" && "$local_size" -eq "$remote_size" ]]; then
      echo "Using complete existing file: $out"
      return 0
    fi
    echo "Resuming/checking file: $out"
  else
    echo "Downloading: $url"
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$out" "$url"
  else
    curl -fL --retry 5 --retry-delay 10 -C - -o "$out" "$url"
  fi
}

image_exists() {
  "${DOCKER[@]}" image inspect "$1" >/dev/null 2>&1
}

load_image() {
  local image="$1"
  local url="$2"
  local tar_path="$IMAGE_DIR/${url##*/}"

  if image_exists "$image"; then
    echo "Docker image already loaded: $image"
    return 0
  fi

  download_file "$url" "$tar_path"
  echo "Loading Docker image: $image"
  "${DOCKER[@]}" load --input "$tar_path"
  rm -f "$tar_path"
}

container_exists() {
  "${DOCKER[@]}" ps -a --format '{{.Names}}' | grep -Fxq "$1"
}

run_or_start() {
  local name="$1"
  shift

  if container_exists "$name"; then
    echo "Starting existing container: $name"
    "${DOCKER[@]}" start "$name" >/dev/null
  else
    echo "Creating container: $name"
    "${DOCKER[@]}" run --name "$name" "$@"
  fi
}

retry() {
  local attempts="$1"
  local delay="$2"
  shift 2

  local n=1
  until "$@"; do
    if [[ "$n" -ge "$attempts" ]]; then
      return 1
    fi
    echo "Retry $n/$attempts failed; waiting ${delay}s: $*"
    n=$((n + 1))
    sleep "$delay"
  done
}

configure_magento() {
  local container="$1"
  local base_url="$2"

  retry 30 10 "${DOCKER[@]}" exec "$container" /var/www/magento2/bin/magento setup:store-config:set --base-url="$base_url"
  retry 30 10 "${DOCKER[@]}" exec "$container" mysql -u magentouser -pMyPassword magentodb \
    -e "UPDATE core_config_data SET value=\"${base_url}/\" WHERE path = \"web/secure/base_url\";"
  retry 30 10 "${DOCKER[@]}" exec "$container" /var/www/magento2/bin/magento cache:flush
}

download_homepage() {
  if [[ -f "$HOMEPAGE_DIR/app.py" && -d "$HOMEPAGE_DIR/templates" ]]; then
    echo "Using existing homepage files: $HOMEPAGE_DIR"
    return 0
  fi

  local tmp_dir
  tmp_dir=$(mktemp -d)
  trap 'rm -rf "$tmp_dir"' RETURN

  echo "Downloading WebArena homepage files..."
  if command -v git >/dev/null 2>&1; then
    git clone --depth 1 https://github.com/web-arena-x/webarena.git "$tmp_dir/webarena"
    rm -rf "$HOMEPAGE_DIR"
    cp -a "$tmp_dir/webarena/environment_docker/webarena-homepage" "$HOMEPAGE_DIR"
  else
    download_file https://github.com/web-arena-x/webarena/archive/refs/heads/main.tar.gz "$tmp_dir/webarena-main.tar.gz"
    tar -xzf "$tmp_dir/webarena-main.tar.gz" -C "$tmp_dir"
    rm -rf "$HOMEPAGE_DIR"
    cp -a "$tmp_dir/webarena-main/environment_docker/webarena-homepage" "$HOMEPAGE_DIR"
  fi

  rm -rf "$tmp_dir"
  trap - RETURN
}

configure_homepage() {
  local homepage_host="http://$WEB_HOST"

  HOMEPAGE_HOST="$homepage_host" perl -0pi -e '
    my $host = $ENV{HOMEPAGE_HOST};
    s|<your-server-hostname>|$host|g;
    s|https?://[^"<>]+:7770|$host . ":7770"|ge;
    s|https?://[^"<>]+:7780|$host . ":7780"|ge;
    s|https?://[^"<>]+:9999|$host . ":9999"|ge;
    s|https?://[^"<>]+:8023|$host . ":8023"|ge;
    s|https?://[^"<>]+:3000|$host . ":3000"|ge;
    s|https?://[^"<>]+:8888|$host . ":8888"|ge;
  ' "$HOMEPAGE_DIR/templates/index.html"
}

free_gb=$(df -BG "$DATA_DIR" | awk 'NR == 2 {gsub("G", "", $4); print $4}')
if [[ "$free_gb" -lt 350 ]]; then
  echo "WARNING: $DATA_DIR has ${free_gb}GB free. WebArena images/data are large; 500GB+ free is safer."
fi

load_image shopping_final_0712 \
  http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar
load_image shopping_admin_final_0719 \
  http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar
load_image postmill-populated-exposed-withimg \
  http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar
load_image gitlab-populated-final-port8023 \
  http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar

download_file \
  http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim \
  "$WIKI_DIR/wikipedia_en_all_maxi_2022-05.zim"
download_homepage
"${DOCKER[@]}" pull ghcr.io/kiwix/kiwix-serve:3.3.0
"${DOCKER[@]}" pull python:3.11-slim

if [[ "$RUN_CONTAINERS" -eq 0 ]]; then
  echo "Download/load complete. Containers were not started because --download-only was used."
  exit 0
fi

if [[ "$RESET" -eq 1 ]]; then
  "${DOCKER[@]}" rm -f shopping shopping_admin forum gitlab wikipedia homepage 2>/dev/null || true
fi

configure_homepage

run_or_start shopping -p 7770:80 -d shopping_final_0712
run_or_start shopping_admin -p 7780:80 -d shopping_admin_final_0719
run_or_start forum -p 9999:80 -d postmill-populated-exposed-withimg
run_or_start gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
run_or_start wikipedia --volume="$WIKI_DIR:/data:ro" -p 8888:80 -d \
  ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim
run_or_start homepage -d -p 4399:4399 --volume="$HOMEPAGE_DIR:/app:ro" -w /app \
  python:3.11-slim sh -c 'pip install --no-cache-dir flask && python app.py'

if [[ "$CONFIGURE" -eq 1 ]]; then
  echo "Waiting 60s before Magento configuration..."
  sleep 60
  configure_magento shopping "http://$WEB_HOST:7770"

  retry 30 10 "${DOCKER[@]}" exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
  retry 30 10 "${DOCKER[@]}" exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
  configure_magento shopping_admin "http://$WEB_HOST:7780"

  echo "Waiting 300s before GitLab reconfigure..."
  sleep 300
  retry 20 15 "${DOCKER[@]}" exec gitlab update-permissions
  retry 20 15 "${DOCKER[@]}" exec gitlab sed -i "s|^external_url.*|external_url 'http://$WEB_HOST:8023'|" /etc/gitlab/gitlab.rb
  retry 3 30 "${DOCKER[@]}" exec gitlab gitlab-ctl reconfigure
fi

echo "Service check from this machine:"
for spec in \
  "Shopping:7770:/" \
  "Shopping Admin:7780:/" \
  "Forum:9999:/" \
  "Wikipedia:8888:/" \
  "GitLab:8023:/explore" \
  "Homepage:4399:/"; do
  IFS=: read -r label port path <<< "$spec"
  curl -s -o /dev/null -w "$label ($port): %{http_code}\n" "http://$WEB_HOST:$port$path" || true
done

cat <<EOF

Use these environment variables for WebArena:
export SHOPPING="http://$WEB_HOST:7770"
export SHOPPING_ADMIN="http://$WEB_HOST:7780/admin"
export REDDIT="http://$WEB_HOST:9999"
export GITLAB="http://$WEB_HOST:8023"
export MAP="http://$WEB_HOST:3000"
export WIKIPEDIA="http://$WEB_HOST:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://$WEB_HOST:4399"

Note: MAP is only valid if you separately set up the WebArena map service on port 3000.
EOF

echo "Done."
