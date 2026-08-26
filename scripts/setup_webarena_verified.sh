#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/setup_webarena_verified.sh --host <hostname-or-ip> [options]

Options:
  --host <hostname-or-ip>  Hostname or IP used by the browser/evaluator.
  --data-dir <path>        Download/data directory. Default: $HOME/webarena_verified_data
  --reset                  Stop existing WebArena-Verified containers before starting.
  --setup-only             Download/setup data and pull/start nothing else.
  --no-wait                Start containers without waiting for health checks.
  --skip-homepage          Do not start the lightweight homepage helper on :4399.
  -h, --help               Show this help.

This uses ServiceNow WebArena-Verified Docker images for:
  shopping, shopping_admin, reddit, gitlab, wikipedia, map

Wikipedia and map require external data downloads. Homepage is not a
WebArena-Verified environment, so this script optionally runs the original
WebArena homepage as a small Flask container for compatibility.
USAGE
}

WEB_HOST=""
DATA_DIR="${WEB_ARENA_VERIFIED_DATA_DIR:-$HOME/webarena_verified_data}"
RESET=0
SETUP_ONLY=0
WAIT_ARGS=()
SKIP_HOMEPAGE=0

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
    --setup-only)
      SETUP_ONLY=1
      shift
      ;;
    --no-wait)
      WAIT_ARGS=(--no-wait)
      shift
      ;;
    --skip-homepage)
      SKIP_HOMEPAGE=1
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

if [[ -z "$WEB_HOST" && "$SETUP_ONLY" -eq 0 ]]; then
  echo "ERROR: --host is required unless --setup-only is used." >&2
  usage >&2
  exit 2
fi

WEB_HOST="${WEB_HOST#http://}"
WEB_HOST="${WEB_HOST#https://}"
WEB_HOST="${WEB_HOST%/}"

WIKI_DATA_DIR="$DATA_DIR/wikipedia"
MAP_DATA_DIR="$DATA_DIR/map"
HOMEPAGE_DIR="$DATA_DIR/webarena-homepage"
mkdir -p "$WIKI_DATA_DIR" "$MAP_DATA_DIR"

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not available to this user. Start Docker or use a docker-enabled user." >&2
  exit 1
fi

if command -v webarena-verified >/dev/null 2>&1; then
  WAV=(webarena-verified)
elif command -v uvx >/dev/null 2>&1; then
  WAV=(uvx webarena-verified)
else
  echo "ERROR: Install uv/uvx or webarena-verified first. Example: pip install webarena-verified" >&2
  exit 1
fi

free_gb=$(df -BG "$DATA_DIR" | awk 'NR == 2 {gsub("G", "", $4); print $4}')
if [[ "$free_gb" -lt 300 ]]; then
  echo "WARNING: $DATA_DIR has ${free_gb}GB free. 350GB+ free is safer for images, wiki, map downloads, and map volumes."
fi

remove_partial_if_size_mismatch() {
  local url="$1"
  local file="$2"
  local remote_size=""
  local local_size=""

  [[ -f "$file" ]] || return 0

  remote_size=$(curl -fsIL "$url" 2>/dev/null | awk 'BEGIN {IGNORECASE=1} /^content-length:/ {gsub("\r", "", $2); size=$2} END {print size}' || true)
  [[ -n "$remote_size" ]] || return 0

  local_size=$(stat -c%s "$file")
  if [[ "$local_size" -ne "$remote_size" ]]; then
    echo "Removing incomplete download: $file (${local_size}/${remote_size} bytes)"
    rm -f "$file"
  fi
}

prepare_download_dirs() {
  remove_partial_if_size_mismatch \
    http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim \
    "$WIKI_DATA_DIR/wikipedia_en_all_maxi_2022-05.zim"
  remove_partial_if_size_mismatch \
    https://webarena-map-server-data.s3.amazonaws.com/osm_tile_server.tar \
    "$MAP_DATA_DIR/osm_tile_server.tar"
  remove_partial_if_size_mismatch \
    https://webarena-map-server-data.s3.amazonaws.com/nominatim_volumes.tar \
    "$MAP_DATA_DIR/nominatim_volumes.tar"
  remove_partial_if_size_mismatch \
    https://webarena-map-server-data.s3.amazonaws.com/osrm_routing.tar \
    "$MAP_DATA_DIR/osrm_routing.tar"
}

download_homepage() {
  if [[ -f "$HOMEPAGE_DIR/app.py" && -d "$HOMEPAGE_DIR/templates" ]]; then
    return 0
  fi

  local tmp_dir
  tmp_dir=$(mktemp -d)
  trap 'rm -rf "$tmp_dir"' RETURN

  git clone --depth 1 https://github.com/web-arena-x/webarena.git "$tmp_dir/webarena"
  rm -rf "$HOMEPAGE_DIR"
  cp -a "$tmp_dir/webarena/environment_docker/webarena-homepage" "$HOMEPAGE_DIR"

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

start_homepage() {
  [[ "$SKIP_HOMEPAGE" -eq 0 ]] || return 0

  download_homepage
  configure_homepage
  docker rm -f webarena_homepage >/dev/null 2>&1 || true
  docker run -d --name webarena_homepage -p 4399:4399 \
    --volume="$HOMEPAGE_DIR:/app:ro" -w /app \
    python:3.11-slim sh -c 'pip install --no-cache-dir flask && python app.py' >/dev/null
}

start_site() {
  local site="$1"
  shift
  "${WAV[@]}" env start --site "$site" "${WAIT_ARGS[@]}" "$@"
}

prepare_download_dirs

echo "Setting up Wikipedia data..."
"${WAV[@]}" env setup init --site wikipedia --data-dir "$WIKI_DATA_DIR"

echo "Setting up Map data and Docker volumes..."
"${WAV[@]}" env setup init --site map --data-dir "$MAP_DATA_DIR"

if [[ "$SETUP_ONLY" -eq 1 ]]; then
  echo "Setup complete. Containers were not started because --setup-only was used."
  exit 0
fi

if [[ "$RESET" -eq 1 ]]; then
  "${WAV[@]}" env stop-all || true
  docker rm -f webarena_homepage >/dev/null 2>&1 || true
fi

start_site shopping --port 7770 --env-ctrl-port 7771 --timeout 300
start_site shopping_admin --port 7780 --env-ctrl-port 7781 --timeout 300
start_site reddit --port 9999 --env-ctrl-port 9998 --timeout 300
start_site gitlab --port 8023 --env-ctrl-port 8024 --timeout 900
start_site wikipedia --port 8888 --env-ctrl-port 8889 --data-dir "$WIKI_DATA_DIR" --timeout 300
start_site map --port 3000 --env-ctrl-port 3001 --timeout 900
start_homepage

echo "Service check from this machine:"
for spec in \
  "Shopping:7770:/" \
  "Shopping Admin:7780:/admin" \
  "Reddit:9999:/" \
  "GitLab:8023:/" \
  "Wikipedia:8888:/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing" \
  "Map:3000:/"; do
  IFS=: read -r label port path <<< "$spec"
  curl -s -o /dev/null -w "$label ($port): %{http_code}\n" "http://$WEB_HOST:$port$path" || true
done
if [[ "$SKIP_HOMEPAGE" -eq 0 ]]; then
  curl -s -o /dev/null -w "Homepage (4399): %{http_code}\n" "http://$WEB_HOST:4399/" || true
fi

homepage_value="PASS"
if [[ "$SKIP_HOMEPAGE" -eq 0 ]]; then
  homepage_value="http://$WEB_HOST:4399"
fi

cat <<EOF

Use these environment variables for WebArena:
export SHOPPING="http://$WEB_HOST:7770"
export SHOPPING_ADMIN="http://$WEB_HOST:7780/admin"
export REDDIT="http://$WEB_HOST:9999"
export GITLAB="http://$WEB_HOST:8023"
export MAP="http://$WEB_HOST:3000"
export WIKIPEDIA="http://$WEB_HOST:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="$homepage_value"
EOF

echo "Done."
