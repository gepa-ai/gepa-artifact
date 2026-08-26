#!/usr/bin/env python3
from pathlib import Path
import dspy

from verusage.output_format import SearchReplaceFormatter


PROMPT_DIR = Path(__file__).resolve().parents[1] / "verusage" / "agents" / "prompts"


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()


def list_action_prompts() -> list[str]:
    names = []
    for path in PROMPT_DIR.glob("*.md"):
        if path.stem == "assertion_reasoning_pipeline":
            continue
        names.append(path.stem)
    return sorted(names)


class RouterSignature(dspy.Signature):
    context: str = dspy.InputField(desc="Error info + code context + available actions")
    action: str = dspy.OutputField(desc="Selected action name, e.g., instantiate_forall")
    guidance: str = dspy.OutputField(desc="High-level guidance for the action")


class ActionSignature(dspy.Signature):
    context: str = dspy.InputField(desc="Error info + code context")
    patch: str = dspy.OutputField(desc="SEARCH/REPLACE response")


class VeruSAGEDSPyProgram(dspy.Module):
    """
    Minimal example: wrap VeruSAGE prompts as DSPy predictors so GEPA-SGD can optimize them.
    This does NOT run Verus; scoring should be done by your external metric.
    """

    def __init__(self):
        super().__init__()
        self.router = dspy.Predict(RouterSignature)
        self.router.signature = self.router.signature.with_instructions(
            load_prompt("assertion_reasoning_pipeline")
        )

        self.action_map = {}
        for name in list_action_prompts():
            pred = dspy.Predict(ActionSignature)
            pred.signature = pred.signature.with_instructions(load_prompt(name))
            setattr(self, name, pred)
            self.action_map[name] = pred

    def forward(self, code: str, error_text: str, error_type: str):
        actions_list = ", ".join(sorted(self.action_map.keys()))
        context = (
            f"Error type: {error_type}\n"
            f"Error text: {error_text}\n"
            f"Available actions: {actions_list}\n"
            f"Code:\n{code}\n"
        )

        router_out = self.router(context=context)
        action_name = (router_out.action or "").strip()
        action_pred = self.action_map.get(action_name)
        if action_pred is None:
            action_pred = self.action_map.get("fallback_llm_repair")

        action_out = action_pred(context=context) if action_pred else dspy.Prediction(patch="")

        patch_text = action_out.patch
        new_code = code
        try:
            ops = SearchReplaceFormatter.parse_search_replace_response(patch_text)
            new_code = SearchReplaceFormatter.apply_search_replace_operations(code, ops)
        except Exception:
            # If parsing fails, keep original code and return raw patch for debugging.
            pass

        return dspy.Prediction(
            action=action_name,
            guidance=router_out.guidance,
            patch=patch_text,
            new_code=new_code,
        )


if __name__ == "__main__":
    # Example usage: run the DSPy program with dummy inputs.
    # Replace LM config and inputs with real data when running.
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="dummy"))
    program = VeruSAGEDSPyProgram()
    result = program(code="verus! { proof fn foo() {} }", error_text="AssertFail", error_type="AssertFail")
    print(result.action)
    print(result.patch)
