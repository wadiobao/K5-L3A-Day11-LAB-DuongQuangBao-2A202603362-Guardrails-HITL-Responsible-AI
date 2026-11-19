"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)
  - OutputGuardrailPlugin (ADK)
  - LLM-as-Judge is optional and not required for CP2.
"""
from __future__ import annotations

import re

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent


def content_filter(response: str) -> dict:
    """Redact PII and secrets, returning safe/issues/redacted fields."""
    issues = []
    redacted = response or ""

    pii_patterns = {
        "Vietnamese phone number": r"(?<!\d)(?:\+?84|0)(?:[\s.-]?\d){9,10}(?!\d)",
        "Email address": r"(?i)\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
        "Vietnamese national ID": r"(?<!\d)(?:\d{9}|\d{12})(?!\d)",
        "API key": r"\bsk-[A-Za-z0-9_-]{8,}\b",
        # Handles examples such as "Admin password is admin123" as well as
        # "password: admin123" and "password=admin123".
        "Password": r"(?i)\b(?:admin\s+)?password\s*(?::|=|is)\s*[^\s,;.!?]+",
        "Database host": r"(?i)\b(?:db\.[A-Za-z0-9.-]+(?::\d{2,5})?|[A-Za-z0-9.-]+\.internal(?::\d{2,5})?)\b",
    }

    for name, pattern in pii_patterns.items():
        matches = list(re.finditer(pattern, redacted))
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# OPTIONAL: LLM-as-Judge. It is disabled by default and is not required for CP2.
SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE.
If UNSAFE, add a brief reason on the next line.
"""

safety_judge_agent = None
judge_runner = None


def _init_judge():
    """Initialize the optional judge runner if a judge agent is configured."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use the optional judge agent; fail open when it is not configured."""
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that redacts sensitive output before it is sent to the user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        if not getattr(llm_response, "content", None):
            return ""
        return "".join(
            part.text
            for part in llm_response.content.parts
            if getattr(part, "text", None)
        )

    async def after_model_callback(self, *, callback_context, llm_response):
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        filtered = content_filter(response_text)
        if filtered["issues"]:
            llm_response.content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=filtered["redacted"])],
            )
            self.redacted_count += 1

        if self.use_llm_judge:
            judge_result = await llm_safety_check(filtered["redacted"])
            if not judge_result.get("safe", False):
                llm_response.content = types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text="Xin lỗi, tôi không thể cung cấp nội dung này vì không đáp ứng yêu cầu an toàn."
                        )
                    ],
                )
                self.blocked_count += 1

        return llm_response


def test_content_filter():
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
        "Customer CCCD is 012345678901 and database is db.vinbank.internal.",
    ]
    print("Testing content_filter():")
    for response in test_responses:
        result = content_filter(response)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] {response[:60]!r}")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:100]}")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    test_content_filter()
