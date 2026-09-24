"""
patches.py
Workaround for a known CrewAI + Groq/LiteLLM compatibility bug where CrewAI's
internal prompt-caching marker ('cache_breakpoint') leaks into requests sent
to non-native providers like Groq, which reject it as an unsupported field.
This strips that marker (and any other CrewAI-internal keys) before the
request reaches litellm.completion().
"""

import litellm

_original_completion = litellm.completion

UNSUPPORTED_MESSAGE_KEYS = ["cache_breakpoint"]


def _clean_messages(messages):
    cleaned = []
    for msg in messages:
        cleaned.append({k: v for k, v in msg.items() if k not in UNSUPPORTED_MESSAGE_KEYS})
    return cleaned


def _patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _clean_messages(kwargs["messages"])
    return _original_completion(*args, **kwargs)


def apply_patch():
    litellm.completion = _patched_completion