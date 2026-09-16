"""
Prompt Loader and Manager for Agent System Prompts & Judge Prompts.
"""

from __future__ import annotations
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def load_prompt(prompt_name: str) -> str:
    file_path = PROMPTS_DIR / prompt_name
    if file_path.exists():
        return file_path.read_text(encoding="utf-8").strip()
    
    # Fallback inline prompts if prompt file missing
    if "agent" in prompt_name:
        return "You are an AI customer support agent. Answer customer queries concisely using retrieved context."
    return "Score the agent response from 1 to 5 on correctness, completeness, and groundedness."


AGENT_SYSTEM_PROMPT = load_prompt("agent_system.txt")
JUDGE_PROMPT_TEMPLATE = load_prompt("judge_prompt.txt")
