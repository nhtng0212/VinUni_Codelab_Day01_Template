"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency_seconds = time.time() - start_time

    response_text = response.choices[0].message.content or ""
    usage_data = response.usage
    usage = {
        "input_tokens": getattr(usage_data, "prompt_tokens", 0),
        "output_tokens": getattr(usage_data, "completion_tokens", 0),
    }

    return response_text, latency_seconds, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency_seconds = time.time() - start_time

    response_text = response.text or ""

    usage_metadata = getattr(response, "usage_metadata", None)
    usage = {
        "input_tokens": getattr(usage_metadata, "prompt_token_count", 0),
        "output_tokens": getattr(usage_metadata, "candidates_token_count", 0),
    }

    return response_text, latency_seconds, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_seconds = time.time() - start_time

    response_text = response.content[0].text if response.content else ""
    usage_data = response.usage
    usage = {
        "input_tokens": getattr(usage_data, "input_tokens", 0),
        "output_tokens": getattr(usage_data, "output_tokens", 0),
    }

    return response_text, latency_seconds, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.
    """
    # 1. Call models
    resp_4o_text, lat_4o, usage_4o = call_openai(prompt, model=OPENAI_MODEL)
    resp_mini_text, lat_mini, usage_mini = call_openai(prompt, model=OPENAI_MINI_MODEL)
    resp_gemini_text, lat_gemini, usage_gemini = call_gemini(prompt, model=GEMINI_MODEL)

    # 2. Calculate costs
    def calc_cost(model_key: str, in_tok: int, out_tok: int) -> float:
        rates = PRICING_1M_TOKENS.get(model_key, {"input": 0.0, "output": 0.0})
        return (in_tok * rates["input"] + out_tok * rates["output"]) / 1_000_000

    cost_4o = calc_cost(
        OPENAI_MODEL, usage_4o["input_tokens"], usage_4o["output_tokens"]
    )
    cost_mini = calc_cost(
        OPENAI_MINI_MODEL, usage_mini["input_tokens"], usage_mini["output_tokens"]
    )
    cost_gemini = calc_cost(
        GEMINI_MODEL, usage_gemini["input_tokens"], usage_gemini["output_tokens"]
    )

    # 3. Build result structure
    return {
        "gpt4o": {
            "response": resp_4o_text,
            "latency": lat_4o,
            "cost": cost_4o,
            "input_tokens": usage_4o["input_tokens"],
            "output_tokens": usage_4o["output_tokens"],
        },
        "gpt4o_mini": {
            "response": resp_mini_text,
            "latency": lat_mini,
            "cost": cost_mini,
            "input_tokens": usage_mini["input_tokens"],
            "output_tokens": usage_mini["output_tokens"],
        },
        "gemini_flash": {
            "response": resp_gemini_text,
            "latency": lat_gemini,
            "cost": cost_gemini,
            "input_tokens": usage_gemini["input_tokens"],
            "output_tokens": usage_gemini["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.
    """
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Sử dụng chats để duy trì lịch sử hội thoại tự động
    chat = client.chats.create(model=GEMINI_MODEL)

    print("Gemini 2.5 Chatbot initialized. Type 'quit' or 'exit' to end.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            if not user_input:
                continue

            print("Gemini: ", end="", flush=True)
            response_stream = chat.send_message_stream(user_input)

            for chunk in response_stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception:
            if attempt >= max_retries:
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list and inject the prompt string.
    """
    results = []
    for prompt in prompts:
        res = compare_models(prompt)
        res["prompt"] = prompt
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.
    | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    table_lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]

    for item in results:
        prompt_str = item.get("prompt", "").replace("\n", " ")
        if len(prompt_str) > 30:
            prompt_str = prompt_str[:27] + "..."

        model_keys = [
            ("gpt4o", "GPT-4o"),
            ("gpt4o_mini", "GPT-4o-Mini"),
            ("gemini_flash", "Gemini Flash"),
        ]

        for m_key, m_label in model_keys:
            m_data = item.get(m_key, {})
            resp = m_data.get("response", "").replace("\n", " ")
            if len(resp) > 50:
                resp = resp[:47] + "..."

            lat = f"{m_data.get('latency', 0.0):.2f}s"
            tokens = f"{m_data.get('input_tokens', 0)}/{m_data.get('output_tokens', 0)}"
            cost = f"${m_data.get('cost', 0.0):.6f}"

            table_lines.append(
                f"| {prompt_str} | {m_label} | {resp} | {lat} | {tokens} | {cost} |"
            )

    return "\n".join(table_lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
