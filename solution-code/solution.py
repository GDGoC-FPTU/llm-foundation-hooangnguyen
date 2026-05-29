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
import openai
import google.genai
from google.genai import types
import anthropic

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
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    response_text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return response_text, latency, usage


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
    client = google.genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )
    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )
    latency = time.time() - start_time
    response_text = response.text
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    return response_text, latency, usage


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
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start_time = time.time()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    response_text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)

def compare_models(prompt: str) -> dict:
    """Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash

    (gemini-2.5-flash) with the same prompt and return a structured comparison
    dictionary.

    Calculate the exact USD token cost for input + output using the prices in
    PRICING_1M_TOKENS.
    """

    res_gpt4o, lat_gpt4o, usage_gpt4o = call_openai(prompt, model="gpt-4o")
    res_mini, lat_mini, usage_mini = call_openai(prompt, model="gpt-4o-mini")
    res_gemini, lat_gemini, usage_gemini = call_gemini(
        prompt, model="gemini-2.5-flash"
    )
    def calculate_cost(usage: dict, model_key: str) -> float:
        in_tokens = usage.get("input_tokens", 0)
        out_tokens = usage.get("output_tokens", 0)

        in_rate = PRICING_1M_TOKENS[model_key]["input"]
        out_rate = PRICING_1M_TOKENS[model_key]["output"]

        return (in_tokens * in_rate + out_tokens * out_rate) / 1_000_000

    cost_gpt4o = calculate_cost(usage_gpt4o, "gpt-4o")
    cost_mini = calculate_cost(usage_mini, "gpt-4o-mini")
    cost_gemini = calculate_cost(usage_gemini, "gemini-2.5-flash")
    comparison_results = {
        "gpt4o": {
            "response": res_gpt4o,
            "latency": lat_gpt4o,
            "cost": cost_gpt4o,
            "input_tokens": usage_gpt4o.get("input_tokens", 0),
            "output_tokens": usage_gpt4o.get("output_tokens", 0),
        },
        "gpt4o_mini": {
            "response": res_mini,
            "latency": lat_mini,
            "cost": cost_mini,
            "input_tokens": usage_mini.get("input_tokens", 0),
            "output_tokens": usage_mini.get("output_tokens", 0),
        },
        "gemini_flash": {
            "response": res_gemini,
            "latency": lat_gemini,
            "cost": cost_gemini,
            "input_tokens": usage_gemini.get("input_tokens", 0),
            "output_tokens": usage_gemini.get("output_tokens", 0),
        },
    }

    return comparison_results
# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    client = google.genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []
    
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
            
        if user_input.strip().lower() in ["quit", "exit"]:
            break
            
        if not user_input.strip():
            continue
            
        history.append({"role": "user", "parts": [{"text": user_input}]})
        
        # Keep history to last 3 turns (3 turns = 6 messages: 3 user + 3 model)
        if len(history) > 6:
            history = history[-6:]
            
        print("Bot: ", end="", flush=True)
        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=history
        )
        
        full_response = ""
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        print()
        
        history.append({"role": "model", "parts": [{"text": full_response}]})


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

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries:
                raise e
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    for prompt in prompts:
        try:
            comp = compare_models(prompt)
        except TypeError:
            comp = compare_models()
        comp_copy = dict(comp)
        comp_copy["prompt"] = prompt
        results.append(comp_copy)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    headers = ["Prompt", "Model", "Response (truncated)", "Latency", "Tokens (In/Out)", "Cost (USD)"]
    model_mapping = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash"
    }
    
    # Table header
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    
    for item in results:
        prompt = item.get("prompt", "")
        for key in ["gpt4o", "gpt4o_mini", "gemini_flash"]:
            if key not in item:
                continue
            model_name = model_mapping.get(key, key)
            stats = item[key]
            
            resp = stats.get("response", "")
            if len(resp) > 50:
                truncated_resp = resp[:47] + "..."
            else:
                truncated_resp = resp
            # remove newlines/tabs from truncated response to keep markdown table clean
            truncated_resp = truncated_resp.replace("\n", " ").replace("\r", "")
                
            latency = f"{stats.get('latency', 0.0):.2f}s"
            tokens = f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)}"
            cost = f"${stats.get('cost', 0.0):.6f}"
            
            row = [prompt, model_name, truncated_resp, latency, tokens, cost]
            lines.append("| " + " | ".join(row) + " |")
            
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
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
