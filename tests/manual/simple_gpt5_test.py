#!/usr/bin/env python3
"""
Minimal non-streaming GPT-5 test.
Tries to load config (preferred). If not available, falls back to environment variables.
Asks a single question and prints the answer only.
"""
import os
import sys

# try to import project config; if running as module this should work
try:
    # ensure project root is on path when running from repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from configs.app_config import Config  # type: ignore
except Exception:
    Config = None  # type: ignore

from openai import AzureOpenAI


def get_gpt5_config_from_settings():
    if not Config:
        return None
    try:
        cfg = Config()
    except Exception:
        return None

    # Try to find a GPT-5 model in azure_llm.models
    try:
        for key, model in cfg.azure_llm.models.items():
            if (
                hasattr(model, "model_name")
                and model.model_name
                and "gpt-5" in model.model_name.lower()
            ):
                return model
            if "gpt_5" in key or "gpt-5" in key:
                return model
    except Exception:
        return None
    return None


def get_gpt5_config_from_env():
    api_key = os.getenv("AZURE_LLM__MODELS__GPT_5__API_KEY")
    endpoint = os.getenv("AZURE_LLM__MODELS__GPT_5__ENDPOINT")
    deployment = os.getenv("AZURE_LLM__MODELS__GPT_5__DEPLOYMENT")
    api_version = os.getenv("AZURE_LLM__MODELS__GPT_5__API_VERSION")
    model_name = os.getenv("AZURE_LLM__MODELS__GPT_5__MODEL_NAME")
    if all([api_key, endpoint, deployment, api_version]):

        class EnvModel:
            pass

        m = EnvModel()
        m.api_key = api_key
        m.endpoint = endpoint
        m.deployment = deployment
        m.api_version = api_version
        m.model_name = model_name or "gpt-5"
        return m
    return None


def minimal_test(question: str = "What is 2 + 2?") -> int:
    # Prefer config
    model_cfg = get_gpt5_config_from_settings() or get_gpt5_config_from_env()
    if not model_cfg:
        print(
            "❌ No GPT-5 configuration found. Export env vars or ensure Config is available."
        )
        return 1

    api_key = getattr(model_cfg, "api_key")
    endpoint = getattr(model_cfg, "endpoint")
    deployment = getattr(model_cfg, "deployment")
    api_version = getattr(model_cfg, "api_version")

    client = AzureOpenAI(
        api_key=api_key, azure_endpoint=endpoint, api_version=api_version
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": question}],
            max_completion_tokens=512,
        )
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return 2

    # Extract answer
    answer = None
    try:
        choices = getattr(response, "choices", None)
        if choices and len(choices) > 0:
            msg = getattr(choices[0], "message", None)
            if msg and getattr(msg, "content", None):
                answer = msg.content.strip()
    except Exception:
        answer = None

    if answer:
        print(answer)
        return 0

    # Fallback: print raw response for debugging
    try:
        if hasattr(response, "to_dict"):
            print(response.to_dict())
        else:
            print(response)
    except Exception:
        print("(no readable response content)")
    return 3


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", nargs="?", default="What is 2 + 2?", help="Question to ask GPT-5"
    )
    args = parser.parse_args()

    exit(minimal_test(args.question))
