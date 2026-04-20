#!/usr/bin/env python3
"""Lightweight Claude API reviewer - replaces Codex MCP for ARIS workflows."""

import argparse
import json
import os
import sys
from anthropic import Anthropic

client = Anthropic()

def review(prompt: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 8192) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

def main():
    parser = argparse.ArgumentParser(description="Claude API reviewer")
    parser.add_argument("--prompt-file", "-f", help="Read prompt from file")
    parser.add_argument("--prompt", "-p", help="Prompt string")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-20250514")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--output", "-o", help="Save response to file")
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = sys.stdin.read()

    result = review(prompt, model=args.model, max_tokens=args.max_tokens)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"Response saved to {args.output}")
    else:
        print(result)

if __name__ == "__main__":
    main()
