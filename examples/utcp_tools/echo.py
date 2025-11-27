#!/usr/bin/env python3
import json
import sys


def main():
    try:
        args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    except json.JSONDecodeError:
        args = {}

    text = args.get("text", "")
    print(json.dumps({"content": f"echo: {text}"}))


if __name__ == "__main__":
    main()
