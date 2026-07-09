#!/usr/bin/env python3
"""CLI: append Alchemy story from latest epoch JSON on stdin or file."""
import json
import sys
from pathlib import Path

from alchemy_story_logger import append_story_log

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "-":
        raw = sys.stdin.read()
    elif len(sys.argv) > 1:
        raw = Path(sys.argv[1]).read_text()
    else:
        raw = sys.stdin.read()
    epoch = json.loads(raw)
    chain = sys.argv[2] if len(sys.argv) > 2 else "base"
    print(json.dumps(append_story_log(epoch, chain=chain)))

if __name__ == "__main__":
    main()