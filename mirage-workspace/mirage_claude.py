#!/usr/bin/env python3
"""mirage_claude.py - mount a Mirage virtual filesystem and run Claude Code inside it.

Mirage (strukto-ai/mirage) presents heterogeneous backends (disk, Notion, GitHub,
Google Drive, S3, Redis, ...) as one FUSE filesystem. Claude Code then reads/writes/
greps/pipes across all of them with plain bash - no per-service SDK or MCP needed.

Default mount layout:
    /          -> DiskResource(~/mirage-workspace/diskroot)   (persistent scratch, WRITE)
    /notion    -> NotionResource         (READ)   [only if NOTION_TOKEN is a real value]
    /github    -> GitHubResource         (READ)   [only if GITHUB_TOKEN/GH_TOKEN is set]

Add more backends in build_resources() below - every resource in mirage.resource.*
follows the same Resource(Config(...)) shape.

Usage:
    uv run python mirage_claude.py             # mount, then launch `claude` in the mount
    uv run python mirage_claude.py --no-claude # mount and print the path; you cd in yourself
    uv run python mirage_claude.py --root DIR  # use a different disk root
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from mirage import Workspace, DiskResource, MountMode

# Tokens that are obviously placeholders (secrets get wiped/placeholdered on reboot).
_PLACEHOLDERS = {"", "placeholder", "changeme", "your_token_here", "xxx", "todo"}


def _real(val):
    if not val:
        return None
    if val.strip().lower() in _PLACEHOLDERS:
        return None
    return val.strip()


def build_resources(root):
    """Return the {mountpoint: (resource, mode)} map for the workspace."""
    os.makedirs(root, exist_ok=True)
    resources = {"/": (DiskResource(root), MountMode.WRITE)}

    notion_key = _real(os.environ.get("NOTION_TOKEN"))
    if notion_key:
        from mirage.resource.notion import NotionResource, NotionConfig
        resources["/notion"] = (NotionResource(NotionConfig(api_key=notion_key)), MountMode.READ)

    gh_token = _real(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
    if gh_token:
        from mirage.resource.github import GitHubResource, GitHubConfig
        resources["/github"] = (GitHubResource(GitHubConfig(token=gh_token)), MountMode.READ)

    return resources


def main():
    ap = argparse.ArgumentParser(description="Mount Mirage and run Claude Code inside it.")
    ap.add_argument("--root", default=os.path.expanduser("~/mirage-workspace/diskroot"),
                    help="Disk-backed root for the writable / mount.")
    ap.add_argument("--no-claude", action="store_true",
                    help="Just mount and print the path; do not launch claude.")
    args = ap.parse_args()

    resources = build_resources(args.root)

    with Workspace(resources, fuse=True, mode=MountMode.WRITE) as ws:
        mp = ws.fuse_mountpoint
        print("\n  Mirage workspace mounted.")
        print("  mountpoint : %s" % mp)
        for path in resources:
            print("     %-10s -> %s" % (path, type(resources[path][0]).__name__))
        print()

        if args.no_claude:
            print("  Run Claude Code against it with:\n\n    cd %s && claude\n" % mp)
            try:
                input("  Press Enter to unmount and exit... ")
            except (EOFError, KeyboardInterrupt):
                pass
            return 0

        print("  Launching Claude Code in %s ...\n" % mp)
        try:
            return subprocess.call(["claude"], cwd=mp)
        except FileNotFoundError:
            print("  `claude` not found on PATH. cd into the mountpoint above and run it manually.",
                  file=sys.stderr)
            return 127


if __name__ == "__main__":
    raise SystemExit(main())
