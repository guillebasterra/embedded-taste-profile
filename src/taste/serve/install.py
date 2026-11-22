"""`taste install`: skill file, CLAUDE.md snippet, MCP config, export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from taste.store import Store

SKILL_NAME = "taste"
SKILL_DESCRIPTION = (
    "The user's personal visual taste profile. Use before any visual or design decision "
    "(UI, web, layout, typography, color, imagery, photo, art, fashion), after producing a "
    "screenshot or image, and whenever the user corrects a visual choice."
)

SKILL_MD = f"""---
name: {SKILL_NAME}
description: {SKILL_DESCRIPTION}
---

# Taste

My visual taste, distilled from my saved images into principles, cluster cards and
exemplar images. Consult it instead of guessing or falling back on generic defaults.

## When to consult it

- Before any visual decision: layout, typography, color, spacing, imagery, art direction,
  photo or outfit selection. Read first, then design.
- After producing anything visual (screenshot, render, mockup, generated image): check it
  before calling it done.
- Every time I correct, reject or praise a visual choice: log it.

## Tools (MCP server `taste`)

- `get_taste(domain?, detail?, brief?)`: no arguments returns `TASTE.md` (summary,
  cross-domain invariants, standing corrections). Pass a domain (`web`, `photo`, `art`,
  `fashion`, `other`) for that domain, and `detail="full"` for its cluster cards with
  principles, do, avoid and tokens. Pass `brief` (a sentence about the project) to get my
  clusters ranked by fit to it, with the best-fitting cards in full.
- `get_exemplars(query, domain?, k?)`: reference images I saved that match a text query,
  as thumbnails. Use when a principle is too abstract to act on.
- `critique(image_path)`: scores an image against my taste and returns the best-matching
  cluster's principles and avoid list. Run it on every screenshot you produce, fix what it
  flags, and run it again.
- `log_feedback(verdict, note, image_path?, domain?, context?)`: records a correction.
  `verdict` is `like`, `dislike` or `rule` (a standing instruction).

## Rules

1. Log every visual correction with `log_feedback`, in my words, without being asked.
   Attach the screenshot when there is one. Use `rule` for anything I say should always
   or never happen.
2. Standing corrections in `TASTE.md` override cluster cards, and cluster cards override
   your defaults.
3. There are several tastes per domain. Pick the cluster that fits the task (call
   `get_taste` with a `brief` to rank them) and stay inside it; do not average them.
4. If a tool says the profile is not built, tell me which `taste` command it named.

## Without the MCP server

The profile is plain files next to this one: `TASTE.md`, `DESIGN.md`,
`domains/<domain>.md`, `clusters/<id>.md`, `exemplars/<cluster-id>/*.jpg`. Read them
directly, and append corrections with `taste feedback add <verdict> "<note>"`.
"""

CLAUDE_SNIPPET_MD = """## Visual taste

I have a taste profile served by the `taste` MCP server.

- Before any visual decision (UI, layout, type, color, imagery), call `get_taste` (with a
  one-sentence `brief` of the project), and `get_exemplars` when you need reference images.
- After producing a screenshot or image, call `critique` on it and fix what it flags.
- Log every visual correction with `log_feedback` (`like`, `dislike` or `rule`), in my
  words, without being asked.
"""


def write_install_files(store: Store) -> list[Path]:
    store.profile_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, text in (("SKILL.md", SKILL_MD), ("CLAUDE.snippet.md", CLAUDE_SNIPPET_MD)):
        path = store.profile_dir / name
        path.write_text(text)
        out.append(path)
    return out


def mcp_config(home: str | Path) -> dict:
    home_abs = str(Path(home).expanduser().resolve())
    return {"mcpServers": {"taste": {"command": "taste", "args": ["serve", "--home", home_abs]}}}


def mcp_config_json(home: str | Path) -> str:
    return json.dumps(mcp_config(home), indent=2)


def export_profile(store: Store, dest: str | Path) -> Path:
    """Copy profile/ into dest (merged over what is there; nothing in dest is deleted)."""
    dest_path = Path(dest).expanduser().resolve()
    src = store.profile_dir.resolve()
    if dest_path == src or src in dest_path.parents or dest_path in src.parents:
        raise ValueError(f"Export target {dest_path} overlaps the profile directory {src}.")
    shutil.copytree(src, dest_path, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.tmp", ".DS_Store"))
    return dest_path
