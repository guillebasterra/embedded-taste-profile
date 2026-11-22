"""Board name -> domain guesses, and boards.toml upkeep that never overwrites the user."""

from __future__ import annotations

import re

from taste.models import BoardEntry
from taste.store import Store

# Checked in this order; the domain with the most keyword hits wins, ties go to the earlier one.
# A token matches a keyword when equal, its plural, or (keywords of 4+ letters) when it starts
# with it: "websites" -> web, "typography" -> web, "posters" -> art. Short keywords such as
# "ui" and "art" only match whole words, so "building" and "party" stay `other`.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "web": ("web", "website", "ui", "ux", "site", "type", "typo", "font", "landing", "interface",
            "saas", "dashboard", "app", "branding", "logo", "layout"),
    "photo": ("photo", "film", "35mm", "analog", "analogue", "camera", "portrait", "cinema",
              "kodak", "leica", "120mm"),
    "fashion": ("outfit", "fit", "fits", "wear", "denim", "fashion", "style", "clothes",
                "clothing", "streetwear", "menswear", "womenswear", "jeans", "shoes", "sneaker",
                "lookbook"),
    "art": ("art", "paint", "illustration", "poster", "drawing", "sketch", "sculpture", "collage",
            "print", "graphic", "artwork"),
}


def _matches(token: str, keyword: str) -> bool:
    if token == keyword or token == keyword + "s":
        return True
    if len(keyword) >= 4 and token.startswith(keyword):
        return True
    return len(keyword) >= 4 and token.endswith(keyword)  # "streetwear", "artwork", "outfit"


def guess_domain(board_name: str) -> str:
    tokens = [t for t in re.split(r"[^a-z0-9]+", board_name.lower()) if t]
    best, best_hits = "other", 0
    for domain, keywords in KEYWORDS.items():
        hits = sum(1 for t in tokens if any(_matches(t, k) for k in keywords))
        if hits > best_hits:
            best, best_hits = domain, hits
    return best


def update_boards(store: Store, board_names: list[str] | None = None,
                  ) -> tuple[dict[str, BoardEntry], list[str]]:
    """Add a guessed entry for every board that has none. Returns (all entries, added names).

    Existing entries are returned untouched, and the file is only rewritten when something
    was added, so hand edits (and comments) survive a no-op run.
    """
    boards = store.read_boards()
    if board_names is None:
        board_names = [p.board for p in store.read_pins()]
    added: list[str] = []
    for name in dict.fromkeys(board_names):
        if name not in boards:
            boards[name] = BoardEntry(domain=guess_domain(name))  # type: ignore[arg-type]
            added.append(name)
    if added or not store.boards_path.exists():
        store.write_boards(boards)
    return boards, added
