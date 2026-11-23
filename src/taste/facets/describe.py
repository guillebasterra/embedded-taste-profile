"""Thread cards: the VLM names each thread and says how to carry it into new work.

Writes profile/threads.json (the cards, read by distill and serve as plain JSON),
profile/threads.md and a few thumbnails per thread under profile/threads/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from taste.interfaces import VLM
from taste.store import Store

from .read import load_readings
from .threads import Thread, spread_examples

PROFILE_FILE = "threads.json"
PROFILE_MD = "threads.md"
THUMB_DIR = "threads"
CARD_IMAGES = 8
THUMBS = 4
THUMB_EDGE = 512
SHOWN_READINGS = 12


class ThreadDraft(BaseModel):
    name: str = Field(description="2 to 5 words naming the quality, not a subject.")
    one_liner: str = Field(description="One sentence an agent can act on.")
    qualities: list[str] = Field(description=(
        "3 to 6 observable statements of what the thread is, each specific enough to check "
        "against an image or a page."))
    in_boards: str = Field(description=(
        "One or two sentences on how it shows up across the boards it spans, naming them."))
    for_web: list[str] = Field(description=(
        "3 to 5 concrete moves for a website or app in this thread: layout, type, color, "
        "imagery, texture, borders, motion. Values where you can (sizes, ratios, hex)."))
    for_other_media: str = Field(description=(
        "One sentence for photos, illustrations, objects or clothing."))
    avoid: list[str] = Field(description="2 to 4 choices that would break the thread.")
    distinctiveness: int = Field(ge=1, le=5, description=(
        "1 = how most saved images on the internet look (e.g. any photo with a blurred "
        "background); 5 = a specific, unusual preference."))


class ThreadCard(ThreadDraft):
    thread_id: str
    spread: float
    pins: int
    boards: dict[str, int]
    domains: dict[str, int]
    terms: list[str]
    measures: dict[str, float]
    exemplars: list[str]  # pin ids
    thumbnails: list[str]  # paths relative to profile/
    model: str
    members_hash: str


SYSTEM = ("You turn evidence about one person's saved images into taste guidance an agent "
          "can apply in any medium. Stay observable and specific. Never use 'clean', "
          "'modern', 'aesthetic', 'elevated' or 'timeless' without saying what makes it so.")


def members_hash(thread: Thread) -> str:
    return hashlib.sha1("\n".join(sorted(thread.pins)).encode()).hexdigest()[:16]


def thread_prompt(thread: Thread, shown: list[str], boards: dict[str, str],
                  readings: dict) -> str:
    lines = [
        f"These {len(shown)} images belong to one thread: a bundle of visual qualities that "
        f"recurs in {len(thread.pins)} of this person's saved images across about "
        f"{thread.spread} boards. It was found from per-image tags, ignoring subject, so "
        "it should hold for different subjects.",
        "",
        f"Top terms: {', '.join(thread.terms)}",
        "Boards (pins in thread): " + ", ".join(f"{b} {c}" for b, c in thread.boards.items()),
        "Domains: " + ", ".join(f"{d} {c}" for d, c in thread.domains.items()),
        "Measured against the person's average pin (positive = more): " + ", ".join(
            f"{k} {v:+.2f}" for k, v in thread.measures.items()),
        "",
        "Per-image notes:",
    ]
    for n, pid in enumerate(shown, start=1):
        r = readings.get(pid)
        if r is not None:
            lines.append(f"- Image {n} ({boards.get(pid, '?')}): {', '.join(r.style_tags)}. "
                         f"{r.distinctive}")
    others = [p for p in thread.pins if p not in shown][: SHOWN_READINGS - len(shown)]
    for pid in others:
        r = readings.get(pid)
        if r is not None:
            lines.append(f"- (not shown, {boards.get(pid, '?')}): {', '.join(r.style_tags)}")
    lines += ["", "Name the thread and write its card. Describe the quality the images share, "
              "not what they depict. `for_web` is the part an agent building a site will use "
              "most: translate the quality, do not copy the subjects."]
    return "\n".join(lines)


def _thumb(src: Path, dest: Path) -> None:
    from PIL import Image, ImageOps

    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((THUMB_EDGE, THUMB_EDGE))
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, "JPEG", quality=85)


def read_cards(store: Store) -> list[ThreadCard]:
    path = store.profile_dir / PROFILE_FILE
    return [ThreadCard.model_validate(c) for c in json.loads(path.read_text())] \
        if path.exists() else []


def describe_threads(store: Store, threads: list[Thread], vlm: VLM, *,
                     force: bool = False) -> tuple[list[ThreadCard], int]:
    """One card per thread; a card whose members and model are unchanged is reused.

    Returns (cards, how many were generated).
    """
    pins = {p.id: p for p in store.read_pins()}
    boards = {i: p.board for i, p in pins.items()}
    readings = load_readings(store)
    old = {c.members_hash: c for c in read_cards(store) if c.model == vlm.name}
    thumb_dir = store.profile_dir / THUMB_DIR
    cards, generated = [], 0
    for t in threads:
        h = members_hash(t)
        shown = spread_examples(t, boards, CARD_IMAGES)
        if not force and h in old:
            draft = ThreadDraft.model_validate(old[h].model_dump())
        else:
            draft = vlm.generate(thread_prompt(t, shown, boards, readings), ThreadDraft,
                                 images=[store.abs(pins[p].image_path) for p in shown],
                                 system=SYSTEM)
            generated += 1
        thumbs = []
        for n, pid in enumerate(shown[:THUMBS], start=1):
            rel = f"{THUMB_DIR}/{t.id}-{n}.jpg"
            _thumb(store.abs(pins[pid].image_path), store.profile_dir / rel)
            thumbs.append(rel)
        cards.append(ThreadCard(
            **draft.model_dump(), thread_id=t.id, spread=t.spread, pins=len(t.pins),
            boards=t.boards, domains=t.domains, terms=t.terms, measures=t.measures,
            exemplars=shown, thumbnails=thumbs, model=vlm.name, members_hash=h))
    keep = {rel for c in cards for rel in c.thumbnails}
    if thumb_dir.exists():
        for f in thumb_dir.glob("*.jpg"):
            if f"{THUMB_DIR}/{f.name}" not in keep:
                f.unlink()
    store.profile_dir.mkdir(parents=True, exist_ok=True)
    (store.profile_dir / PROFILE_FILE).write_text(
        json.dumps([c.model_dump() for c in cards], indent=1, ensure_ascii=False))
    (store.profile_dir / PROFILE_MD).write_text(render_threads_md(cards))
    return cards, generated


def render_threads_md(cards: list[ThreadCard]) -> str:
    out = ["# Threads", "",
           "Visual qualities that recur across my boards whatever the subject, widest first. "
           "Carry the wide ones into any medium; a thread with low distinctiveness is common "
           "ground, not a signature.", ""]
    for c in cards:
        out += [f"## {c.thread_id} · {c.name}", "",
                f"{c.one_liner}", "",
                f"*{c.pins} pins across about {c.spread} boards · distinctiveness "
                f"{c.distinctiveness}/5 · top boards: {', '.join(list(c.boards)[:5])}*", ""]
        out += [f"- {q}" for q in c.qualities]
        out += ["", c.in_boards, "", "**On the web:**", ""]
        out += [f"- {m}" for m in c.for_web]
        out += ["", f"**Elsewhere:** {c.for_other_media}", "", "**Avoid:**", ""]
        out += [f"- {a}" for a in c.avoid]
        if c.thumbnails:
            out += ["", " ".join(f"![]({p})" for p in c.thumbnails)]
        out.append("")
    return "\n".join(out).rstrip() + "\n"
