"""Tool logic for the MCP server and the feedback CLI.

Plain functions: they take a Store and return data, so they are testable without a
transport. `server.py` wraps them for MCP, `cli.py` for the shell.
"""

from __future__ import annotations

import io
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from taste.config import DOMAINS, Settings, load_settings
from taste.interfaces import Embedder, FakeEmbedder
from taste.models import (
    ClusterCard,
    ClusterRecord,
    CritiqueResult,
    DomainSummary,
    ExemplarHit,
    Feedback,
)
from taste.scoring import api as scoring_api
from taste.store import Store

TOOL_NAMES = ("get_taste", "get_exemplars", "critique", "log_feedback")
VERDICTS = ("like", "dislike", "rule")
DETAILS = ("summary", "full")
THUMB_MAX_PX = 512
MAX_K = 12
BRIEF_TOP_CARDS = 3  # cards shown in full for a brief; the rest are one line each
BRIEF_THREADS = 6  # signature threads shown with their moves for a brief
BRIEF_RANKED_SHOWN = 10  # clusters listed in the ranking; the tail is counted
BRIEF_THREAD_MOVES = 3  # web moves per thread; the rest are in profile/threads.md
BRIEF_THREAD_AVOID = 2
BRIEF_CARD_DROP = ("## Measured palette", "## Neighbors", "## Exemplars")  # not for agents
THREAD_SIGNATURE_MIN = 3  # distinctiveness below this is common ground, named only
THREADS_FILE = "threads.json"  # profile/threads.json, written by `taste threads`
PIPELINE = "taste ingest -> taste domains -> taste embed -> taste cluster -> taste distill"

TOOL_DESCRIPTIONS = {
    "get_taste": (
        "Read the user's visual taste profile. Call this BEFORE making any visual decision "
        "(layout, typography, color, imagery, styling, art direction, outfit or photo "
        "selection), at the start of any design or frontend task. With no domain it returns "
        "TASTE.md: the summary, the threads (qualities that recur across every kind of "
        "board), cross-domain invariants and standing corrections. Pass a "
        "domain (web, photo, art, fashion, other) for that domain's summary, and "
        "detail='full' to also get every taste cluster card in it (principles, do, avoid, "
        "tokens). When you know what you are designing, pass brief (a sentence or two about "
        "the project, e.g. 'landing page for a climbing gym, playful and loud'): you get the "
        "threads with concrete moves to carry into the project whatever its medium, then "
        "every taste cluster ranked by fit to it with the best-fitting cards in full, so you "
        "know which of the user's several tastes to follow. Text only."
    ),
    "get_exemplars": (
        "Find reference images the user has saved that match a text query, returned as text "
        "plus thumbnails you can look at. Call this BEFORE designing or generating something "
        "visual when you need concrete references (e.g. 'dark editorial landing page', "
        "'grainy blue film photo'), and whenever a principle from get_taste is too abstract "
        "to act on. Optionally restrict to one domain; k is the number of images (default 6)."
    ),
    "critique": (
        "Score an image against the user's taste. Call this AFTER producing a screenshot, "
        "render, mockup or generated image, and before showing it as finished. Pass an "
        "absolute path to the image file. Returns a 0..1 score, the nearest taste clusters, "
        "and the best-matching cluster's principles and avoid list so you know what to fix. "
        "Revise and critique again when the score is low or a note flags a problem."
    ),
    "log_feedback": (
        "Record a visual correction or preference. Call this EVERY time the user corrects, "
        "rejects or praises a visual choice ('too much padding', 'never use gradients', "
        "'yes, this one'), without being asked. verdict is 'like', 'dislike' or 'rule' "
        "('rule' for a standing instruction that should always apply). note is the "
        "correction in the user's own words. Attach image_path (absolute path, e.g. the "
        "screenshot being judged) when there is one, plus the domain and a short context "
        "(project or task). Logged feedback is folded into the profile on the next distill."
    ),
}

SERVER_INSTRUCTIONS = (
    "The user's visual taste profile. Call get_taste before any visual decision, "
    "get_exemplars when you need reference images, critique after producing a screenshot "
    "or image, and log_feedback every time the user corrects a visual choice."
)


class ToolInputError(ValueError):
    """Bad arguments from the caller. The message is safe to show to the agent."""


# ---------------------------------------------------------------------------
# Embedder (lazy)
# ---------------------------------------------------------------------------

_EMBEDDERS: dict[tuple[str, str, str], Embedder] = {}


def _get_embedder(settings: Settings) -> Embedder:
    """Build the embedder. Module B's factory when present, the fake when asked for."""
    try:
        from taste.embed import get_embedder
    except ImportError:
        if settings.embedder == "fake":
            return FakeEmbedder()
        raise ToolInputError(
            f"Embedder '{settings.embedder}' is not available in this install "
            "(taste.embed.get_embedder is missing)."
        ) from None
    return get_embedder(settings)


def embedder_for(store: Store) -> Embedder:
    """Cached embedder for a home. Nothing is loaded until the first call."""
    settings = load_settings(store.home)
    key = (str(store.home), settings.embedder, settings.embed_model)
    if key not in _EMBEDDERS:
        _EMBEDDERS[key] = _get_embedder(settings)
    return _EMBEDDERS[key]


# ---------------------------------------------------------------------------
# Build state and friendly messages
# ---------------------------------------------------------------------------


def _has_embeddings(store: Store) -> bool:
    return any((store.home / "embeddings").glob("*/vectors.npy"))


def next_command(store: Store) -> str:
    """The `taste` command that moves this home one step closer to a built profile."""
    if not store.pins_path.exists():
        step = "taste ingest"
    elif not _has_embeddings(store):
        step = "taste embed"
    elif not store.clusters_path.exists():
        step = "taste cluster"
    elif (store.profile_dir / "profile.json").exists():
        step = "taste distill --render-only"
    else:
        step = "taste distill"
    return f"{step} --home {store.home}"


def not_built_message(store: Store, missing: str) -> str:
    return (
        f"The taste profile is not built yet ({missing} is missing). "
        f"Ask the user to run `{next_command(store)}` in a terminal. "
        f"Full pipeline: {PIPELINE}."
    )


def _check_domain(domain: str | None) -> str | None:
    if domain is None or not str(domain).strip():
        return None
    d = str(domain).strip().lower()
    if d not in DOMAINS:
        raise ToolInputError(f"Unknown domain '{domain}'. Use one of: {', '.join(DOMAINS)}.")
    return d


# ---------------------------------------------------------------------------
# get_taste
# ---------------------------------------------------------------------------


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def render_domain(s: DomainSummary) -> str:
    """Fallback renderer for when module C's markdown file is absent."""
    parts = [f"# {s.domain}", s.summary]
    if s.principles:
        parts += ["## Principles", _bullets(s.principles)]
    if s.cluster_ids:
        parts += ["## Clusters", _bullets(s.cluster_ids)]
    return "\n\n".join(parts)


def render_card(c: ClusterCard) -> str:
    """Fallback renderer for when module C's markdown file is absent."""
    parts = [f"## {c.name} ({c.cluster_id})", c.one_liner]
    for title, items in (("Principles", c.principles), ("Do", c.do), ("Avoid", c.avoid)):
        if items:
            parts += [f"### {title}", _bullets(items)]
    if c.contrast_note:
        parts += ["### Contrast", c.contrast_note]
    if c.tokens:
        tokens = {k: v for k, v in c.tokens.model_dump().items() if v}
        if tokens:
            parts += ["### Tokens", _bullets([f"{k}: {v}" for k, v in tokens.items()])]
    return "\n\n".join(parts)


def _domain_text(store: Store, domain: str) -> tuple[str | None, DomainSummary | None]:
    summary = next((s for s in store.read_domain_summaries() if s.domain == domain), None)
    md = store.profile_dir / "domains" / f"{domain}.md"
    if md.exists():
        return md.read_text(), summary
    return (render_domain(summary) if summary else None), summary


def _card_text(store: Store, card_id: str, cards: dict[str, ClusterCard]) -> str | None:
    md = store.profile_dir / "clusters" / f"{card_id}.md"
    if md.exists():
        return md.read_text()
    return render_card(cards[card_id]) if card_id in cards else None


def get_taste(store: Store, domain: str | None = None, detail: str = "summary",
              brief: str | None = None, embedder: Embedder | None = None) -> str:
    """TASTE.md, a domain file, or a domain file plus its cluster cards. With `brief`, the
    clusters ranked by how well they fit that project description instead."""
    if detail not in DETAILS:
        raise ToolInputError(f"detail must be one of: {', '.join(DETAILS)}.")
    domain = _check_domain(domain)
    if brief is not None and brief.strip():
        return taste_for_brief(store, brief.strip(), domain, embedder)

    if domain is None:
        path = store.profile_dir / "TASTE.md"
        return path.read_text() if path.exists() else not_built_message(store, "profile/TASTE.md")

    text, summary = _domain_text(store, domain)
    if text is None:
        built = sorted({s.domain for s in store.read_domain_summaries()}
                       | {p.stem for p in (store.profile_dir / "domains").glob("*.md")})
        if not built:
            return not_built_message(store, f"profile/domains/{domain}.md")
        return (f"There is no '{domain}' taste profile: no boards are mapped to that domain. "
                f"Available domains: {', '.join(built)}.")
    if detail == "summary":
        return text

    cards = {c.cluster_id: c for c in store.read_cards()}
    ids = list(summary.cluster_ids) if summary else []
    ids += [cid for cid, c in cards.items() if c.domain == domain and cid not in ids]
    sections = [t for cid in ids if (t := _card_text(store, cid, cards))]
    if not sections:
        return text + "\n\n(No cluster cards have been distilled for this domain yet.)"
    return "\n\n---\n\n".join([text.rstrip(), *[s.rstrip() for s in sections]]) + "\n"


def rank_clusters(store: Store, brief: str, domain: str | None = None,
                  embedder: Embedder | None = None) -> list[tuple[ClusterRecord, float]]:
    """Clusters by cosine similarity between the brief (text tower) and their centroids."""
    import numpy as np

    clusters = [c for c in store.read_clusters() if domain is None or c.domain == domain]
    if not clusters:
        return []
    embedder = embedder or embedder_for(store)
    query = np.asarray(embedder.embed_texts([brief])[0], dtype=np.float32)
    cents = np.asarray([c.centroid for c in clusters], dtype=np.float32)
    if cents.shape[1] != query.shape[0]:
        raise ToolInputError("clusters.json was built with a different embedding model. "
                             f"Ask the user to run `taste cluster --home {store.home}`.")
    cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12)
    sims = cents @ (query / max(float(np.linalg.norm(query)), 1e-12))
    order = sorted(range(len(clusters)), key=lambda i: (-float(sims[i]), clusters[i].id))
    return [(clusters[i], float(sims[i])) for i in order]


def read_threads(store: Store) -> list[dict]:
    import json

    path = store.profile_dir / THREADS_FILE
    return json.loads(path.read_text()) if path.exists() else []


def threads_for_brief(threads: list[dict]) -> str | None:
    """Signature threads, widest first, with their web moves; common ones named only."""
    signature = [t for t in threads if t.get("distinctiveness", 0) >= THREAD_SIGNATURE_MIN]
    common = [t for t in threads if t.get("distinctiveness", 0) < THREAD_SIGNATURE_MIN]
    if not signature and not common:
        return None
    parts = ["## Threads to carry over\n\nThese qualities recur across the user's boards "
             "whatever the subject (clothes, rooms, objects, drawings, software), so they hold "
             "for this brief too. Translate each into the brief's medium; where two pull "
             "against each other, the best-fitting clusters below say which way to lean."]
    for t in signature[:BRIEF_THREADS]:
        moves = "\n".join(f"- {m}" for m in t.get("for_web", [])[:BRIEF_THREAD_MOVES])
        avoid = "; ".join(a.rstrip(". ") for a in t.get("avoid", [])[:BRIEF_THREAD_AVOID])
        parts.append(f"### {t['name']} ({t['spread']:g} boards)\n\n{t['one_liner']}\n\n"
                     f"On the web:\n{moves}\n\nAvoid: {avoid}.")
    rest = [t["name"] for t in signature[BRIEF_THREADS:]]
    if rest:
        parts.append("More threads in profile/threads.md: " + ", ".join(rest) + ".")
    if common:
        parts.append("Common ground, not a signature: "
                     + ", ".join(t["name"] for t in common) + ".")
    return "\n\n".join(parts)


def brief_card(text: str) -> str:
    """A card's markdown without the sections an agent cannot act on (images, pin ids)."""
    keep, out = True, []
    for line in text.splitlines():
        if line.startswith("## "):
            keep = not line.startswith(BRIEF_CARD_DROP)
        if keep:
            out.append(line)
    return "\n".join(out).rstrip() + "\n"


def taste_for_brief(store: Store, brief: str, domain: str | None = None,
                    embedder: Embedder | None = None) -> str:
    """Standing rules, every cluster ranked against the brief, and the top cards in full."""
    if not store.clusters_path.exists():
        return not_built_message(store, "clusters.json")
    try:
        ranked = rank_clusters(store, brief, domain, embedder)
    except RuntimeError as exc:  # the embedder cannot load (missing extra, no weights)
        return f"Could not rank clusters for the brief: {exc}"
    if not ranked:
        where = f"the '{domain}' domain" if domain else "this profile"
        return f"There are no taste clusters in {where} yet."
    cards = {c.cluster_id: c for c in store.read_cards()}
    lines = [f"# Taste for this brief\n\nBrief: {brief}"]
    profile = store.read_profile()
    if profile and profile.rules:
        lines.append("## Standing corrections (always apply)\n\n" + _bullets(profile.rules))
    threads = threads_for_brief(read_threads(store))
    if threads:
        lines.append(threads)
    table = []
    for n, (c, sim) in enumerate(ranked[:BRIEF_RANKED_SHOWN], 1):
        card = cards.get(c.id)
        label = f"{card.name}: {card.one_liner}" if card else f"boards: {', '.join(c.top_boards)}"
        table.append(f"{n}. {c.id} ({c.domain}, {c.size} pins, fit {sim:.3f}) {label}")
    if len(ranked) > BRIEF_RANKED_SHOWN:
        table.append(f"({len(ranked) - BRIEF_RANKED_SHOWN} weaker fits not listed.)")
    lines.append("## Clusters, best fit first\n\n" + "\n".join(table) + "\n\nFit is the "
                 "cosine similarity between the brief and the cluster's images; compare "
                 "values, not their size. Lead with the first cluster and borrow from the "
                 "next ones only where they agree with it.")
    top = [brief_card(t) for c, _ in ranked[:BRIEF_TOP_CARDS]
           if (t := _card_text(store, c.id, cards))]
    if top:
        lines.append("## Best-fitting cards")
        lines += [t.rstrip() for t in top]
    else:
        lines.append("(No cluster cards have been distilled yet; run `taste distill`.)")
    return "\n\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# get_exemplars
# ---------------------------------------------------------------------------


@dataclass
class Thumbnail:
    pin_id: str
    data: bytes  # JPEG, at most THUMB_MAX_PX on the long edge
    format: str = "jpeg"


@dataclass
class ExemplarsReply:
    text: str
    hits: list[ExemplarHit] = field(default_factory=list)
    thumbnails: list[Thumbnail] = field(default_factory=list)


def make_thumbnail(path: Path, max_px: int = THUMB_MAX_PX) -> bytes:
    """JPEG bytes, long edge at most max_px."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def get_exemplars(store: Store, query: str, domain: str | None = None, k: int = 6,
                  embedder: Embedder | None = None) -> ExemplarsReply:
    """Text-to-image search over the user's pins, with thumbnails."""
    if not query or not query.strip():
        raise ToolInputError("query must not be empty.")
    domain = _check_domain(domain)
    k = max(1, min(int(k), MAX_K))
    if not store.pins_path.exists():
        return ExemplarsReply(not_built_message(store, "pins.jsonl"))

    try:
        hits = scoring_api.search(store, embedder or embedder_for(store), query.strip(),
                                  domain=domain, k=k)
    except NotImplementedError:
        return ExemplarsReply("Search is not available in this install of taste.")
    except RuntimeError as exc:  # embeddings or clusters not built yet
        return ExemplarsReply(f"{exc} Run `taste embed` and `taste cluster` first.")
    hits = list(hits)[:k]
    if not hits:
        if not _has_embeddings(store):
            return ExemplarsReply(not_built_message(store, "embeddings/"))
        scope = f" in domain '{domain}'" if domain else ""
        return ExemplarsReply(f"No saved images matched '{query}'{scope}.")

    names = {c.cluster_id: c.name for c in store.read_cards()}
    lines = [f"{len(hits)} saved reference image(s) for '{query}', best first. "
             "Thumbnails follow in the same order."]
    thumbs: list[Thumbnail] = []
    for i, h in enumerate(hits, 1):
        bits = [f"{i}. {h.pin_id}", f"similarity {h.similarity:.2f}"]
        if h.cluster_id:
            name = names.get(h.cluster_id)
            bits.append(f"cluster {h.cluster_id}" + (f" ({name})" if name else ""))
        if h.board:
            bits.append(f"board {h.board}")
        if h.link:
            bits.append(h.link)
        try:
            thumbs.append(Thumbnail(h.pin_id, make_thumbnail(store.abs(h.image_path))))
        except (OSError, ValueError):
            bits.append("(image file unavailable)")
        lines.append(" | ".join(bits))
    return ExemplarsReply("\n".join(lines), hits, thumbs)


# ---------------------------------------------------------------------------
# critique
# ---------------------------------------------------------------------------


@dataclass
class CritiqueReply:
    text: str
    result: CritiqueResult | None = None
    card: ClusterCard | None = None

    def to_dict(self) -> dict:
        card = self.card
        return {
            "result": self.result.model_dump(mode="json") if self.result else None,
            "matched_cluster": {
                "cluster_id": card.cluster_id,
                "name": card.name,
                "one_liner": card.one_liner,
                "principles": card.principles,
                "avoid": card.avoid,
            } if card else None,
        }


def _image_file(image_path: str | Path) -> Path:
    """Resolve and verify that the path is a readable image."""
    from PIL import Image

    path = Path(str(image_path)).expanduser()
    if not path.is_file():
        raise ToolInputError(f"Image not found: {path}. Pass an absolute path to an image file.")
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception:
        raise ToolInputError(f"Not a readable image file: {path}.") from None
    return path.resolve()


def critique(store: Store, image_path: str | Path,
             embedder: Embedder | None = None) -> CritiqueReply:
    """Score an image and attach the best-matching cluster card's principles and avoid list."""
    path = _image_file(image_path)
    if not store.clusters_path.exists():
        return CritiqueReply(not_built_message(store, "clusters.json"))
    try:
        result = scoring_api.critique(store, embedder or embedder_for(store), path)
    except NotImplementedError:
        return CritiqueReply("Critique is not available in this install of taste.")
    except (RuntimeError, FileNotFoundError) as exc:  # profile not built yet, or image gone
        return CritiqueReply(f"{exc} Run `taste embed` and `taste cluster` first.")

    cards = {c.cluster_id: c for c in store.read_cards()}
    best = result.nearest[0] if result.nearest else None
    card = cards.get(best.cluster_id) if best else None

    lines = [f"Taste score: {result.score:.2f} (0 = off-taste, 1 = as on-taste as the user's "
             f"own saved images). Raw similarity {result.raw_similarity:.2f}."]
    if result.domain:
        lines.append(f"Closest domain: {result.domain}.")
    if result.nearest:
        lines.append("Nearest taste clusters:")
        for m in result.nearest:
            name = m.name or (cards[m.cluster_id].name if m.cluster_id in cards else None)
            lines.append(f"- {m.cluster_id}" + (f" ({name})" if name else "")
                         + f": similarity {m.similarity:.2f}")
    if result.nearest_anti:
        lines.append("Similarity to disliked references: "
                     + ", ".join(f"{s:.2f}" for s in result.nearest_anti) + ".")
    if result.notes:
        lines += ["Notes:", _bullets(result.notes)]
    if card:
        lines += ["", f"Best match: {card.name} ({card.cluster_id}). {card.one_liner}"]
        if card.principles:
            lines += ["Principles to hold to:", _bullets(card.principles)]
        if card.avoid:
            lines += ["Avoid:", _bullets(card.avoid)]
    elif best:
        lines += ["", f"Cluster {best.cluster_id} has no distilled card yet; ask the user to "
                      f"run `taste distill --home {store.home}` for principles."]
    return CritiqueReply("\n".join(lines), result, card)


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------


def log_feedback(store: Store, verdict: str, note: str, image_path: str | Path | None = None,
                 domain: str | None = None, context: str | None = None) -> Feedback:
    """Validate, copy the image into feedback/, append to profile/feedback.jsonl."""
    verdict = (verdict or "").strip().lower()
    if verdict not in VERDICTS:
        raise ToolInputError(f"verdict must be one of: {', '.join(VERDICTS)}.")
    note = (note or "").strip()
    if not note:
        raise ToolInputError("note must not be empty: say what was right or wrong.")
    domain = _check_domain(domain)
    context = (context or "").strip() or None

    fb_id = uuid.uuid4().hex[:12]
    rel: str | None = None
    if image_path is not None and str(image_path).strip():
        src = _image_file(image_path)
        store.feedback_dir.mkdir(parents=True, exist_ok=True)
        dest = store.feedback_dir / f"{fb_id}{src.suffix.lower() or '.img'}"
        shutil.copy2(src, dest)
        rel = store.rel(dest)

    fb = Feedback(id=fb_id, ts=datetime.now(timezone.utc), verdict=verdict, note=note,
                  image_path=rel, domain=domain, context=context)
    store.append_feedback(fb)
    return fb


def feedback_confirmation(store: Store, fb: Feedback) -> str:
    bits = [f"Logged {fb.verdict} feedback {fb.id}"]
    if fb.domain:
        bits.append(f"domain {fb.domain}")
    if fb.image_path:
        bits.append(f"image saved to {fb.image_path}")
    return (", ".join(bits) + f". It applies to the profile after the next "
            f"`taste distill --home {store.home}`.")


def list_feedback(store: Store) -> list[Feedback]:
    return store.read_feedback()
