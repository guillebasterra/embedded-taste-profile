"""Markdown renderers. Pure functions: data in, markdown string out. No I/O, no VLM.

Paths inside the markdown are relative to `profile/` for TASTE.md and DESIGN.md, and relative
to the file's own folder for cluster and domain pages, so the folder can be copied anywhere.
"""

from __future__ import annotations

import json
import re

from taste.models import ClusterCard, ClusterRecord, DomainSummary, TasteProfile, WebTokens

MCP_TOOLS = ("get_taste", "get_exemplars", "critique", "log_feedback")
TASTE_MD_MAX_LINES = 80
MAX_INVARIANTS_SHOWN = 10
MAX_THREADS_SHOWN = 10
SIGNATURE_MIN = 3  # thread distinctiveness at or above this is a signature, below is common
MAX_RULES_SHOWN = 20
MAX_SUMMARY_CHARS = 2400  # one paragraph; the line budget is what keeps TASTE.md short

_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_DIMENSION = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:px|rem|em))\b")
_PLAIN_KEY = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
_FONT_ROLES = ("display", "body", "accent")


def _flat(text: str) -> str:
    return " ".join(str(text).split())


def _first_sentence(text: str, limit: int = 220) -> str:
    flat = _flat(text)
    m = re.search(r"(?<=[.!?])\s", flat)
    out = flat[: m.start()] if m else flat
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


def _bullets(items: list[str]) -> list[str]:
    return [f"- {_flat(i)}" for i in items if _flat(i)]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


# ---------------------------------------------------------------------------
# YAML (a small, strict emitter: nested maps, string/number scalars, flat lists)
# ---------------------------------------------------------------------------

def _yaml_key(key: str) -> str:
    return key if _PLAIN_KEY.match(key) else json.dumps(key, ensure_ascii=False)


def _yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)  # JSON strings are valid YAML


def to_yaml(data: dict, indent: int = 0) -> str:
    """Emit a nested dict as block YAML. Strings are always double-quoted."""
    pad = "  " * indent
    lines: list[str] = []
    for key, value in data.items():
        k = _yaml_key(str(key))
        if isinstance(value, dict):
            if not value:
                continue
            lines.append(f"{pad}{k}:")
            lines.append(to_yaml(value, indent + 1))
        elif isinstance(value, (list, tuple)):
            if not value:
                continue
            lines.append(f"{pad}{k}: [{', '.join(_yaml_scalar(v) for v in value)}]")
        elif value is not None:
            lines.append(f"{pad}{k}: {_yaml_scalar(value)}")
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# Cluster and domain pages
# ---------------------------------------------------------------------------

def render_tokens_md(tokens: WebTokens) -> list[str]:
    lines: list[str] = []
    if tokens.fonts:
        lines.append(f"- **Fonts:** {', '.join(tokens.fonts)}")
    if tokens.type_scale:
        lines.append(f"- **Type scale:** {_flat(tokens.type_scale)}")
    if tokens.palette:
        swatches = ", ".join(f"{role} `{value}`" for role, value in tokens.palette.items())
        lines.append(f"- **Palette:** {swatches}")
    for label, value in (("Spacing", tokens.spacing), ("Radius", tokens.radius),
                         ("Layout", tokens.layout), ("Imagery", tokens.imagery),
                         ("Motion", tokens.motion)):
        if value:
            lines.append(f"- **{label}:** {_flat(value)}")
    return lines


def render_cluster_md(card: ClusterCard, record: ClusterRecord | None = None,
                      neighbor_names: dict[str, str] | None = None) -> str:
    """profile/clusters/<id>.md"""
    neighbor_names = neighbor_names or {}
    meta = [f"Domain: [{card.domain}](../domains/{card.domain}.md)"]
    if record is not None:
        meta.append(f"{record.size} pins")
        meta.append(f"cohesion {record.cohesion:.2f}")
        if record.top_boards:
            meta.append("boards: " + ", ".join(record.top_boards))
    out = [f"# {_flat(card.name)} (`{card.cluster_id}`)", "", f"> {_flat(card.one_liner)}", "",
           " · ".join(meta), ""]
    for title, items in (("Principles", card.principles), ("Do", card.do),
                         ("Avoid", card.avoid)):
        out += [f"## {title}", "", *_bullets(items), ""]
    out += ["## What sets it apart", "", _flat(card.contrast_note), ""]
    if card.tokens is not None:
        token_lines = render_tokens_md(card.tokens)
        if token_lines:
            out += ["## Tokens", "", *token_lines, ""]
    if record is not None and record.palette:
        out += ["## Measured palette", "",
                "Dominant colors of the exemplars (k-means): "
                + " ".join(f"`{c}`" for c in record.palette), ""]
    if record is not None and record.neighbor_cluster_ids:
        out += ["## Neighbors", ""]
        for nid in record.neighbor_cluster_ids:
            name = neighbor_names.get(nid)
            label = f"{name} (`{nid}`)" if name else f"`{nid}`"
            out.append(f"- [{label}]({nid}.md)")
        out.append("")
    if card.exemplar_pin_ids:
        out += ["## Exemplars", ""]
        out.append(" ".join(
            f"![{card.cluster_id} exemplar {n}](../exemplars/{card.cluster_id}/{n}.jpg)"
            for n in range(1, len(card.exemplar_pin_ids) + 1)))
        out += ["", "Pins, in the same order: "
                + ", ".join(f"`{p}`" for p in card.exemplar_pin_ids), ""]
    return "\n".join(out).rstrip() + "\n"


def render_domain_md(summary: DomainSummary, cards: list[ClusterCard]) -> str:
    """profile/domains/<domain>.md"""
    by_id = {c.cluster_id: c for c in cards}
    out = [f"# Taste: {summary.domain}", "", summary.summary.strip(), "",
           "## Principles", "", *_bullets(summary.principles), "", "## Clusters", ""]
    for cid in summary.cluster_ids:
        card = by_id.get(cid)
        if card is None:
            out.append(f"- `{cid}`")
            continue
        out.append(f"- [{_flat(card.name)}](../clusters/{cid}.md) (`{cid}`): "
                   f"{_flat(card.one_liner)}")
    out.append("")
    for cid in summary.cluster_ids:
        card = by_id.get(cid)
        if card is None:
            continue
        out += [f"### {_flat(card.name)} (`{cid}`)", "",
                f"**Apart from the rest:** {_flat(card.contrast_note)}", "",
                "**Do:** " + "; ".join(_flat(d) for d in card.do), "",
                "**Avoid:** " + "; ".join(_flat(a) for a in card.avoid), ""]
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# TASTE.md
# ---------------------------------------------------------------------------

def render_threads_section(threads: list[dict]) -> list[str]:
    if not threads:
        return []
    signature = [t for t in threads if t["distinctiveness"] >= SIGNATURE_MIN]
    common = [t for t in threads if t["distinctiveness"] < SIGNATURE_MIN]
    out = ["## Threads", "",
           "Qualities that recur across my boards whatever the subject, widest first. Carry "
           "them into any medium. Full cards, with concrete moves for the web: "
           "[threads.md](threads.md).", ""]
    for t in signature[:MAX_THREADS_SHOWN]:
        out.append(f"- **{_flat(t['name'])}** ({t['spread']:g} boards): "
                   f"{_first_sentence(t['one_liner'])}")
    if common:
        out.append("- Common ground, not a signature: "
                   + ", ".join(_flat(t["name"]) for t in common) + ".")
    return out + [""]


def render_taste_md(profile: TasteProfile, cards: list[ClusterCard] | None = None,
                    threads: list[dict] | None = None) -> str:
    """profile/TASTE.md: the entry point. Kept under TASTE_MD_MAX_LINES lines."""
    cards = cards or []
    names = {c.cluster_id: _flat(c.name) for c in cards}
    summary = _flat(profile.summary)
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[: MAX_SUMMARY_CHARS - 1].rstrip() + "…"

    out = ["# TASTE", "",
           "My visual taste, distilled from the images I save. Read this before any visual "
           "decision made on my behalf.", "",
           "## Summary", "", summary, "",
           *render_threads_section(threads or []),
           "## Invariants", "",
           "Hold in every domain. Each one lists the clusters it rests on.", ""]
    if profile.invariants:
        for inv in profile.invariants[:MAX_INVARIANTS_SHOWN]:
            evidence = ", ".join(f"`{c}`" for c in inv.evidence_cluster_ids)
            tail = f" (evidence: {evidence})" if evidence else ""
            out.append(f"- **{_flat(inv.name)}:** {_flat(inv.statement)}{tail}")
        extra = len(profile.invariants) - MAX_INVARIANTS_SHOWN
        if extra > 0:
            out.append(f"- …and {extra} more in `profile.json`.")
    else:
        out.append("- None established yet: nothing recurs across two or more domains.")

    out += ["", "## Standing corrections", "",
            "Rules I stated directly. They override everything else in this profile.", ""]
    if profile.rules:
        out += [f"- {_flat(r)}" for r in profile.rules[:MAX_RULES_SHOWN]]
        extra = len(profile.rules) - MAX_RULES_SHOWN
        if extra > 0:
            out.append(f"- …and {extra} more in `profile.json`.")
    else:
        out.append("- None yet. Log one with `log_feedback` (verdict `rule`).")

    out += ["", "## Domains", ""]
    if profile.domains:
        for d in profile.domains:
            clusters = ", ".join(
                f"[{names.get(cid, cid)}](clusters/{cid}.md)" for cid in d.cluster_ids)
            line = f"- **[{d.domain}](domains/{d.domain}.md):** {_first_sentence(d.summary)}"
            if clusters:
                line += f" Clusters: {clusters}."
            out.append(line)
    else:
        out.append("- No domains distilled yet.")

    out += ["", "## Query deeper", "",
            "Web tokens live in [DESIGN.md](DESIGN.md). With the `taste` MCP server connected:",
            "",
            "- `get_taste(domain, detail, brief)`: a domain file, with `detail=\"full\"` its "
            "cluster cards, with `brief` the threads to carry over plus the clusters ranked "
            "for a project. Call before visual decisions.",
            "- `get_exemplars(query, domain, k)`: reference images for a text query. Look at "
            "them instead of guessing from words.",
            "- `critique(image_path)`: scores a screenshot or draft against the clusters and "
            "says what to fix. Call after producing anything visual.",
            "- `log_feedback(verdict, note, ...)`: records a like, dislike or rule. Call "
            "every time I correct a visual choice.",
            "",
            f"Generated {profile.generated_at:%Y-%m-%d} · VLM `{profile.vlm_model}` · "
            f"embeddings `{profile.embed_model}`"]
    return "\n".join(out[:TASTE_MD_MAX_LINES]).rstrip() + "\n"


# ---------------------------------------------------------------------------
# DESIGN.md (Google Labs DESIGN.md format: YAML front matter tokens + markdown sections)
# ---------------------------------------------------------------------------

def _norm(value: str) -> str:
    return _flat(value).lower()


def _clean_palette(palette: dict[str, str]) -> dict[str, str]:
    out = {}
    for role, value in palette.items():
        v = str(value).strip()
        if _HEX.match(v):
            out[slugify(str(role))] = v.upper()
    return out


def _dimension(text: str | None) -> str | None:
    m = _DIMENSION.search(text or "")
    return m.group(1) if m else None


def _variant_keys(cards: list[ClusterCard]) -> dict[str, str]:
    """cluster id -> unique variant key (slug of the name, id appended on collision)."""
    slugs = [slugify(c.name) for c in cards]
    return {c.cluster_id: (s if slugs.count(s) == 1 else f"{s}-{c.cluster_id}")
            for c, s in zip(cards, slugs)}


def merge_web_tokens(cards: list[ClusterCard]) -> dict:
    """Merge the tokens of web cards into shared values and named variants.

    A value is shared when every card that carries tokens states it identically (compared
    case-insensitively). Everything else stays with its cluster as a variant. With a single
    card, everything is shared.

    Returns {"shared": {...}, "variants": {key: {"cluster_id", "name", ...fields}}} where the
    field dicts hold `colors` (role -> hex), `fonts` (list) and the free-text fields.
    """
    with_tokens = [c for c in cards if c.tokens is not None]
    keys = _variant_keys(with_tokens)
    text_fields = ("type_scale", "spacing", "radius", "layout", "imagery", "motion")

    per: dict[str, dict] = {}
    for c in with_tokens:
        t = c.tokens
        entry: dict = {"colors": _clean_palette(t.palette),
                       "fonts": [_flat(f) for f in t.fonts if _flat(f)]}
        for f in text_fields:
            if getattr(t, f):
                entry[f] = _flat(getattr(t, f))
        per[c.cluster_id] = entry

    shared: dict = {"colors": {}, "fonts": []}
    if per:
        entries = list(per.values())
        first = entries[0]
        for role, value in first["colors"].items():
            if all(e["colors"].get(role) == value for e in entries):
                shared["colors"][role] = value
        if all([_norm(f) for f in e["fonts"]] == [_norm(f) for f in first["fonts"]]
               for e in entries):
            shared["fonts"] = list(first["fonts"])
        for f in text_fields:
            if f in first and all(_norm(e.get(f, "")) == _norm(first[f]) for e in entries):
                shared[f] = first[f]

    variants: dict[str, dict] = {}
    if len(per) > 1:
        for c in with_tokens:
            e = per[c.cluster_id]
            v: dict = {"cluster_id": c.cluster_id, "name": _flat(c.name)}
            colors = {r: x for r, x in e["colors"].items() if r not in shared["colors"]}
            if colors:
                v["colors"] = colors
            if e["fonts"] and not shared["fonts"]:
                v["fonts"] = e["fonts"]
            for f in text_fields:
                if f in e and f not in shared:
                    v[f] = e[f]
            variants[keys[c.cluster_id]] = v
    return {"shared": shared, "variants": variants}


def _typography_tokens(fonts: list[str]) -> dict:
    out = {}
    for i, font in enumerate(fonts):
        role = _FONT_ROLES[i] if i < len(_FONT_ROLES) else f"font-{i + 1}"
        out[role] = {"fontFamily": font}
    return out


def design_front_matter(merged: dict, name: str, description: str) -> dict:
    """Token tree for the YAML front matter. Variants become nested groups per token type."""
    shared, variants = merged["shared"], merged["variants"]
    fm: dict = {"version": "alpha", "name": name, "description": description}
    colors: dict = dict(shared["colors"])
    typography: dict = _typography_tokens(shared["fonts"])
    rounded: dict = {}
    spacing: dict = {}
    if _dimension(shared.get("radius")):
        rounded["default"] = _dimension(shared.get("radius"))
    if _dimension(shared.get("spacing")):
        spacing["base"] = _dimension(shared.get("spacing"))
    for key, v in variants.items():
        if v.get("colors"):
            colors[key] = dict(v["colors"])
        if v.get("fonts"):
            typography[key] = _typography_tokens(v["fonts"])
        if _dimension(v.get("radius")):
            rounded[key] = {"default": _dimension(v.get("radius"))}
        if _dimension(v.get("spacing")):
            spacing[key] = {"base": _dimension(v.get("spacing"))}
    for section, value in (("colors", colors), ("typography", typography),
                           ("rounded", rounded), ("spacing", spacing)):
        if value:
            fm[section] = value
    return fm


def _design_section(title: str, field: str | None, merged: dict,
                    lead: list[str] | None = None) -> list[str]:
    """One markdown section: shared prose, then one bullet per variant that differs."""
    shared, variants = merged["shared"], merged["variants"]
    body: list[str] = list(lead or [])
    if field and shared.get(field):
        body.append(shared[field])
    for key, v in variants.items():
        if field and v.get(field):
            body.append(f"- **{v['name']}** (`{key}`): {v[field]}")
    if not body:
        return []
    out = [f"## {title}", ""]
    for line in body:
        out.append(line)
        if not line.startswith("- "):
            out.append("")
    if out[-1] != "":
        out.append("")
    return out


def render_design_md(web_cards: list[ClusterCard],
                     records: list[ClusterRecord] | None = None) -> str:
    """profile/DESIGN.md from the web cluster cards."""
    title = "Personal taste: web"
    if not web_cards:
        fm = {"version": "alpha", "name": title,
              "description": "Placeholder: no web clusters in the profile yet."}
        return "\n".join([
            "---", to_yaml(fm), "---", "",
            "# DESIGN", "",
            "## Overview", "",
            "There are no web clusters in this profile yet, so there are no design tokens to "
            "report. Map at least one board to the `web` domain in `boards.toml`, then run "
            "`taste embed`, `taste cluster` and `taste distill`.", ""])

    records_by_id = {r.id: r for r in (records or [])}
    merged = merge_web_tokens(web_cards)
    shared, variants = merged["shared"], merged["variants"]
    keys = _variant_keys([c for c in web_cards if c.tokens is not None])
    description = (
        f"Web design tokens distilled from {len(web_cards)} cluster(s) of saved references. "
        + ("Clusters that disagree appear as named variants: pick one per project."
           if variants else "One direction, no variants."))
    fm = design_front_matter(merged, title, description)

    out = ["---", to_yaml(fm), "---", "", "# DESIGN", "", "## Overview", ""]
    out.append("Tokens read off the websites I save, one cluster of references at a time. "
               "Values in the front matter are the ones the images support. Prose below "
               "carries the rest.")
    out.append("")
    if variants:
        out += ["The web clusters disagree on some tokens, so those tokens are grouped under "
                "a named variant (the nested groups in the front matter). Choose one variant "
                "for a project and stay inside it. Do not average variants.", ""]
    if not any(c.tokens is not None for c in web_cards):
        out += ["No tokens have been extracted yet: the cluster cards below carry the "
                "direction in prose.", ""]
    for c in web_cards:
        key = keys.get(c.cluster_id)
        tag = f" · variant `{key}`" if key and key in variants else ""
        out.append(f"- [{_flat(c.name)}](clusters/{c.cluster_id}.md) (`{c.cluster_id}`{tag}): "
                   f"{_flat(c.one_liner)}")
    out.append("")

    color_lead: list[str] = []
    if shared["colors"]:
        color_lead.append("- Shared: " + ", ".join(
            f"{role} `{value}`" for role, value in shared["colors"].items()))
    for key, v in variants.items():
        if v.get("colors"):
            color_lead.append(f"- **{v['name']}** (`{key}`): " + ", ".join(
                f"{role} `{value}`" for role, value in v["colors"].items()))
    for c in web_cards:
        rec = records_by_id.get(c.cluster_id)
        if rec is not None and rec.palette:
            color_lead.append(f"- Measured from `{c.cluster_id}` exemplars (k-means): "
                              + " ".join(f"`{p}`" for p in rec.palette))
    out += _design_section("Colors", None, merged, color_lead)

    type_lead: list[str] = []
    if shared["fonts"]:
        type_lead.append("- Shared faces (display, body, accent order): "
                         + ", ".join(shared["fonts"]))
    for key, v in variants.items():
        if v.get("fonts"):
            type_lead.append(f"- **{v['name']}** (`{key}`) faces: " + ", ".join(v["fonts"]))
    type_section = _design_section("Typography", "type_scale", merged, type_lead)
    out += type_section

    layout = _design_section("Layout", "layout", merged)
    spacing = _design_section("Layout", "spacing", merged)
    if layout and spacing:
        out += layout + ["**Spacing**", ""] + spacing[2:]
    else:
        out += layout or spacing
    out += _design_section("Shapes", "radius", merged)
    out += _design_section("Motion & Micro-interaction", "motion", merged)
    out += _design_section("Imagery", "imagery", merged)

    out += ["## Do's and Don'ts", ""]
    for c in web_cards:
        out += [f"### {_flat(c.name)} (`{c.cluster_id}`)", ""]
        out += [f"- Do: {_flat(d)}" for d in c.do]
        out += [f"- Don't: {_flat(a)}" for a in c.avoid]
        out.append("")
    return "\n".join(out).rstrip() + "\n"
