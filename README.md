# taste

Turn my Pinterest boards into a visual taste profile that AI agents read before making design decisions for me.

## At a glance

```
Pinterest boards ──▶ images on disk ──▶ what each image looks like ──▶ taste profile ──▶ agents
   (gallery-dl)        (~1,300 pins)      (pixels + Claude reading)     (TASTE.md)      (MCP server)
```

- **Ingest** my boards (images + titles) into `~/.taste`.
- **Read every pin for how it looks, not what it shows:** palette, light, texture, composition, finish, era. Measured locally, read by Claude Haiku.
- **Find threads:** bundles of qualities that recur across unrelated boards (film grain in clothes, rooms and surf photos; flat-ink print logic in web, sketchbooks and software). These are the taste that holds in any medium.
- **Cluster** each domain (web, art, fashion, other) into concrete taste groups with SigLIP embeddings, and have Claude write a card for each: principles, do, avoid, tokens.
- **Serve** it to Claude Code over MCP. `get_taste(brief="portfolio site...")` returns the threads to carry over plus the clusters that best fit the project; `critique` scores a screenshot against my taste; `log_feedback` keeps my corrections as standing rules.

Output lives in `~/.taste/profile/`: `TASTE.md` (the entry point), `threads.md`, `DESIGN.md` (web tokens), one card per cluster, thumbnails.

`SPEC.md` is the contract, `DECISIONS.md` logs every judgment call, `NEXT.md` is the first-live-run checklist.

## Setup

```bash
uv sync --extra ml          # torch + transformers for the real SigLIP embedder
mkdir -p ~/.taste && $EDITOR ~/.taste/.env
```

`~/.taste/.env`:

```
ANTHROPIC_API_KEY=...
# only for the official Pinterest API route:
PINTEREST_ACCESS_TOKEN=...
```

## Get the pins in (pick one)

**A. gallery-dl (works today, unofficial).** Runs gallery-dl into `~/.taste/gallery-dl/` (images plus a metadata sidecar each, no videos, only new pins on reruns), then imports the folder. Public boards only, unless you add `--browser chrome` (or another browser) to use its logged-in session for secret boards.

```bash
uv run taste ingest --source gallery-dl --url https://www.pinterest.com/<username>/
```

Any other folder where each first-level subfolder is a board works with `--source local --path <dir>`.

**B. Official API.** Needs an approved developer app (website + privacy policy).

```bash
uv run taste auth            # full OAuth, includes secret boards
uv run taste ingest          # or set PINTEREST_ACCESS_TOKEN and skip auth
```

**C. One public board, no login.** `uv run taste ingest --source public --url <board url>` (latest pins only).

## Build the profile

```bash
uv run taste domains         # then edit ~/.taste/boards.toml: fix domains, exclude boards
uv run taste embed           # SigLIP 2, local; ~1 min for 1,400 pins on Apple silicon
uv run taste facets          # measure + read every pin (Claude Haiku, ~$1 for 1,300 pins)
uv run taste threads         # cross-board threads + their cards (Claude Sonnet);
                             #   contact sheet at ~/.taste/facets/threads.html
uv run taste cluster --threshold 0.35 --min-size 8   # tune against the printed medians
uv run taste distill         # cluster cards, domain summaries, TASTE.md (Claude Sonnet)
uv run taste install --export ~/me/vault/taste
```

Later, one command refreshes everything incrementally (replays the last `taste ingest`, keeps cluster ids and your tuned threshold, re-distills only changed clusters, re-exports):

```bash
uv run taste update --export ~/me/vault/taste   # --export is remembered afterwards
```

## Use it

```bash
claude mcp add -s user taste -- "$PWD/.venv/bin/taste" serve --home ~/.taste
cp ~/.taste/profile/SKILL.md ~/.claude/skills/taste/SKILL.md
uv run taste critique ./screenshot.png
uv run taste search "dense editorial grid"
uv run taste feedback add rule "Never center body text"
uv run taste eval --distractors ./some-generic-images
```

MCP tools: `get_taste`, `get_exemplars`, `critique`, `log_feedback`. The server needs the full `TASTE_HOME` (embeddings and images); the exported `profile/` folder is the read-only text and thumbnail layer.

## Tests

```bash
uv run pytest -q             # no network, no model weights
```
