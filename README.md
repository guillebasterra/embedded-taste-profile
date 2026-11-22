# taste

Turn my Pinterest boards into a portable visual taste profile any AI agent can query. `SPEC.md` is the contract, `DECISIONS.md` logs every call made along the way, `NEXT.md` is the first-live-run checklist.

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

**A. Local folder (works today).** Any folder where each first-level subfolder is a board.

```bash
# one way to fill it: gallery-dl with a logged-in browser session (unofficial; my own boards)
uvx gallery-dl --cookies-from-browser chrome "https://www.pinterest.com/<username>/"
uv run taste ingest --source local --path ./gallery-dl/pinterest/<username>
```

**B. Official API.** Needs an approved developer app (website + privacy policy).

```bash
uv run taste auth            # full OAuth, includes secret boards
uv run taste ingest          # or set PINTEREST_ACCESS_TOKEN and skip auth
```

**C. One public board, no login.** `uv run taste ingest --source public --url <board url>` (latest pins only).

## Build the profile

```bash
uv run taste domains         # then edit ~/.taste/boards.toml: fix domains, exclude boards
uv run taste embed
uv run taste cluster         # tune --threshold against the printed median distances
uv run taste distill
uv run taste install --export ~/me/vault/taste
```

## Use it

```bash
claude mcp add taste -- taste serve --home ~/.taste
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
