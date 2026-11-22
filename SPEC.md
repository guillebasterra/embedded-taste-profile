# taste — spec v1

One command path turns my Pinterest boards into a portable taste profile that any agent can query.

## Principles
- Embeddings are the index; text and exemplar images are the interface. Agents read principles and look at exemplars. Vectors do clustering, retrieval and scoring.
- Several tastes per domain. One vector cannot hold "brutalist sites" and "warm editorial sites" at once (the PinnerSage idea).
- Contrastive descriptions. Every cluster card says what separates it from its neighbors and from the generic version of the same thing.
- The profile is plain files. `profile/` can be committed to a context repo and read by any model.
- Corrections compound. Every visual correction I make is logged and folded back in.

## Shared contracts (do not edit in feature branches)
- `src/taste/config.py` — `Settings`, `load_settings()`, domains, Pinterest constants
- `src/taste/models.py` — `Pin`, `BoardEntry`, `ClusterRecord`, `ClusterCard`, `WebTokens`, `DomainSummary`, `Invariant`, `TasteProfile`, `Feedback`, `CritiqueResult`, `ExemplarHit`
- `src/taste/store.py` — `Store`: the on-disk layout and typed read/write. All modules talk to each other through it.
- `src/taste/interfaces.py` — `Embedder`, `VLM` protocols; `FakeEmbedder`, `FakeVLM`
- `src/taste/fixtures.py` — `make_fixture_home()` synthetic dataset
- `src/taste/cli.py` — root Typer app; each package adds commands in `<pkg>/cli.py::register(app)`
- `src/taste/scoring/api.py` — signatures of `critique()` and `search()` (B implements, D consumes)

Every CLI command takes `--home` (default from `TASTE_HOME`, else `~/.taste`).
Tests never touch the network or real model weights: use `FakeEmbedder`, `FakeVLM`, `respx`, and `make_fixture_home(tmp_path)`.

## Module A — auth + ingest (`src/taste/ingest/`)
Commands: `taste auth`, `taste ingest`, `taste domains`.

- `taste auth`
  - Authorization-code flow against `https://www.pinterest.com/oauth/` with `PINTEREST_SCOPES` and `PINTEREST_REDIRECT_URI`; random `state`, verified on return.
  - Local HTTP listener on port 8085 catches the redirect; prints the URL and tries to open the browser; `--no-browser` supported.
  - Token exchange: `POST https://api.pinterest.com/v5/oauth/token`, HTTP Basic `app_id:app_secret`, form body.
  - Tokens saved to `auth/pinterest.json` with mode 600; refreshed automatically when expired (`grant_type=refresh_token`).
  - `PINTEREST_ACCESS_TOKEN` in env short-circuits all of this.
  - Never print secrets or tokens.
- `taste ingest`
  - `--source api` (default): `GET /v5/boards` (paginate with `bookmark`, `page_size` max, `privacy=ALL`), `GET /v5/boards/{id}/sections`, `GET /v5/boards/{id}/pins`, `GET /v5/boards/{id}/sections/{sid}/pins`. Pick the largest image in `media.images`. `--board` filter (repeatable), `--limit`.
  - `--source public --url <board url>`: no login; best effort via the board RSS feed (`<board url>.rss`), clearly documented as partial (latest pins only).
  - `--source local --path <dir>`: every image under the dir; first-level subfolder name is the board; id is `local:<sha1 of bytes>[:16]`.
  - Downloads images to `images/` (concurrent, retries, skips existing, verifies with Pillow, records width/height), then `Store.upsert_pins`.
  - Handles 401 (refresh once), 429 (respect `Retry-After` / back off), and the 1,000 requests/day trial cap: stop cleanly, keep progress, say how to resume. Re-running is incremental.
- `taste domains`
  - Creates or updates `boards.toml`: every board gets a domain guess from keyword heuristics on the board name (e.g. "web", "ui", "site", "type" -> `web`; "photo", "film", "35mm" -> `photo`; "outfit", "fits", "wear", "denim" -> `fashion`; "art", "paint", "illustration", "poster" -> `art`; else `other`). Never overwrites an entry the user already has. Prints a table and tells the user to edit the file.
- Acceptance: unit tests with `respx` for OAuth exchange/refresh, pagination, 429 handling, incremental re-run; local-folder ingest end to end.

## Module B — embed + cluster + scoring (`src/taste/embed/`, `src/taste/cluster/`, `src/taste/scoring/`)
Commands: `taste embed`, `taste cluster`, `taste critique <image>`, `taste search <query>`, `taste eval`.

- `taste.embed.get_embedder(settings) -> Embedder`
  - `"fake"` -> `FakeEmbedder`.
  - `"siglip2"` -> transformers `AutoModel`/`AutoProcessor` for `settings.embed_model`, lazy import (extra `ml`), device auto (mps, cuda, cpu), batched, image and text towers, L2-normalized. Clear error if the extra is missing.
- `taste embed`: embeds every included pin with an image, incremental (only new ids), writes via `Store.write_embeddings(embedder.name, ...)`.
- `taste cluster`
  - Per domain (from `Store.pin_domains()`): agglomerative clustering on cosine distance (average linkage) with a distance threshold, `--threshold`, plus `--min-size` (default 4). Small clusters are merged into the nearest cluster when close enough, otherwise left unclustered.
  - A domain with fewer than `min-size` pins is skipped.
  - Per cluster: centroid, medoid, up to 12 exemplars nearest the centroid, cohesion, top boards, palette (k-means over downsampled pixels of the exemplars, at most 6 hex colors), up to 3 nearest neighbor clusters (same domain first).
  - Stable ids `<domain>-<nn>` ordered by size descending. Writes `clusters.json`. Prints a summary table.
- `taste.scoring.api` (signatures fixed in the stub)
  - `search(store, embedder, query, domain=None, k=8) -> list[ExemplarHit]`: text-to-image cosine search over pins.
  - `critique(store, embedder, image_path) -> CritiqueResult`: similarity to cluster centroids; `score` is the percentile of that similarity within the distribution of my own pins' similarity to their nearest centroid; `nearest_anti` from images attached to `dislike` feedback; plain-language `notes` (e.g. "closer to a disliked reference than to any cluster").
- `taste eval --distractors <dir>`: hold out 20% of pins, rebuild centroids from the rest, score held-out pins vs. distractor images, report ROC AUC and a suggested accept threshold.
- Acceptance: on the fixture home with `FakeEmbedder`, clustering recovers the 5 groups (2 web, 2 photo, 1 fashion); `search("blue")` returns blue pins; a red image scores higher than a magenta-noise image; eval AUC > 0.9 with random-noise distractors.

## Module C — distill + profile writers (`src/taste/distill/`)
Command: `taste distill`.

- `taste.distill.llm.get_vlm(settings) -> VLM`
  - `"fake"` -> `FakeVLM`.
  - `"anthropic"` -> Anthropic SDK, model `settings.vlm_model`, images sent as base64 (downscaled to at most 1024 px long edge, JPEG), structured output by forcing a single tool call whose input schema is the pydantic schema, validated, one retry on validation failure. Clear error if `ANTHROPIC_API_KEY` is missing.
- Steps
  1. Cluster cards: for each cluster, send its exemplars (up to 8) plus up to 2 exemplars from each neighbor cluster, labeled, with board names and pin titles as hints. Ask for a `ClusterCard`: name, one-liner, principles, do, avoid, and a `contrast_note` on what separates it from the neighbors and from the generic version. `tokens` only for `web`, only what the images support.
  2. Domain summaries: text-only, from that domain's cards -> `DomainSummary`.
  3. Cross-domain invariants: text-only, from all cards and summaries -> summary + `Invariant` list, each citing the clusters it rests on. Prompt must forbid invariants supported by a single domain.
  4. Feedback: `rule` feedback notes go into `TasteProfile.rules` verbatim (deduplicated); `like`/`dislike` notes are passed to steps 1-3 as extra evidence.
- Outputs (all under `profile/`)
  - `profile.json`, `clusters/<id>.json|.md`, `domains/<domain>.json|.md`
  - `exemplars/<cluster-id>/<n>.jpg` thumbnails (max 512 px) so the profile is portable without `images/`
  - `TASTE.md`: at most ~80 lines. Summary, invariants, standing corrections, one line per domain with links to the domain files, and how to query deeper (MCP tools).
  - `DESIGN.md`: web tokens merged across web clusters, following Google's open DESIGN.md format (YAML front matter tokens + markdown sections); when web clusters disagree, list them as named variants.
- Incremental: skip clusters whose card exists and whose `exemplar_pin_ids` are unchanged; `--force` redoes everything; `--only <cluster-id>`.
- Renderers are pure functions (data -> markdown) and can be re-run without the VLM: `taste distill --render-only`.
- Acceptance: with `FakeVLM` on a clustered fixture home, all files are produced, markdown renders, re-run makes zero VLM calls, feedback rules show up in `TASTE.md`.

## Module D — serve + feedback + install (`src/taste/serve/`)
Commands: `taste serve`, `taste feedback`, `taste install`.

- `taste serve`: MCP server over stdio (official `mcp` Python SDK, FastMCP). Tools:
  - `get_taste(domain: str | None, detail: "summary" | "full" = "summary")`: no domain -> `TASTE.md`; domain + summary -> the domain file; full -> domain file plus its cluster cards. Text only.
  - `get_exemplars(query: str, domain: str | None, k: int = 6)`: calls `scoring.api.search`; returns hits as text plus thumbnails as MCP image content (max 512 px).
  - `critique(image_path: str)`: calls `scoring.api.critique`; returns the result plus the matched cluster card's principles and avoid list so the agent knows what to fix.
  - `log_feedback(verdict, note, image_path?, domain?, context?)`: validates, copies the image into `feedback/`, appends via `Store.append_feedback`.
  - Tool descriptions tell the agent when to call each one (before visual decisions; after producing a screenshot; whenever I correct a visual choice).
  - The embedder loads lazily on first use so the server starts fast.
- `taste feedback <verdict> <note> [--image] [--domain] [--context]`: same thing from the shell. `taste feedback list`.
- `taste install`
  - Writes `profile/SKILL.md` (a Claude skill: when to consult the profile, the tool list, the rule "log every visual correction with log_feedback") and `profile/CLAUDE.snippet.md` (a short block to paste or import into a `CLAUDE.md` / `AGENTS.md`).
  - `--print-mcp-config` prints the JSON block for Claude Code / Claude Desktop (`command: taste`, `args: [serve, --home, <home>]`).
  - `--export <dir>` copies `profile/` to a context repo path (e.g. `~/me/vault/taste`), no images besides thumbnails.
- Acceptance: tool functions tested directly with a fixture home (scoring mocked where B is not merged yet), feedback round-trip, install outputs exist and mention every tool.

## End-to-end (orchestrator)
```
TASTE_EMBEDDER=fake TASTE_VLM=fake taste ingest --source local --path <fixture>
taste domains && taste embed && taste cluster && taste distill && taste install
taste critique <image> && taste search "blue"
```
