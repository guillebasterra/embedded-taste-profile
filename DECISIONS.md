# Decisions

Calls made without asking, newest last. One line each.

## Orchestrator
- Python + uv + Typer; storage is plain files under `TASTE_HOME` so the profile is portable and diffable.
- Default embedder `google/siglip2-base-patch16-384` (image and text towers share a space, so text-to-image retrieval works). Swappable via `TASTE_EMBED_MODEL`.
- Agglomerative clustering with a distance threshold instead of a fixed k: the number of tastes per domain should come from the data.
- Three ingest sources behind one `Pin` shape, so the pipeline runs on a local folder while Pinterest app approval is pending.
- Built in a cloud workspace that cannot reach Hugging Face, Pinterest or the Anthropic API: the real SigLIP, Pinterest and Claude paths are written against their docs and covered by mocked tests, and need a first live run on my machine.

## A — ingest
- Local ingest copies each image into `images/local-<sha1[:16]>.<ext>` so the home is self-contained and the source folder can move; byte-identical duplicates collapse to one pin.
- Local: loose files in the root get the root folder's name as board; a second-level subfolder becomes `section`; dot-folders and files Pillow cannot verify are skipped (and counted).
- Image file extension comes from the verified bytes (Pillow format), not the URL; files are written as `.part` and renamed only after verification.
- A 429 is waited out in-process only when the server asks for 120 s or less (Retry-After, else X-RateLimit-Reset, else exponential backoff, 4 retries); anything longer is treated as the daily cap and stops the run.
- Rate-limit stop exits with code 3 (not 0) so `taste ingest && taste embed` chains do not run on a half-finished pull by accident; pins and images fetched so far are saved, including the partial board.
- Resume: boards finished before a rate-limit stop are listed in `TASTE_HOME/ingest_state.json` and skipped on the next run; the file is deleted after a complete unfiltered pass, so later runs re-check every board for new pins. This file is outside the Store layout (module-private).
- Added `--max-requests` (stop before a known daily budget) and `--concurrency`; no persistent request counter, the server's 429 is the source of truth.
- `--limit` counts pins seen in this run (across boards), not new pins.
- `--board` matches board name (case-insensitive) or board id; an unknown board is an error rather than a silent no-op.
- Both the board pin listing and each section pin listing are called (per spec) and deduplicated by pin id; the section listing wins for the section name.
- Pins without `media.images` (or whose download fails) are still written to `pins.jsonl` without `image_path`; downstream ignores them via `Store.pin_domains()`, and a re-run retries the download.
- Re-runs refresh pin metadata but keep the existing image, width and height; an image already on disk is never fetched again (a corrupt one is).
- Public source: RSS thumbnails are `236x`; the pin's `image_url` is rewritten to `736x` with the original thumbnail as fallback. Feeds containing a DOCTYPE/ENTITY are rejected (stdlib XML parser, no defusedxml).
- `taste auth` binds the listener to 127.0.0.1:8085 only, silences the HTTP server's request log (the request line carries the code), and has a `--timeout` (default 300 s).
- Access tokens are refreshed 60 s before expiry; when Pinterest does not rotate the refresh token the old one is kept. With `PINTEREST_ACCESS_TOKEN` set, a 401 fails immediately (nothing to refresh with).
- Domain heuristics match whole word tokens (plus plural, and prefix/suffix for keywords of 4+ letters) instead of substrings, so "building" is not `ui` and "party" is not `art`; most hits wins, ties go web > photo > fashion > art. Keyword lists extend the spec's examples.
- `taste domains` rewrites `boards.toml` only when it adds a board, so hand edits and comments survive no-op runs (adding a board re-serializes the file and drops comments: `Store.write_boards` limitation). Boards with no pins left are kept.
- Live run: Pinterest answers 401 `{code: 3, "consumer type is not supported"}` for an app whose access level is not enough. That is not retried with a token refresh and gets its own message pointing at the developer portal and the local/public sources; other 401s now include Pinterest's message and mention `PINTEREST_ACCESS_TOKEN` (since `taste auth` is a no-op while it is set).
- Pinterest app review site lives in `docs/` (GitHub Pages from `main:/docs`): a landing page and a privacy policy that discloses the Anthropic upload of exemplar images; contact is GitHub issues, so no email is published.
- While the API app is unapproved, pins come from gallery-dl (public boards, no cookies) into `TASTE_HOME/gallery-dl/` with a download archive, then `taste ingest --source local`; kept in the home so reruns are incremental.

## B — embed / cluster / scoring
- Embedder `name` for SigLIP is the model id itself (`Store` slugs it), so switching `TASTE_EMBED_MODEL` gets its own embeddings directory.
- `get_embedder` never loads weights; torch/transformers import and the model load happen on the first embed call, so `taste cluster` and the MCP server start fast.
- SigLIP text is lowercased and padded to 64 tokens (how the text tower was trained); no prompt template is added to queries.
- `get_image_features`/`get_text_features` results are accepted as a tensor or as an output object with `pooler_output`, since this differs across transformers versions.
- `taste embed` keys on pin id only: a replaced image under the same id is not re-embedded (use `--force`). Unreadable images are skipped and reported, and the index is checkpointed every 8 chunks so an interrupted run resumes.
- Default clustering threshold is 0.35 cosine distance: pins in one domain already sit close together in SigLIP space, so a looser cut risks one cluster per domain. It is a starting guess, untested on real weights; `taste cluster` prints each domain's median pairwise distance to tune `--threshold` against.
- Clustering runs on raw cosine distance with no per-domain mean-centering: centering would spread the scale but breaks a domain that holds a single taste.
- A cluster under `--min-size` joins the nearest full-size cluster when their average-linkage distance is within 1.25x the threshold; otherwise its pins stay unclustered (absent from `clusters.json`, counted in the summary).
- Cluster ids are deterministic (size descending, ties by smallest pin id) but can shift when pins are added and sizes reorder.
- Medoid is the member with the highest mean similarity to the other members; `pin_ids` are sorted by id; centroids are rounded to 6 decimals to keep `clusters.json` small.
- Palette: exemplars downsampled to 32x32, k-means with k=6, then colors within 28 RGB units are folded together so a flat image yields one color.
- Neighbors: same-domain clusters by centroid similarity first, other domains fill the remaining slots.
- Critique reference distribution is every included, embedded pin (unclustered ones too) against its nearest centroid. My own pins therefore score uniformly 0..1, so the notes call anything at or above 0.10 on taste, 0.01 to 0.10 borderline, below that off taste.
- `critique` returns the 3 nearest clusters and up to 3 disliked-reference similarities; a note fires when a disliked reference is closer than the best cluster, or within 0.05 of it. Disliked images are embedded on the fly and cached in-process by path and mtime.
- `critique` and `search` raise `RuntimeError` with the command to run when clusters or embeddings are missing, and `FileNotFoundError` for a missing image; the CLI turns both into a one-line message and exit code 1.
- `taste eval` re-clusters the held-in 80% (it imports `taste.cluster.build_clusters`, the cluster package's public function) instead of reusing `clusters.json`, so held-out pins never shape the centroids. It never writes `clusters.json`.
- Suggested accept threshold is the cut that maximizes TPR minus FPR, moved halfway toward the nearest rejected distractor for margin. It applies to `raw_similarity`.
- Extras beyond the spec: `--json` on `critique` and `search`, `--force` and `--chunk-size` on `embed`, `--holdout`, `--seed`, `--threshold`, `--min-size` on `eval`.

## C — distill
- The VLM fills small draft schemas (`CardDraft`, `WebCardDraft`, `DomainDraft`, `CrossDomainDraft` in `distill/schemas.py`) instead of the stored contracts; `cluster_id`, `domain`, `exemplar_pin_ids`, `cluster_ids` are stamped from the records afterwards. Field descriptions in the drafts are part of the prompt (they reach the tool schema).
- Only web clusters get a schema with `tokens`; non-hex palette values are dropped and an empty token set becomes `null`.
- Images are labeled by position: the Anthropic client puts a text block "Image N:" before each image, and the prompt legend says `Image N: TARGET|NEIGHBOR <cluster-id> (board, title)`. FakeVLM sees the same list order.
- "Exemplars" means the first 8 of `exemplar_pin_ids` that have an image file on disk; that list is what the card stores, what is thumbnailed (`exemplars/<id>/<n>.jpg`, n = 1-based position) and what the incremental check compares.
- Neighbors: up to 2 exemplars from each of the record's neighbor clusters (so at most 14 images per call), cross-domain neighbors included as given by module B.
- Invariants are checked in code as well as in the prompt: unknown cluster ids are removed, and an invariant whose evidence spans fewer than two domains is dropped and reported. With FakeVLM (no evidence) the profile therefore has zero invariants.
- Incremental chain: a card is redone when its exemplars change; a domain summary when one of its cards was redone or its cluster list changed; the profile summary/invariants when any summary was redone or the domain set changed. A second run is zero VLM calls.
- `--only <id>` always redoes that card (no `--force` needed), then the chain above refreshes that domain's summary and the invariants (2 text-only calls).
- New `rule` feedback is folded into `profile.json` and `TASTE.md` on every run, including `--render-only`, without a VLM call. New like/dislike notes do not invalidate cards on their own: the CLI says how many are newer than the profile and suggests `--force`.
- Like/dislike notes go to a cluster or domain prompt when their domain matches or is unset; all of them go to the invariants prompt; capped at the latest 30. Feedback images are not sent.
- Profile files for clusters or domains no longer in `clusters.json` are deleted on each run (cluster ids are reassigned when clustering is re-run).
- The VLM is resolved lazily, so `--render-only` and no-op runs work without `ANTHROPIC_API_KEY`.
- DESIGN.md follows the google-labs-code/design.md spec (version `alpha`; `colors`, `typography`, `rounded`, `spacing`; canonical section order). A token is shared when every web card with tokens states it identically; otherwise it sits in a nested group named after the cluster (the variant). `rounded`/`spacing` get a front-matter value only when the text contains a CSS dimension; the prose always appears in the sections. `Imagery` is an extra section.
- No YAML dependency: `render.to_yaml` emits a small subset (nested maps, JSON-quoted strings, inline lists); tests parse it with a matching mini-parser.
- TASTE.md caps: 10 invariants, 20 rules, 1200 summary characters, with a pointer to `profile.json` for the rest (68 lines worst case).
- `TasteProfile.embed_model` is `settings.embed_model` (or "fake"); `vlm_model` is the VLM's `name`.
- Anthropic client: no temperature or thinking parameters (forced tool choice), `max_tokens` 4096, SDK retries 4; a truncated (`max_tokens`) or missing tool call counts as a validation failure and gets the one retry.

## D — serve
- The installed SDK is mcp 2.3, where FastMCP was renamed `MCPServer` (`mcp.server.fastmcp` no longer imports). `server.py` tries `mcp.server.mcpserver` first and falls back to `mcp.server.fastmcp`, so it runs on both sides of the `mcp>=1.2` pin.
- Tool logic lives in `tools.py` (Store in, data out); `server.py` only converts to MCP content. Tools reply with text and images, no structured output: the reader is an agent.
- Bad arguments raise `ToolInputError`, which the server turns into the SDK's `ToolError` so the agent reads the real message (`isError`). "Profile not built" is a normal reply naming the next `taste` command for that home's state (ingest, embed, cluster, distill, or `distill --render-only`).
- `taste feedback` is a sub-app: `taste feedback add <verdict> <note>` and `taste feedback list` (the spec's bare `taste feedback <verdict> <note>` would collide with `list`).
- Thumbnails for `get_exemplars` are made on the fly from `images/` (EXIF-rotated, max 512 px, JPEG q82), since the mapping from pin to `profile/exemplars/` files belongs to module C. `k` is clamped to 1..12 to keep replies small.
- `critique` and `log_feedback` verify the path is a real image with Pillow before use, so an agent cannot copy an arbitrary file (a `.env`, say) into the home. Feedback images are stored as `feedback/<feedback id>.<ext>`.
- `critique` checks for `clusters.json` before calling scoring and answers with the build hint when it is missing. With no distilled card for the best cluster it says to run `taste distill`.
- When module C's `.md` files are missing but the `.json` exists, `get_taste` renders the domain or card itself with a small local renderer (no import of distill internals).
- `_get_embedder` uses `taste.embed.get_embedder` when it exists; if the import fails it returns `FakeEmbedder` only for `TASTE_EMBEDDER=fake` and errors otherwise (a silent fake would score in the wrong space). One embedder is cached per home and model.
- Everything a tool prints is redirected to stderr (`contextlib.redirect_stdout`) so model-library chatter cannot corrupt the stdio stream; a subprocess test checks that stdout is JSON-RPC only.
- `taste install --print-mcp-config` prints the JSON and nothing else, and writes no files, so it can be piped. The home path in it is made absolute.
- `--export` merges into the target and never deletes anything there (it is the user's context repo); it refuses a target that overlaps `profile/`. `feedback.jsonl` is exported, the images in `feedback/` are not.
- SKILL.md has a "without the MCP server" section pointing at the plain files, so the exported profile is usable by agents that have no MCP.
