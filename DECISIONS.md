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

## B — embed / cluster / scoring

## C — distill

## D — serve
