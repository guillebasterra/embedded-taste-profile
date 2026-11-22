# Decisions

Calls made without asking, newest last. One line each.

## Orchestrator
- Python + uv + Typer; storage is plain files under `TASTE_HOME` so the profile is portable and diffable.
- Default embedder `google/siglip2-base-patch16-384` (image and text towers share a space, so text-to-image retrieval works). Swappable via `TASTE_EMBED_MODEL`.
- Agglomerative clustering with a distance threshold instead of a fixed k: the number of tastes per domain should come from the data.
- Three ingest sources behind one `Pin` shape, so the pipeline runs on a local folder while Pinterest app approval is pending.
- Built in a cloud workspace that cannot reach Hugging Face, Pinterest or the Anthropic API: the real SigLIP, Pinterest and Claude paths are written against their docs and covered by mocked tests, and need a first live run on my machine.

## A — ingest

## B — embed / cluster / scoring

## C — distill

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
