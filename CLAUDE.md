# taste — working notes for agents

- Read `SPEC.md` first. It is the contract.
- Python 3.12+, `uv`. Run tests with `uv run pytest -q`.
- Shared contract files are marked at the top (`SHARED CONTRACT FILE`). Feature branches do not edit them, `pyproject.toml`, or another module's package. If something is missing, work around it locally and say so in your report.
- Modules talk through `taste.store.Store` only. No module imports another module's internals, with one exception: `taste.scoring.api`, `taste.embed.get_embedder`, `taste.distill.llm.get_vlm`.
- Tests never hit the network or load model weights: `FakeEmbedder`, `FakeVLM`, `respx`, `make_fixture_home(tmp_path)`.
- Heavy imports (torch, transformers, anthropic, mcp) are lazy, inside the function that needs them.
- Secrets and tokens are never printed or logged.
- Log every judgment call in `DECISIONS.md` under your module heading (one line each).
