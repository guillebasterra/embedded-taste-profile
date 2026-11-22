from __future__ import annotations

from typer.testing import CliRunner

from taste.cli import app

runner = CliRunner()


def invoke(store, *args, vlm="fake"):
    env = {"TASTE_VLM": vlm, "TASTE_EMBEDDER": "fake", "ANTHROPIC_API_KEY": "", "COLUMNS": "200"}
    return runner.invoke(app, ["distill", "--home", str(store.home), *args], env=env)


def test_distill_cli_end_to_end(store):
    first = invoke(store)
    assert first.exit_code == 0, first.output
    assert "VLM calls: 9" in first.output
    p = store.profile_dir
    assert (p / "TASTE.md").exists() and (p / "DESIGN.md").exists()
    assert (p / "profile.json").exists() and (p / "clusters" / "web-01.md").exists()
    assert (p / "exemplars" / "photo-01" / "1.jpg").exists()

    second = invoke(store)
    assert second.exit_code == 0 and "VLM calls: 0" in second.output

    only = invoke(store, "--only", "fashion-01")
    assert only.exit_code == 0 and "Cards: 1 generated" in only.output

    forced = invoke(store, "--force")
    assert forced.exit_code == 0 and "VLM calls: 9" in forced.output

    (p / "TASTE.md").unlink()
    rendered = invoke(store, "--render-only", vlm="anthropic")  # no key needed: no VLM
    assert rendered.exit_code == 0, rendered.output
    assert (p / "TASTE.md").exists() and "VLM calls" not in rendered.output


def test_distill_cli_errors(store, tmp_path):
    assert invoke(store, "--only", "nope-01").exit_code == 1
    assert invoke(store, "--render-only", "--force").exit_code == 2
    missing_key = invoke(store, vlm="anthropic")
    assert missing_key.exit_code == 1 and "ANTHROPIC_API_KEY" in missing_key.output
    assert not (store.profile_dir / "profile.json").exists()
    unchanged = invoke(store)
    assert unchanged.exit_code == 0
    assert invoke(store, vlm="anthropic").exit_code == 0  # nothing to generate, no key needed
