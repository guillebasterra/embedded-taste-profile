import json

import numpy as np
from typer.testing import CliRunner

from taste.cli import app
from taste.fixtures import GROUPS, make_image
from taste.store import Store

runner = CliRunner()


def make_folder(root, n=6, start=0, seed=0):
    rng = np.random.default_rng(seed)
    for board, _domain, group, rgb in GROUPS:
        for i in range(start, start + n):
            make_image(root / board / f"{group}-{i}.jpg", rgb, rng)


def run(*args):
    return runner.invoke(app, [str(a) for a in args])


def test_update_needs_a_recorded_source(tmp_path):
    result = run("update", "--home", tmp_path / "home")
    assert result.exit_code == 1 and "no recorded ingest source" in result.output


def test_update_chains_every_step_and_is_incremental(tmp_path, monkeypatch):
    monkeypatch.setenv("TASTE_EMBEDDER", "fake")
    monkeypatch.setenv("TASTE_VLM", "fake")
    monkeypatch.setenv("COLUMNS", "200")
    home, src, out = tmp_path / "home", tmp_path / "boards", tmp_path / "vault"
    make_folder(src)
    assert run("ingest", "--home", home, "--source", "local", "--path", src).exit_code == 0
    # what a user would set after `taste domains`
    boards = "\n".join(f'[boards."{b}"]\ndomain = "{d}"\ninclude = true\n'
                       for b, d in {(b, d) for b, d, _, _ in GROUPS})
    (home / "boards.toml").write_text(boards)

    first = run("update", "--home", home, "--export", out)
    assert first.exit_code == 0, first.output
    for step in ("[1/8] taste ingest --source local", "[3/8] taste embed",
                 "[4/8] taste facets", "[5/8] taste threads", "Skipped threads",
                 "[7/8] taste distill", "[8/8] taste install --export", "up to date"):
        assert step in first.output
    store = Store(home)
    ids_before = {c.id: set(c.pin_ids) for c in store.read_clusters()}
    assert len(ids_before) == 5 and (out / "TASTE.md").exists()
    assert json.loads((home / "update_state.json").read_text())["export"] == str(out.resolve())

    make_folder(src, n=3, start=6, seed=1)  # new pins on existing boards
    second = run("update", "--home", home)
    assert second.exit_code == 0, second.output
    assert "15 new" in second.output and "Kept 5 ids" in second.output
    assert f"taste install --export {out.resolve()}" in second.output  # remembered
    after = {c.id: set(c.pin_ids) for c in store.read_clusters()}
    assert set(after) == set(ids_before)
    assert all(ids_before[c] < after[c] for c in after)


def test_update_stops_at_the_first_failing_step(tmp_path, monkeypatch):
    monkeypatch.setenv("TASTE_EMBEDDER", "fake")
    home = tmp_path / "home"
    src = tmp_path / "boards"
    make_folder(src, n=2)
    assert run("ingest", "--home", home, "--source", "local", "--path", src).exit_code == 0
    src.rename(tmp_path / "moved")  # the recorded folder is gone
    result = run("update", "--home", home)
    assert result.exit_code == 1
    assert "Stopped: `taste ingest` exited with 1" in result.output
    assert "taste embed" not in result.output
