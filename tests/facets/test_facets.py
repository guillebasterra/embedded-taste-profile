import json

from PIL import Image
from typer.testing import CliRunner

from taste.cli import app
from taste.facets.measure import measure
from taste.facets.read import (MEASURES_FILE, READINGS_FILE, PinReading, facets_dir,
                               load_measures, load_readings, run_facets)
from taste.fixtures import make_fixture_home
from taste.interfaces import FakeVLM

runner = CliRunner()


def test_measure_tells_warm_vivid_from_cool_grey(tmp_path):
    red, grey = tmp_path / "red.png", tmp_path / "grey.png"
    Image.new("RGB", (300, 150), (210, 40, 30)).save(red)
    Image.new("RGB", (100, 100), (120, 125, 130)).save(grey)
    r, g = measure("r", red), measure("g", grey)
    assert r.palette[0].hex == "#d2281e" and r.palette[0].share == 1.0
    assert r.warmth > 0.5 and r.saturation > 0.8 and r.aspect == 2.0
    assert g.warmth < 0 and g.saturation < 0.1 and g.contrast == 0.0
    assert r.colorfulness > g.colorfulness and g.busyness == 0.0


def test_run_reads_every_included_pin_once(tmp_path):
    store = make_fixture_home(tmp_path, n_per_group=3)
    included = store.pin_domains()
    vlm = FakeVLM()
    stats = run_facets(store, vlm, workers=4)
    assert stats.measured == stats.read == len(included) and not stats.failed
    assert set(load_measures(store)) == set(load_readings(store)) == set(included)
    # context goes in, the reading comes back validated
    call = vlm.calls[0]
    assert call["schema"] is PinReading and len(call["images"]) == 1
    assert "Board: " in call["prompt"] and "Title: " in call["prompt"]

    again = run_facets(store, vlm)
    assert again.measured == again.read == 0 and again.kept == len(included)
    assert len(vlm.calls) == len(included)


def test_sample_measure_only_and_a_new_model(tmp_path):
    store = make_fixture_home(tmp_path, n_per_group=3)
    n = len(store.pin_domains())
    stats = run_facets(store, None, sample=2)
    assert stats.measured == 2 and stats.read == 0
    assert not (facets_dir(store) / READINGS_FILE).exists()

    run_facets(store, None)
    assert len(load_measures(store)) == n
    first = FakeVLM()
    run_facets(store, first)
    other = FakeVLM()
    other.name = "another-model"
    stats = run_facets(store, other)  # a different model's readings do not count as done
    assert stats.read == n and stats.measured == 0
    rows = (facets_dir(store) / READINGS_FILE).read_text().splitlines()
    assert len(rows) == 2 * n and json.loads(rows[-1])["model"] == "another-model"


def test_a_failed_pin_is_reported_and_retried_next_run(tmp_path):
    store = make_fixture_home(tmp_path, n_per_group=2)
    included = store.pin_domains()
    broken = next(p for p in store.read_pins() if p.id in included)
    store.abs(broken.image_path).write_bytes(b"not an image")
    stats = run_facets(store, FakeVLM())
    assert [pid for pid, _ in stats.failed] == [broken.id]
    assert broken.id not in load_measures(store)
    make_fixture_home(tmp_path, n_per_group=2)  # image back
    stats = run_facets(store, FakeVLM())
    assert stats.measured == stats.read == 1 and not stats.failed


def test_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("TASTE_VLM", "fake")
    monkeypatch.setenv("COLUMNS", "200")
    store = make_fixture_home(tmp_path, n_per_group=2)
    n = len(store.pin_domains())
    result = runner.invoke(app, ["facets", "--home", str(tmp_path), "--measure-only"])
    assert result.exit_code == 0, result.output
    assert f"{n} measured, 0 read" in result.output
    result = runner.invoke(app, ["facets", "--home", str(tmp_path)])
    assert result.exit_code == 0 and f"0 measured, {n} read" in result.output
    assert (facets_dir(store) / MEASURES_FILE).exists()
