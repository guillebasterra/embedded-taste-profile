import json

import pytest
from PIL import Image
from typer.testing import CliRunner

from taste.cli import app
from taste.serve import tools

runner = CliRunner()


def run(*args):
    return runner.invoke(app, [str(a) for a in args])


def test_commands_are_registered():
    out = run("--help").output
    assert "serve" in out and "feedback" in out and "install" in out
    for cmd in (("serve",), ("install",), ("feedback", "add"), ("feedback", "list")):
        assert "--home" in run(*cmd, "--help").output


# ---- feedback ----------------------------------------------------------------

def test_feedback_add_and_list(bare, tmp_path):
    shot = tmp_path / "shot.png"
    Image.new("RGB", (20, 20), (1, 2, 3)).save(shot)
    r = run("feedback", "add", "dislike", "too much padding", "--image", shot,
            "--domain", "web", "--context", "portfolio", "--home", bare.home)
    assert r.exit_code == 0, r.output
    assert "Logged dislike feedback" in r.output
    assert run("feedback", "add", "rule", "never use gradients", "--home", bare.home).exit_code == 0

    first, second = bare.read_feedback()
    assert (first.verdict, first.note, first.domain, first.context) == (
        "dislike", "too much padding", "web", "portfolio")
    assert bare.abs(first.image_path).read_bytes() == shot.read_bytes()
    assert second.verdict == "rule"

    listing = run("feedback", "list", "--home", bare.home)
    assert listing.exit_code == 0
    assert "too much padding" in listing.output and "never use gradients" in listing.output
    assert first.id in listing.output and "portfolio" in listing.output


def test_feedback_list_empty(bare):
    r = run("feedback", "list", "--home", bare.home)
    assert r.exit_code == 0 and "No feedback yet" in r.output


def test_feedback_add_rejects_bad_verdict(bare):
    r = run("feedback", "add", "meh", "note", "--home", bare.home)
    assert r.exit_code == 2 and "verdict must be one of" in r.output
    assert bare.read_feedback() == []


def test_home_from_env(bare, monkeypatch):
    monkeypatch.setenv("TASTE_HOME", str(bare.home))
    assert run("feedback", "add", "like", "good").exit_code == 0
    assert len(bare.read_feedback()) == 1


# ---- install -----------------------------------------------------------------

def test_install_writes_skill_and_snippet(built):
    r = run("install", "--home", built.home)
    assert r.exit_code == 0, r.output
    skill = (built.profile_dir / "SKILL.md").read_text()
    snippet = (built.profile_dir / "CLAUDE.snippet.md").read_text()
    for name in tools.TOOL_NAMES:
        assert name in skill and name in snippet

    assert skill.startswith("---\n")
    front = skill.split("---\n")[1]
    keys = dict(line.split(": ", 1) for line in front.strip().splitlines())
    assert keys["name"] == "taste" and len(keys["description"]) > 40
    assert "Log every visual correction with `log_feedback`" in skill
    assert "Log every visual correction with `log_feedback`" in snippet
    assert len(snippet.splitlines()) <= 12
    assert "not built" not in r.output


def test_install_on_unbuilt_home_says_what_to_run(bare):
    r = run("install", "--home", bare.home)
    assert r.exit_code == 0 and "taste embed" in r.output
    assert (bare.profile_dir / "SKILL.md").exists()


def test_print_mcp_config_is_valid_json(built, monkeypatch):
    monkeypatch.chdir(built.home.parent)
    r = run("install", "--print-mcp-config", "--home", built.home.name)  # relative on purpose
    assert r.exit_code == 0
    cfg = json.loads(r.stdout)
    assert cfg == {"mcpServers": {"taste": {
        "command": "taste", "args": ["serve", "--home", str(built.home.resolve())]}}}
    assert not (built.profile_dir / "SKILL.md").exists()  # print only, no side effects


def test_export_copies_profile_without_source_images(built, tmp_path):
    tools.log_feedback(built, "dislike", "nope", built.abs("images/red-00.jpg"))
    dest = tmp_path / "vault" / "taste"
    dest.mkdir(parents=True)
    (dest / "keep.txt").write_text("mine")
    r = run("install", "--home", built.home, "--export", dest)
    assert r.exit_code == 0, r.output
    for rel in ("TASTE.md", "SKILL.md", "CLAUDE.snippet.md", "profile.json", "feedback.jsonl",
                "domains/web.md", "clusters/web-01.json", "exemplars/web-01/0.jpg"):
        assert (dest / rel).exists(), rel
    assert (dest / "keep.txt").read_text() == "mine"
    exported = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert not any(p.startswith(("images/", "feedback/")) for p in exported)
    assert [p for p in exported if p.endswith(".jpg")] == ["exemplars/web-01/0.jpg"]


@pytest.mark.parametrize("target", ["profile", "profile/sub", "."])
def test_export_refuses_overlapping_target(built, target):
    r = run("install", "--home", built.home, "--export", built.home / target)
    assert r.exit_code == 2 and "overlaps" in r.output
