from pathlib import Path

from typer.testing import CliRunner

from egodup.cli import app


runner = CliRunner()


def _video(path: Path):
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=12",
            "-t",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_exact_compare_does_not_require_model(tmp_path: Path):
    original = tmp_path / "original.mp4"
    alias = tmp_path / "alias.mp4"
    report = tmp_path / "out.json"
    _video(original)
    alias.write_bytes(original.read_bytes())
    result = runner.invoke(
        app, ["compare", str(original), str(alias), "--device", "cpu", "--report", str(report)]
    )
    assert result.exit_code == 0, result.output
    assert '"decision": "exact_duplicate"' in report.read_text()


def test_init_stats(tmp_path: Path):
    root = tmp_path / "db"
    assert runner.invoke(app, ["init", "--db", str(root)]).exit_code == 0
    result = runner.invoke(app, ["stats", "--db", str(root)])
    assert result.exit_code == 0
    assert '"references": 0' in result.output


def test_exact_non_video_still_preserves_hash_finding(tmp_path: Path):
    original = tmp_path / "broken-a.mp4"
    alias = tmp_path / "broken-b.mp4"
    original.write_bytes(b"same bytes, not decodable video")
    alias.write_bytes(original.read_bytes())
    result = runner.invoke(app, ["compare", str(original), str(alias), "--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert '"decision": "exact_duplicate"' in result.output
    assert "Media probing failed" in result.output
