from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .config import FeatureProfile
from .db import connect, references
from .media import discover
from .model import MODEL_NAME, cache_dir, fetch_model
from .pipeline import compare as compare_pair
from .pipeline import index as index_files
from .pipeline import scan as scan_files
from .reporting import load_results, render_html, write_result


app = typer.Typer(
    help="Find reused source footage with inspectable evidence.", no_args_is_help=True
)
models_app = typer.Typer(help="Manage approved descriptor models.")
app.add_typer(models_app, name="models")


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
):
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def doctor() -> None:
    checks = {
        "python": platform.python_version(),
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
    }
    try:
        import av
        import faiss
        import torch

        checks.update(
            {
                "pyav": av.__version__,
                "faiss": getattr(faiss, "__version__", "available"),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": torch.cuda.device_count(),
                "model_cached": (
                    cache_dir() / "models" / "sscd_disc_mixup.no_l2_norm.torchscript.pt"
                ).exists(),
            }
        )
    except Exception as exc:
        checks["dependency_error"] = str(exc)
    typer.echo(json.dumps(checks, indent=2))


@models_app.command("fetch")
def models_fetch(
    name: Annotated[str, typer.Argument(help="Approved model name.")] = MODEL_NAME,
    offline: bool = False,
    source: Annotated[Path | None, typer.Option(help="Operator-approved local checkpoint.")] = None,
) -> None:
    if name != MODEL_NAME:
        raise typer.BadParameter(f"Only approved model is {MODEL_NAME}")
    typer.echo(str(fetch_model(offline=offline, source=source)), err=True)


@app.command("init")
def init_db(db: Annotated[Path, typer.Option("--db")]) -> None:
    connect(db.resolve()).close()
    typer.echo(f"Initialized {db.resolve()}", err=True)


@app.command()
def compare(
    reference: Path,
    query: Path,
    device: str = "auto",
    report: Annotated[Path | None, typer.Option()] = None,
    verify_samples: Annotated[int, typer.Option(min=0)] = 0,
    overwrite: bool = False,
    fail_on_match: bool = False,
) -> None:
    result = compare_pair(query.resolve(), reference.resolve(), device, verify_samples)
    if report:
        write_result(result, report.resolve(), overwrite)
    else:
        typer.echo(result.model_dump_json(indent=2))
    if fail_on_match and result.decision in {"exact_duplicate", "suspected_copy"}:
        raise typer.Exit(3)


@app.command("index")
def index_cmd(
    source: Path,
    db: Annotated[Path, typer.Option("--db")],
    recursive: bool = False,
    device: str = "auto",
) -> None:
    paths = discover(source.resolve(), recursive)
    if not paths:
        raise typer.BadParameter("No supported videos found")
    value = index_files(paths, db.resolve(), device)
    typer.echo(json.dumps(value, indent=2), err=True)
    if value["failed"]:
        raise typer.Exit(1)


@app.command()
def scan(
    source: Path,
    db: Annotated[Path, typer.Option("--db")],
    recursive: bool = False,
    within_batch: bool = False,
    device: str = "auto",
    report: Annotated[Path | None, typer.Option()] = None,
    verify_samples: Annotated[int, typer.Option(min=0)] = 0,
    overwrite: bool = False,
    fail_on_match: bool = False,
) -> None:
    paths = discover(source.resolve(), recursive)
    if not paths:
        raise typer.BadParameter("No supported videos found")
    results = scan_files(paths, db.resolve(), device, within_batch, verify_samples)
    text = "\n".join(x.model_dump_json() for x in results) + "\n"
    if report:
        target = report.resolve()
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {target}; pass --overwrite")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    else:
        typer.echo(text, nl=False)
    if any(x.execution_status == "failed" for x in results):
        raise typer.Exit(1)
    if fail_on_match and any(x.decision in {"exact_duplicate", "suspected_copy"} for x in results):
        raise typer.Exit(3)


@app.command()
def report(
    results: Path, html: Annotated[Path, typer.Option("--html")], overwrite: bool = False
) -> None:
    target = render_html(load_results(results.resolve()), html.resolve(), overwrite)
    typer.echo(str(target), err=True)


@app.command()
def stats(db: Annotated[Path, typer.Option("--db")]) -> None:
    con = connect(db.resolve())
    fp = FeatureProfile()
    rows = references(con, fp.identity)
    typer.echo(
        json.dumps(
            {
                "references": len(rows),
                "feature_profile_sha256": fp.identity,
                "descriptor_payload_bytes": sum(
                    Path(r["feature_path"], "raw.npy").stat().st_size for r in rows
                ),
            },
            indent=2,
        )
    )


@app.command()
def rebuild(db: Annotated[Path, typer.Option("--db")]) -> None:
    from .features import load_features
    import faiss
    import numpy as np
    import uuid

    root = db.resolve()
    con = connect(root)
    fp = FeatureProfile()
    rows = references(con, fp.identity)
    generation = f"generation-{uuid.uuid4().hex}"
    target = root / "generations" / generation
    target.mkdir(parents=True)
    vectors, ids = [], []
    for row in rows:
        raw, _ = load_features(Path(row["feature_path"]))
        vectors.append(np.asarray(raw))
        ids.extend([row["id"]] * len(raw))
    if vectors:
        matrix = np.concatenate(vectors).astype("float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(target / "shard-000.faiss"))
        np.save(target / "shard-000-ids.npy", np.asarray(ids, dtype="int64"))
    (target / "manifest.json").write_text(
        json.dumps({"id": generation, "vectors": len(ids), "complete": True}, indent=2)
    )
    con.execute(
        "INSERT INTO index_generations(id,profile_sha256,state) VALUES(?,?,'active')",
        (generation, fp.identity),
    )
    con.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES('active_generation',?)", (generation,)
    )
    con.commit()
    typer.echo(generation, err=True)


if __name__ == "__main__":
    app()
