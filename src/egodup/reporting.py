from __future__ import annotations

import html
import json
from pathlib import Path

from .schemas import Result


def write_result(result: Result, path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2) + "\n")


def load_results(path: Path) -> list[Result]:
    text = path.read_text().strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        values = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [Result.model_validate(x) for x in values]


def render_html(results: list[Result], output: Path, overwrite: bool = False) -> Path:
    target = output / "index.html" if output.suffix.lower() != ".html" else output
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {target}; pass --overwrite")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    cards = []
    for result in results:
        matches = []
        for match in result.matches:
            segs = "".join(
                f"<tr><td>{s.query_interval_seconds[0]:.1f}–{s.query_interval_seconds[1]:.1f}s</td>"
                f"<td>{s.reference_interval_seconds[0]:.1f}–{s.reference_interval_seconds[1]:.1f}s</td>"
                f"<td>{s.median_cosine:.3f}</td><td>{s.support_coverage:.0%}</td>"
                f"<td>a={s.time_relation.a:.3f}, b={s.time_relation.b:.2f}</td></tr>"
                for s in match.segments
            )
            matches.append(
                f"<h3>{html.escape(match.reference_path)}</h3><table><thead><tr><th>Query interval</th><th>Reference interval</th><th>Median cosine</th><th>Coverage</th><th>Time relation</th></tr></thead><tbody>{segs}</tbody></table>"
            )
        cards.append(
            f"<article><span class='decision'>{html.escape(str(result.decision))}</span><h2>{html.escape(result.query.path)}</h2>{''.join(matches) or '<p>No supported match.</p>'}</article>"
        )
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>egodup evidence</title><style>body{{font:15px system-ui;background:#0b1020;color:#dbe6ff;margin:0}}header,main{{max-width:1100px;margin:auto;padding:32px}}h1{{font-size:42px}}article{{background:#131b31;border:1px solid #2b385b;border-radius:14px;padding:24px;margin:18px 0}}.decision{{float:right;color:#67e8f9}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #2b385b}}small{{color:#9eabc7}}</style></head><body><header><h1>egodup evidence</h1><p>Local, inspectable copy-detection signals. Human review is required.</p></header><main>{"".join(cards)}<small>Cosine scores are SSCD descriptor similarities. Thresholds are provisional; a negative is not an originality certificate.</small></main></body></html>"""
    target.write_text(doc)
    target.chmod(0o600)
    return target
