"""Rendering evaluation results into the tables the README reports.

Kept separate from ``metrics.py`` so that formatting choices never leak into
measurement. One formatter for every table means the baseline rows and the
model rows stay column-aligned and directly comparable, which is the entire
point of scoring everything through one harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from melochron.eval.metrics import DEFAULT_KS, SlicedResult

#: Order slices so the honest ones sit next to the flattering one.
SLICE_ORDER = ["overall", "repeat", "novel", "cold_user", "cold_item"]

SLICE_NOTES = {
    "overall": "all test instances; dominated by repeats",
    "repeat": "target already in the user's history",
    "novel": "target never played by this user before",
    "cold_user": "user held out of training entirely",
    "cold_item": "target in vocabulary but absent from training",
}


def results_to_frame(results: list[SlicedResult], ks: tuple[int, ...] = DEFAULT_KS) -> pd.DataFrame:
    rows: list[dict] = []
    for result in results:
        rows.extend(result.as_rows(ks))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["slice"] = pd.Categorical(df["slice"], categories=SLICE_ORDER, ordered=True)
    return df.sort_values(["slice", "model"], kind="stable").reset_index(drop=True)


def format_markdown(
    results: list[SlicedResult],
    ks: tuple[int, ...] = DEFAULT_KS,
    primary_k: int = 10,
) -> str:
    """One markdown table per slice, models as rows."""
    df = results_to_frame(results, ks)
    if df.empty:
        return "_no results_\n"

    metric_cols = [f"HR@{k}" for k in ks] + [f"NDCG@{primary_k}", f"MRR@{primary_k}"]
    out: list[str] = []

    for slice_name in SLICE_ORDER:
        part = df[df["slice"] == slice_name]
        if part.empty:
            continue

        n = int(part["n"].iloc[0])
        out.append(f"### {slice_name}  (n = {n:,})")
        out.append("")
        out.append(f"_{SLICE_NOTES.get(slice_name, '')}_")
        out.append("")
        out.append("| model | " + " | ".join(metric_cols) + " |")
        out.append("|" + "---|" * (len(metric_cols) + 1))

        # Best model per slice by the primary cutoff, marked in bold. Only when
        # there is a real winner: on a slice where everything scores 0.0000,
        # bolding every row reads as three winners rather than none, which is
        # the opposite of what that slice is telling you.
        scores = part[f"HR@{primary_k}"]
        best = scores.max()
        has_winner = best > 0 and (scores == best).sum() < len(scores)

        for _, row in part.iterrows():
            cells = [f"{row[c]:.4f}" for c in metric_cols]
            is_best = has_winner and row[f"HR@{primary_k}"] == best
            label = f"**{row['model']}**" if is_best else str(row["model"])
            out.append(f"| {label} | " + " | ".join(cells) + " |")

        if best == 0:
            out.append("")
            out.append(f"_No baseline scores above zero on this slice at k={primary_k}._")
        out.append("")

    return "\n".join(out)


def write(
    results: list[SlicedResult],
    outdir: str | Path,
    stem: str = "results",
    ks: tuple[int, ...] = DEFAULT_KS,
    context: dict | None = None,
) -> dict[str, Path]:
    """Write results as CSV, JSON and markdown.

    ``context`` records how the numbers were produced (corpus, cutoff, vocab
    size, split fractions). A metrics table without it is not reproducible, and
    the whole point of the README table is that someone can check it.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frame = results_to_frame(results, ks)
    paths = {
        "csv": outdir / f"{stem}.csv",
        "json": outdir / f"{stem}.json",
        "markdown": outdir / f"{stem}.md",
    }

    frame.to_csv(paths["csv"], index=False)
    paths["json"].write_text(
        json.dumps(
            {"context": context or {}, "rows": frame.to_dict(orient="records")},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    md = format_markdown(results, ks)
    if context:
        lines = ["## Run context", ""]
        lines += [f"- **{k}**: {v}" for k, v in context.items()]
        md = "\n".join(lines) + "\n\n" + md
    paths["markdown"].write_text(md, encoding="utf-8")

    return paths
