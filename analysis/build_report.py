"""Build the Markdown report with descriptive stats and filtered regression summaries."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from generate_descriptive_summary import (
    THRESHOLD_SHARE_GT1,
    summarize_full_panel,
    export_descriptive_outputs,
)

DATASETS = {
    "lags": "Lags",
    "leads": "Leads",
    "contemp": "Contemporaneous",
}


def load_scope_outputs(dataset: str) -> List[Tuple[str, Path]]:
    base_dir = PROJECT_ROOT / "outputs" / dataset
    if not base_dir.exists():
        return []
    scopes: List[Tuple[str, Path]] = []
    for scope_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        scopes.append((scope_dir.name, scope_dir))
    return scopes


def term_variant(term: str, base: str) -> str | None:
    """Return "count" or "binary" if term belongs to the base family."""
    binary_prefix = f"{base}__bin"
    if term.startswith(binary_prefix):
        return "binary"
    if term == base or term.startswith(f"{base}_"):
        return "count"
    return None


def should_include_term(term: str, selection_map: Dict[str, Dict[str, object]]) -> bool:
    for base, info in selection_map.items():
        variant = term_variant(term, base)
        if variant is None:
            continue
        return variant == info.get("use")
    return True


def read_best_model(scope_dir: Path) -> Dict[str, object]:
    model_weights_path = scope_dir / "model_weights.csv"
    best = None
    with model_weights_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                weight = float(row.get("akaike_weight", "0") or 0.0)
            except ValueError:
                weight = 0.0
            if best is None or weight > best["akaike_weight"]:
                best = {
                    "spec_id": row.get("spec_id", ""),
                    "dv": row.get("dv", ""),
                    "akaike_weight": weight,
                }
    return best or {"spec_id": "", "dv": "", "akaike_weight": 0.0}


def read_top_coefficients(
    scope_dir: Path,
    selection_map: Dict[str, Dict[str, object]],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    coeff_path = scope_dir / "weighted_coefficients.csv"
    rows: List[Tuple[str, float]] = []
    with coeff_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            term = row.get("term", "")
            if not should_include_term(term, selection_map):
                continue
            try:
                estimate = float(row.get("weighted_estimate", "0") or 0.0)
            except ValueError:
                continue
            rows.append((term, estimate))
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    return rows[:top_k]


def fmt_int(value: float | int) -> str:
    return f"{int(round(value)):,}"


def fmt_float(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def fmt_percent(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def ensure_summary() -> Dict[str, object]:
    summary_path = PROJECT_ROOT / "analysis" / "descriptive_summary.json"
    data_path = PROJECT_ROOT / "data" / "full_panel.csv"
    summary = summarize_full_panel(data_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    export_descriptive_outputs(summary, PROJECT_ROOT / "outputs" / "descriptive")
    return summary


def build_descriptive_section(summary: Dict[str, object]) -> List[str]:
    lines: List[str] = ["# Regression Scope Report", ""]
    overall = summary.get("overall", {})
    years = overall.get("years", [])
    if years:
        start_year, end_year = years[0], years[-1]
    else:
        start_year = end_year = "?"
    lines.append(
        "This report starts with descriptive insights from the full panel ("
        f"{start_year}-{end_year}, {fmt_int(overall.get('rows', 0))} country-year observations across "
        f"{overall.get('countries', 0)} countries) before summarising the model-averaged regressions."
    )
    lines.append(
        "Intercepts, outbreak controls, and `scale(year)` terms were already excluded from the regression "
        "visualisations; here we additionally avoid reporting both count and binary versions of the same "
        "covariate."
    )
    lines.append("")

    lines.append("## Descriptive overview")
    lines.append("")
    lines.append(
        "- Total outbreak events recorded: "
        f"{fmt_int(overall.get('outbreak_events', 0))} ("
        f"{fmt_int(overall.get('cases_total', 0))} cases, {fmt_int(overall.get('deaths_total', 0))} deaths)."
    )
    lines.append(
        "- Geographic coverage: "
        f"{overall.get('continents', 0)} continents, {overall.get('countries', 0)} countries."
    )
    lines.append("")

    lines.append("### Outbreak intensity by continent")
    lines.append("")
    lines.append("| Continent | Observations | Outbreaks | Outbreak rate | Mean cases | Mean deaths |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for entry in summary.get("by_continent", []):
        lines.append(
            "| {continent} | {rows:,} | {outbreaks:,} | {rate} | {cases} | {deaths} |".format(
                continent=entry.get("continent", ""),
                rows=entry.get("rows", 0),
                outbreaks=entry.get("outbreaks", 0),
                rate=fmt_percent(entry.get("outbreak_rate", 0.0)),
                cases=fmt_float(entry.get("mean_cases", 0.0)),
                deaths=fmt_float(entry.get("mean_deaths", 0.0)),
            )
        )
    lines.append("")
    lines.append("![Outbreak rate by continent](outputs/descriptive/outbreak_rate.svg)")
    lines.append("")
    lines.append("![Mean outbreak cases by continent](outputs/descriptive/mean_cases.svg)")
    lines.append("")
    lines.append("![Mean outbreak deaths by continent](outputs/descriptive/mean_deaths.svg)")
    lines.append("")

    threshold = summary.get("threshold_share_gt1", THRESHOLD_SHARE_GT1)
    selection = summary.get("variable_selection", {})

    lines.append("### Count vs. binary decision")
    lines.append("")
    lines.append(
        "We inspected each hazard/exposure covariate that had both a count and binary version in the raw panel. "
        f"If at least {fmt_percent(threshold, 0)} of the non-zero observations exceeded one event, we kept the count "
        "series; otherwise we reported the binary indicator."
    )
    lines.append("")
    lines.append("| Variable | Encoding | Non-zero obs | Share >1 | Rationale |")
    lines.append("|---|---|---:|---:|---|")
    for base, info in sorted(selection.items()):
        share = info.get("share_gt1", 0.0)
        nonzero = info.get("nonzero_count", 0)
        encoding = info.get("use")
        if encoding == "binary":
            rationale = f"Only {fmt_percent(share)} of non-zero observations exceed one event; a binary indicator captures presence without adding noise."
        else:
            rationale = f"{fmt_percent(share)} of non-zero observations exceed one event (>= {fmt_percent(threshold)} threshold), so the count series conveys intensity."
        lines.append(f"| {base} | {encoding} | {nonzero:,} | {fmt_percent(share)} | {rationale} |")
    lines.append("")
    lines.append(
        "All subsequent regression summaries only reference the chosen encoding for each family (for example, we report "
        "`INTERSTATE__bin` rather than `INTERSTATE` because interstate disputes rarely exceed a single episode in a year)."
    )
    lines.append("")
    return lines


def build_regression_sections(summary: Dict[str, object]) -> List[str]:
    lines: List[str] = []
    selection = summary.get("variable_selection", {})

    for dataset, title in DATASETS.items():
        scopes = load_scope_outputs(dataset)
        if not scopes:
            continue
        lines.append(f"## {title}")
        lines.append("")
        for scope_name, scope_dir in scopes:
            best_model = read_best_model(scope_dir)
            top_coeffs = read_top_coefficients(scope_dir, selection)
            lines.append(f"### {title} - {scope_name.capitalize()}")
            lines.append(
                "- Best-scoring specification `{spec}` for DV `{dv}` with Akaike weight {wt:.3f}.".format(
                    spec=best_model["spec_id"],
                    dv=best_model["dv"],
                    wt=best_model["akaike_weight"],
                )
            )
            if top_coeffs:
                formatted_terms = ", ".join(
                    f"{term} ({estimate:+.3f})" for term, estimate in top_coeffs
                )
                lines.append(f"- Top weighted coefficients: {formatted_terms}.")
            else:
                lines.append("- Top weighted coefficients: (none retained after filtering).")

            top_model_svg = (scope_dir / "top_model_weights.svg").relative_to(PROJECT_ROOT).as_posix()
            lines.append(f"- Top-model weights figure: `{top_model_svg}`")
            lines.append("")

            coeff_svg = (scope_dir / "weighted_coefficients.svg").relative_to(PROJECT_ROOT).as_posix()
            importance_svg = (scope_dir / "predictor_importance.svg").relative_to(PROJECT_ROOT).as_posix()
            lines.append(f"![Weighted coefficients]({coeff_svg})")
            lines.append("")
            lines.append(f"![Predictor importance]({importance_svg})")
            lines.append("")
        lines.append("")
    return lines


def build_report() -> None:
    summary = ensure_summary()
    report_lines = build_descriptive_section(summary)
    report_lines.extend(build_regression_sections(summary))

    report_path = PROJECT_ROOT / "report.md"
    report_path.write_text("\n".join(report_lines))
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    build_report()


