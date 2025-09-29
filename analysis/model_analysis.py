"""Common utilities for model result analysis."""
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class ModelResult:
    spec_id: str
    scope: str
    dv: str
    formula: str
    n: Optional[int]
    aic: float
    aicc: float
    bic: float
    log_lik: float
    converged: Optional[bool]
    fixed_effects: Dict[str, Dict[str, float]]
    error: Optional[str]
    delta_aicc: float = 0.0
    akaike_weight: float = 0.0
    aicc_rank: int = 0


def is_excluded_term(term: str) -> bool:
    term_lower = term.lower()
    if term_lower in {"(intercept)", "intercept"}:
        return True
    if "outbreak" in term_lower:
        return True
    return "scale(year" in term_lower


def slugify(value: str) -> str:
    """Create a filesystem-friendly slug from a scope name."""
    cleaned = ''.join((ch.lower() if ch.isalnum() else '_') for ch in value)
    slug = cleaned.strip('_')
    return slug or 'scope'


def parse_fixed_effects(field: str) -> Dict[str, Dict[str, float]]:
    if field in ("", None):
        return {}
    if isinstance(field, str) and field.upper() == "NA":
        return {}
    cleaned = field.replace('""', '"')
    parsed = json.loads(cleaned)
    return {
        term: {
            "Estimate": to_float(stats.get("Estimate")),
            "StdError": to_float(stats.get("StdError")),
        }
        for term, stats in parsed.items()
    }


def to_bool(value: str) -> Optional[bool]:
    if value in ("", None) or value.upper() == "NA":
        return None
    return value.upper() == "TRUE"


def to_int(value: str) -> Optional[int]:
    if value in ("", None) or value.upper() == "NA":
        return None
    return int(float(value))


def to_float(value) -> float:
    if value in ("", None):
        return math.nan
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.upper() in {"NA", "NAN", "NULL"}:
            return math.nan
        try:
            return float(cleaned)
        except ValueError:
            return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_model_results(path: Path) -> List[ModelResult]:
    results: List[ModelResult] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            error_value = row.get("error")
            error = None if not error_value or error_value == "NA" else error_value
            aic = to_float(row.get("aic"))
            aicc = to_float(row.get("aicc"))
            bic = to_float(row.get("bic"))
            log_lik = to_float(row.get("logLik"))
            if error is not None or math.isnan(aicc):
                continue
            result = ModelResult(
                spec_id=row["spec_id"],
                scope=row["scope"],
                dv=row["dv"],
                formula=row["formula"],
                n=to_int(row.get("n")),
                aic=aic,
                aicc=aicc,
                bic=bic,
                log_lik=log_lik,
                converged=to_bool(row.get("converged")),
                fixed_effects=parse_fixed_effects(row.get("FixedEffects", "")),
                error=error,
            )
            results.append(result)
    return results


def compute_model_weights(models: List[ModelResult]) -> None:
    if not models:
        return
    min_aicc = min(model.aicc for model in models)
    weight_values: List[float] = []
    for model in models:
        model.delta_aicc = model.aicc - min_aicc
        weight_values.append(math.exp(-0.5 * model.delta_aicc))
    weight_sum = sum(weight_values)
    for model, weight in zip(models, weight_values):
        model.akaike_weight = weight / weight_sum if weight_sum else 0.0
    models.sort(key=lambda m: m.aicc)
    for index, model in enumerate(models, start=1):
        model.aicc_rank = index


def extract_coefficients(models: Iterable[ModelResult]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for model in models:
        for term, stats in model.fixed_effects.items():
            rows.append(
                {
                    "spec_id": model.spec_id,
                    "term": term,
                    "estimate": stats.get("Estimate", math.nan),
                    "std_error": stats.get("StdError", math.nan),
                    "akaike_weight": model.akaike_weight,
                }
            )
    return rows


def compute_weighted_coefficients(coefficients: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for row in coefficients:
        term = row["term"]
        if is_excluded_term(term):
            continue
        estimate = row["estimate"]
        std_error = row["std_error"]
        weight = row["akaike_weight"]
        if math.isnan(weight) or math.isnan(estimate):
            continue
        if term not in grouped:
            grouped[term] = {
                "term": term,
                "weight_sum": 0.0,
                "weighted_estimate_sum": 0.0,
                "weighted_square_sum": 0.0,
                "weighted_stderr_sum": 0.0,
            }
        entry = grouped[term]
        entry["weight_sum"] += weight
        entry["weighted_estimate_sum"] += weight * estimate
        entry["weighted_square_sum"] += weight * estimate * estimate
        if not math.isnan(std_error):
            entry["weighted_stderr_sum"] += weight * std_error * std_error

    summary: List[Dict[str, float]] = []
    for entry in grouped.values():
        if entry["weight_sum"] == 0:
            continue
        mean = entry["weighted_estimate_sum"] / entry["weight_sum"]
        variance = max(
            entry["weighted_square_sum"] / entry["weight_sum"] - mean * mean,
            0.0,
        )
        stderr = math.sqrt(entry["weighted_stderr_sum"] / entry["weight_sum"]) if entry["weighted_stderr_sum"] else 0.0
        summary.append(
            {
                "term": entry["term"],
                "weight_sum": entry["weight_sum"],
                "weighted_estimate": mean,
                "weighted_sd": math.sqrt(variance),
                "weighted_std_error": stderr,
            }
        )
    summary.sort(key=lambda item: item["weighted_estimate"], reverse=True)
    return summary


def compute_variable_importance(coefficients: List[Dict[str, float]]) -> List[Dict[str, float]]:
    importance: Dict[str, float] = defaultdict(float)
    for row in coefficients:
        term = row["term"]
        if is_excluded_term(term):
            continue
        importance[term] += row["akaike_weight"]
    ordered = sorted(importance.items(), key=lambda item: item[1], reverse=True)
    return [
        {"term": term, "importance_weight": weight}
        for term, weight in ordered
    ]


def group_scope_dv(models: Iterable[ModelResult]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, Dict[str, object]] = {}
    for model in models:
        key = (model.scope, model.dv)
        if key not in grouped:
            grouped[key] = {
                "scope": model.scope,
                "dv": model.dv,
                "n_models": 0,
                "best_aicc": model.aicc,
                "top_weight": model.akaike_weight,
            }
        entry = grouped[key]
        entry["n_models"] += 1
        entry["best_aicc"] = min(entry["best_aicc"], model.aicc)
        entry["top_weight"] = max(entry["top_weight"], model.akaike_weight)
    return list(grouped.values())


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: float, digits: int = 6) -> float:
    return float(f"{value:.{digits}f}")


def create_bar_chart_svg(
    labels: List[str],
    values: List[float],
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    if not values:
        return
    width, height = 900, 120 + 40 * len(values)
    margin_left, margin_right, margin_top, margin_bottom = 220, 40, 60, 60
    plot_width = width - margin_left - margin_right
    max_value = max(values)
    scale = plot_width / max_value if max_value else 0

    def bar(y: int, label: str, value: float) -> str:
        bar_width = value * scale
        text_y = margin_top + y * 40 + 25
        rect_y = margin_top + y * 40 + 10
        return (
            f'<text x="{margin_left - 10}" y="{text_y}" text-anchor="end"'
            f' font-size="14">{label}</text>'
            f'<rect x="{margin_left}" y="{rect_y}" width="{bar_width:.2f}" height="20" fill="#4682b4" />'
            f'<text x="{margin_left + bar_width + 5}" y="{text_y}" font-size="12">{value:.3f}</text>'
        )

    bars = "".join(bar(i, label, value) for i, (label, value) in enumerate(zip(labels, values)))
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
    <style>
        text {{ font-family: Arial, sans-serif; }}
    </style>
    <rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='black' stroke-width='1'/>
    <text x='{width/2}' y='{margin_top/2}' text-anchor='middle' font-size='18'>{title}</text>
    <text x='{width/2}' y='{height - margin_bottom/2}' text-anchor='middle' font-size='14'>{x_label}</text>
    {bars}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def create_error_bar_chart_svg(
    labels: List[str],
    estimates: List[float],
    errors: List[float],
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    if not estimates:
        return
    width, height = 1000, 140 + 40 * len(estimates)
    margin_left, margin_right, margin_top, margin_bottom = 260, 60, 60, 60
    plot_width = width - margin_left - margin_right
    max_extent = max(abs(est) + err for est, err in zip(estimates, errors))
    scale = plot_width / (2 * max_extent) if max_extent else 1

    def row(y: int, label: str, estimate: float, error: float) -> str:
        center_x = margin_left + plot_width / 2 + estimate * scale
        base_y = margin_top + y * 40 + 20
        err_width = error * scale
        line_start = center_x - err_width
        line_end = center_x + err_width
        return (
            f'<text x="{margin_left - 15}" y="{base_y + 5}" text-anchor="end" font-size="14">{label}</text>'
            f'<line x1="{line_start:.2f}" y1="{base_y}" x2="{line_end:.2f}" y2="{base_y}" stroke="#000" stroke-width="2" />'
            f'<line x1="{line_start:.2f}" y1="{base_y - 6}" x2="{line_start:.2f}" y2="{base_y + 6}" stroke="#000" stroke-width="2" />'
            f'<line x1="{line_end:.2f}" y1="{base_y - 6}" x2="{line_end:.2f}" y2="{base_y + 6}" stroke="#000" stroke-width="2" />'
            f'<circle cx="{center_x:.2f}" cy="{base_y}" r="6" fill="#8b0000" />'
            f'<text x="{center_x + 10:.2f}" y="{base_y - 8}" font-size="12">{estimate:.3f} +/- {error:.3f}</text>'
        )

    zero_line = margin_left + plot_width / 2
    rows_svg = "".join(
        row(i, label, estimate, error)
        for i, (label, estimate, error) in enumerate(zip(labels, estimates, errors))
    )
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
    <style>
        text {{ font-family: Arial, sans-serif; }}
    </style>
    <rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='black' stroke-width='1'/>
    <line x1='{zero_line:.2f}' y1='{margin_top - 10}' x2='{zero_line:.2f}' y2='{height - margin_bottom + 10}' stroke='#666' stroke-width='1' stroke-dasharray='4 4' />
    <text x='{width/2}' y='{margin_top/2}' text-anchor='middle' font-size='18'>{title}</text>
    <text x='{width/2}' y='{height - margin_bottom/2}' text-anchor='middle' font-size='14'>{x_label}</text>
    {rows_svg}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def run_analysis(
    data_path: Path,
    output_dir: Path,
    top_n: int = 20,
    scope: Optional[str] = None,
    models: Optional[List[ModelResult]] = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    working_models = list(models) if models is not None else load_model_results(data_path)
    if scope is not None:
        working_models = [m for m in working_models if m.scope == scope]
    compute_model_weights(working_models)

    model_rows = [
        {
            "spec_id": m.spec_id,
            "scope": m.scope,
            "dv": m.dv,
            "formula": m.formula,
            "n": m.n if m.n is not None else "",
            "aic": format_float(m.aic),
            "aicc": format_float(m.aicc),
            "bic": format_float(m.bic),
            "logLik": format_float(m.log_lik),
            "converged": "" if m.converged is None else str(m.converged),
            "delta_aicc": format_float(m.delta_aicc),
            "akaike_weight": format_float(m.akaike_weight),
            "aicc_rank": m.aicc_rank,
        }
        for m in working_models
    ]
    model_weights_path = output_dir / "model_weights.csv"
    write_csv(
        model_weights_path,
        [
            "spec_id",
            "scope",
            "dv",
            "formula",
            "n",
            "aic",
            "aicc",
            "bic",
            "logLik",
            "converged",
            "delta_aicc",
            "akaike_weight",
            "aicc_rank",
        ],
        model_rows,
    )

    coefficients = extract_coefficients(working_models)
    coefficient_summary = compute_weighted_coefficients(coefficients)
    weighted_coefficients_path = output_dir / "weighted_coefficients.csv"
    write_csv(
        weighted_coefficients_path,
        [
            "term",
            "weight_sum",
            "weighted_estimate",
            "weighted_sd",
            "weighted_std_error",
        ],
        coefficient_summary,
    )

    importance = compute_variable_importance(coefficients)
    predictor_importance_path = output_dir / "predictor_importance.csv"
    write_csv(
        predictor_importance_path,
        ["term", "importance_weight"],
        importance,
    )

    scope_summary = group_scope_dv(working_models)
    scope_summary_path = output_dir / "scope_dv_summary.csv"
    write_csv(
        scope_summary_path,
        ["scope", "dv", "n_models", "best_aicc", "top_weight"],
        scope_summary,
    )

    top_models = model_rows[:top_n]
    top_model_weights_svg = output_dir / "top_model_weights.svg"
    create_bar_chart_svg(
        [row["spec_id"] for row in top_models],
        [float(row["akaike_weight"]) for row in top_models],
        top_model_weights_svg,
        "Top models by AICc weight",
        "Akaike weight",
    )

    weighted_coefficients_svg = None
    if coefficient_summary:
        weighted_coefficients_svg = output_dir / "weighted_coefficients.svg"
        create_error_bar_chart_svg(
            [row["term"] for row in coefficient_summary],
            [float(row["weighted_estimate"]) for row in coefficient_summary],
            [float(row["weighted_sd"]) for row in coefficient_summary],
            weighted_coefficients_svg,
            "Model-averaged fixed effects",
            "Coefficient value",
        )

    predictor_importance_svg = None
    if importance:
        predictor_importance_svg = output_dir / "predictor_importance.svg"
        create_bar_chart_svg(
            [row["term"] for row in importance],
            [float(row["importance_weight"]) for row in importance],
            predictor_importance_svg,
            "Predictor importance (summed weights)",
            "Summed Akaike weight",
        )

    return {
        "scope": scope,
        "model_count": len(working_models),
        "model_rows": model_rows,
        "coefficient_summary": coefficient_summary,
        "importance": importance,
        "scope_summary": scope_summary,
        "outputs": {
            "model_weights": model_weights_path,
            "weighted_coefficients": weighted_coefficients_path,
            "predictor_importance": predictor_importance_path,
            "scope_summary": scope_summary_path,
            "top_model_weights_svg": top_model_weights_svg,
            "weighted_coefficients_svg": weighted_coefficients_svg,
            "predictor_importance_svg": predictor_importance_svg,
        },
    }



def run_scope_analyses(
    data_path: Path,
    base_output_dir: Path,
    top_n: int = 20,
) -> Dict[str, Dict[str, object]]:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    all_models = load_model_results(data_path)
    scopes = sorted({model.scope for model in all_models})
    results: Dict[str, Dict[str, object]] = {}
    for scope in scopes:
        scope_dir = base_output_dir / slugify(scope)
        scope_models = [model for model in all_models if model.scope == scope]
        results[scope] = run_analysis(
            data_path=data_path,
            output_dir=scope_dir,
            top_n=top_n,
            scope=scope,
            models=list(scope_models),
        )
    return results


