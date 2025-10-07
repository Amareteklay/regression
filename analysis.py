import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
FILTERED_DIR = OUTPUT_DIR / "filtered"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
TEXT_DIR = OUTPUT_DIR / "text"

SCOPES = ["Global", "Africa", "America", "Asia", "Europe"]
DROP_SCOPES = {"Oceania"}
CONTROL_TERMS = {"(Intercept)", "scale(Year)", "Outbreak_lag", "Outbreak_lead"}

ALLOWED_HAZARDS = {
    "lags": {"ECONOMIC", "CLIMATIC", "GEOPHYSICAL", "INTERSTATE", "INTRASTATE", "ECOLOGICAL", "TECHNOLOGICAL"},
    "leads": {"ECONOMIC", "INTERSTATE", "INTRASTATE", "TERRORISM", "TECHNOLOGICAL"},
    "contemp": {"ECONOMIC", "CLIMATIC", "GEOPHYSICAL", "INTERSTATE", "INTRASTATE", "ECOLOGICAL", "TECHNOLOGICAL", "TERRORISM"},
}

DROP_HAZARDS = {
    "lags": {"POISONING", "TERRORISM"},
    "leads": {"GEOPHYSICAL", "CLIMATIC", "ECOLOGICAL", "POISONING"},
    "contemp": {"POISONING"},
}

ENCODING_RULES = {
    "ECOLOGICAL": "binary",
    "GEOPHYSICAL": "count",
    "INTERSTATE": "binary",
    "POISONING": "binary",
}

BASE_LABELS = {
    "ECONOMIC": "Economic shocks",
    "CLIMATIC": "Climatic shocks",
    "GEOPHYSICAL": "Geophysical shocks",
    "INTERSTATE": "Interstate conflict",
    "INTRASTATE": "Intrastate conflict",
    "ECOLOGICAL": "Ecological disruption",
    "TERRORISM": "Terrorism incidents",
    "POISONING": "Poisoning events",
    "TECHNOLOGICAL": "Technological accidents",
}

TIME_LABELS = {
    "lags": "prior",
    "leads": "subsequent",
    "contemp": "concurrent",
}

DATASET_TITLES = {
    "lags": "Prior shocks",
    "leads": "Subsequent shocks",
    "contemp": "Concurrent shocks",
}

DATASETS = {
    "lags": DATA_DIR / "lags.csv",
    "leads": DATA_DIR / "leads.csv",
    "contemp": DATA_DIR / "contemp.csv",
}


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "t", "1", "yes"}
    return False


def parse_fixed_effects(text):
    if pd.isna(text):
        return {}
    text = str(text).strip()
    if not text or text == "NA":
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def get_hazard_name(var_name):
    if not isinstance(var_name, str):
        return ""
    parts = var_name.split("_")
    return parts[0] if parts else var_name


def get_encoding_type(var_name):
    return "binary" if "__bin" in var_name else "count"


def encoding_allowed(var_name):
    hazard = get_hazard_name(var_name)
    rule = ENCODING_RULES.get(hazard)
    if not rule:
        return True
    return get_encoding_type(var_name) == rule


def format_label(var_name, dataset):
    hazard = get_hazard_name(var_name)
    base = BASE_LABELS.get(hazard, hazard.title().replace("_", " "))
    timing = TIME_LABELS.get(dataset, dataset)
    return f"{base} ({timing})"


def ensure_directories():
    for path in [OUTPUT_DIR, FILTERED_DIR, TABLE_DIR, FIGURE_DIR, TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def process_dataset(name, path):
    df = pd.read_csv(path)
    original_count = len(df)
    df = df[~df["scope"].isin(DROP_SCOPES)].copy()
    df["scope"] = df["scope"].astype(str).str.strip()
    df = df[df["scope"].isin(SCOPES)].copy()
    df["dv"] = df["dv"].astype(str).str.strip()
    df = df[df["dv"].str.lower() == "outbreak"].copy()

    keep_indices = []
    records = []

    for idx, row in df.iterrows():
        scope = row["scope"]
        if not to_bool(row.get("converged", True)):
            continue
        coeffs = parse_fixed_effects(row.get("FixedEffects"))
        if not coeffs:
            continue

        predictors = {}
        valid_model = True
        for var_name, stats in coeffs.items():
            if var_name in CONTROL_TERMS:
                continue
            if not isinstance(stats, dict):
                valid_model = False
                break
            est = stats.get("Estimate")
            se = stats.get("StdError")
            if est is None or se is None:
                valid_model = False
                break
            try:
                est_val = float(est)
                se_val = float(se)
            except (TypeError, ValueError):
                valid_model = False
                break
            if not math.isfinite(est_val) or not math.isfinite(se_val):
                valid_model = False
                break
            predictors[var_name] = {
                "estimate": est_val,
                "std_error": se_val,
            }

        if not valid_model or not predictors:
            continue
        if len(predictors) > 3:
            continue
        if any(not encoding_allowed(var) for var in predictors):
            continue

        hazards = {get_hazard_name(var) for var in predictors}
        if not hazards <= ALLOWED_HAZARDS[name]:
            continue
        if hazards & DROP_HAZARDS[name]:
            continue

        if any(pd.isna(row.get(col)) for col in ["aic", "aicc", "bic", "logLik", "n"]):
            continue

        keep_indices.append(idx)
        records.append(
            {
                "dataset": name,
                "spec_id": row["spec_id"],
                "scope": scope,
                "aic": float(row["aic"]),
                "aicc": float(row["aicc"]),
                "bic": float(row["bic"]),
                "logLik": float(row["logLik"]),
                "n": int(row["n"]),
                "predictors": predictors,
            }
        )

    if keep_indices:
        filtered_df = df.loc[keep_indices].copy()
    else:
        filtered_df = df.iloc[0:0].copy()

    if "converged" in filtered_df.columns:
        filtered_df["converged"] = filtered_df["converged"].apply(to_bool)

    return filtered_df, records, original_count


def compute_model_averages(records):
    results = []
    weights = []
    if not records:
        return pd.DataFrame(), pd.DataFrame()

    for dataset in DATASETS:
        dataset_records = [r for r in records if r["dataset"] == dataset]
        for scope in SCOPES:
            scope_records = [r for r in dataset_records if r["scope"] == scope]
            if not scope_records:
                continue
            min_aicc = min(r["aicc"] for r in scope_records)
            for r in scope_records:
                delta = r["aicc"] - min_aicc
                weight_raw = math.exp(-0.5 * delta)
                r["delta_aicc"] = delta
                r["weight_raw"] = weight_raw
            weight_sum = sum(r["weight_raw"] for r in scope_records)
            for r in scope_records:
                if weight_sum == 0:
                    r["akaike_weight"] = 0.0
                else:
                    r["akaike_weight"] = r["weight_raw"] / weight_sum
                weights.append(
                    {
                        "dataset": dataset,
                        "scope": scope,
                        "spec_id": r["spec_id"],
                        "delta_aicc": r["delta_aicc"],
                        "akaike_weight": r["akaike_weight"],
                        "predictor_count": len(r["predictors"]),
                    }
                )

            all_vars = sorted({var for r in scope_records for var in r["predictors"]})
            for var in all_vars:
                avg = 0.0
                for r in scope_records:
                    theta = r["predictors"][var]["estimate"] if var in r["predictors"] else 0.0
                    avg += r["akaike_weight"] * theta
                variance = 0.0
                inclusion = 0.0
                for r in scope_records:
                    weight = r["akaike_weight"]
                    theta = r["predictors"][var]["estimate"] if var in r["predictors"] else 0.0
                    se_i = r["predictors"][var]["std_error"] if var in r["predictors"] else 0.0
                    variance += weight * (((theta - avg) ** 2) + (se_i ** 2))
                    if var in r["predictors"]:
                        inclusion += weight
                variance = max(variance, 0.0)
                std_error = math.sqrt(variance)
                ci_lower = avg - 1.96 * std_error
                ci_upper = avg + 1.96 * std_error
                label = format_label(var, dataset)
                sign = 1 if ci_lower > 0 else -1 if ci_upper < 0 else 0

                results.append(
                    {
                        "dataset": dataset,
                        "scope": scope,
                        "variable": var,
                        "label": label,
                        "estimate": avg,
                        "std_error": std_error,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "inclusion_weight": inclusion,
                        "n_models": len(scope_records),
                        "sign": sign,
                    }
                )

    results_df = pd.DataFrame(results)
    weights_df = pd.DataFrame(weights)
    if not results_df.empty:
        results_df.sort_values(["dataset", "scope", "label"], inplace=True)
    if not weights_df.empty:
        weights_df.sort_values(["dataset", "scope", "delta_aicc"], inplace=True)
    return results_df, weights_df


def plot_coefficients(subset, dataset, scope, output_dir):
    subset = subset.sort_values("estimate").reset_index(drop=True)
    if subset.empty:
        return
    y_pos = np.arange(len(subset))
    fig_height = max(2.5, 1.2 + 0.6 * len(subset))
    fig, ax = plt.subplots(figsize=(6, fig_height))
    ax.hlines(y=y_pos, xmin=subset["ci_lower"], xmax=subset["ci_upper"], color="#4B8BBE", linewidth=2)
    ax.scatter(subset["estimate"], y_pos, color="#1f77b4", zorder=3)
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subset["label"])
    ax.set_xlabel("Model-averaged coefficient")
    ax.set_title(f"{scope} - {DATASET_TITLES[dataset]}")
    fig.tight_layout()
    filename = f"{dataset}_{scope}_coefficients.png".replace(" ", "_")
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)


def plot_inclusion_weights(subset, dataset, scope, output_dir):
    subset = subset.sort_values("inclusion_weight").reset_index(drop=True)
    if subset.empty:
        return
    fig_height = max(2.5, 1.2 + 0.6 * len(subset))
    fig, ax = plt.subplots(figsize=(6, fig_height))
    ax.barh(subset["label"], subset["inclusion_weight"], color="#1f77b4")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Akaike inclusion weight")
    ax.set_title(f"{scope} - {DATASET_TITLES[dataset]}")
    for idx, value in enumerate(subset["inclusion_weight"]):
        text_x = min(value + 0.02, 1.0)
        ax.text(text_x, idx, f"{value:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    filename = f"{dataset}_{scope}_inclusion.png".replace(" ", "_")
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)


def build_descriptive_stats():
    panel_path = DATA_DIR / "full_panel.csv"
    if not panel_path.exists():
        return pd.DataFrame()
    panel = pd.read_csv(panel_path)
    panel = panel[~panel["Continent"].isin(DROP_SCOPES)].copy()
    panel["Continent"] = panel["Continent"].astype(str).str.strip()
    panel = panel[panel["Continent"].isin(SCOPES)].copy()
    panel["Outbreak"] = pd.to_numeric(panel["Outbreak"], errors="coerce").fillna(0)
    panel["Deaths"] = pd.to_numeric(panel["Deaths"], errors="coerce").fillna(0)
    if "CasesTotal" in panel.columns:
        panel["CasesTotal"] = pd.to_numeric(panel["CasesTotal"], errors="coerce").fillna(0)
    else:
        panel["CasesTotal"] = 0

    grouped = panel.groupby("Continent", as_index=False).agg(
        outbreaks=("Outbreak", "sum"),
        deaths=("Deaths", "sum"),
        cases=("CasesTotal", "sum"),
        observations=("Outbreak", "size"),
    )
    grouped["outbreak_rate"] = grouped["outbreaks"] / grouped["observations"]

    totals = pd.DataFrame(
        {
            "Continent": ["Global"],
            "outbreaks": [grouped["outbreaks"].sum()],
            "deaths": [grouped["deaths"].sum()],
            "cases": [grouped["cases"].sum()],
            "observations": [grouped["observations"].sum()],
            "outbreak_rate": [grouped["outbreaks"].sum() / grouped["observations"].sum()],
        }
    )

    summary = pd.concat([totals, grouped], ignore_index=True)
    summary["outbreaks"] = summary["outbreaks"].round(0).astype(int)
    summary["deaths"] = summary["deaths"].round(0).astype(int)
    summary["cases"] = summary["cases"].round(0).astype(int)
    summary["observations"] = summary["observations"].astype(int)
    summary["outbreak_rate"] = summary["outbreak_rate"].round(3)

    return summary[["Continent", "outbreaks", "deaths", "cases", "observations", "outbreak_rate"]]


def dataframe_to_markdown(df):
    if df is None or df.empty:
        return "| Column | Value |\n| --- | --- |\n| No data available | |"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows)


def format_statement(row, direction):
    return (
        f"{row['label']} {direction} outbreak risk "
        f"(beta={row['estimate']:.2f}, 95% CI [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}], "
        f"inclusion weight {row['inclusion_weight']:.2f})"
    )


def describe_figures(subset, dataset, scope):
    if subset is None or subset.empty:
        return []
    base = f"{dataset}_{scope}".replace(' ', '_')
    coeff_file = f"outputs/figures/{base}_coefficients.png"
    incl_file = f"outputs/figures/{base}_inclusion.png"

    sig = subset[subset['sign'] != 0]
    if not sig.empty:
        top_sig = sig.sort_values(['inclusion_weight', 'estimate'], ascending=[False, False]).iloc[0]
        direction = 'increased' if top_sig['sign'] == 1 else 'reduced'
        coeff_caption = (
            f"{scope} {DATASET_TITLES[dataset]} coefficients. {top_sig['label']} {direction} outbreak risk while the remaining predictors show weaker or uncertain effects."
        )
    else:
        top_any = subset.sort_values(['inclusion_weight', 'estimate'], ascending=[False, False]).iloc[0]
        coeff_caption = (
            f"{scope} {DATASET_TITLES[dataset]} coefficients. Coefficient intervals overlap zero; {top_any['label']} has the largest point estimate but remains statistically uncertain."
        )

    top_weight = subset.sort_values('inclusion_weight', ascending=False).iloc[0]
    weight_value = top_weight['inclusion_weight']
    if weight_value >= 0.75:
        incl_phrase = (
            f"Model weight concentrates on {top_weight['label']} (inclusion weight {weight_value:.2f}), indicating consistent selection across top models."
        )
    elif weight_value >= 0.5:
        incl_phrase = (
            f"{top_weight['label']} carries the highest support (inclusion weight {weight_value:.2f}) but alternative predictors still appear in the model set."
        )
    else:
        incl_phrase = (
            f"Inclusion weights remain diffuse (maximum {weight_value:.2f}), indicating no single predictor dominates model selection."
        )
    incl_caption = f"{scope} {DATASET_TITLES[dataset]} inclusion weights. {incl_phrase}"

    return [
        {'path': coeff_file, 'caption': coeff_caption},
        {'path': incl_file, 'caption': incl_caption},
    ]


def assemble_results_text(descriptive_df, results_df):
    lines = ["# Results", ""]
    figure_counter = 1
    lines.extend(
        [
            "## Analytical Workflow",
            "",
            "- Screening: we restricted analysis to the five focal scopes, removed Oceania, retained only outbreak outcomes, dropped intercept, scaled year, and outbreak controls, and capped models at three substantive predictors to preserve interpretability.",
            "- Encoding: hazard variables followed predetermined rules so Ecological, Interstate, and Poisoning shocks entered as binary indicators while Geophysical shocks remained counts, preventing mixed representations within a hazard class.",
            "- Plausibility: we discarded any model containing hazards flagged as implausible for each timing window (for example, technological or poisoning lags) so inference rests on theoretically defensible predictors only.",
            "- Model averaging: within each scope we ranked eligible models by AICc, transformed delta AICc values into Akaike weights, and combined estimates with unconditional standard errors, confidence intervals, and inclusion weights to capture model-selection uncertainty.",
            "",
        ]
    )

    if descriptive_df is not None and not descriptive_df.empty:
        lines.extend(
            [
                "## Descriptive Patterns",
                "",
                "Outbreak activity remains uneven across continents once Oceania is excluded. "
                "The table summarises annual outbreaks, deaths, reported cases, and the share of observations with at least one outbreak.",
                "",
                dataframe_to_markdown(descriptive_df),
                "",
            ]
        )

    lines.append("## Scope-Specific Findings")
    lines.append("")

    if results_df.empty:
        lines.append("No models passed the filtering criteria, so no inferential results are available.")
        return "\n".join(lines)

    for scope in SCOPES:
        scope_df = results_df[results_df["scope"] == scope]
        if scope_df.empty:
            continue
        lines.append(f"### {scope}")
        lines.append("")
        for dataset in DATASETS:
            subset = scope_df[scope_df["dataset"] == dataset]
            if subset.empty:
                lines.append(f"- {DATASET_TITLES[dataset]}: No eligible models.")
                continue
            sig_pos = subset[subset["ci_lower"] > 0].sort_values("inclusion_weight", ascending=False)
            sig_neg = subset[subset["ci_upper"] < 0].sort_values("inclusion_weight", ascending=False)
            statements = []
            for _, row in sig_pos.iterrows():
                statements.append(format_statement(row, "increased"))
            for _, row in sig_neg.iterrows():
                statements.append(format_statement(row, "reduced"))
            if statements:
                lines.append(f"- {DATASET_TITLES[dataset]}: {'; '.join(statements)}.")
            else:
                lines.append(f"- {DATASET_TITLES[dataset]}: No predictors retained with 95% confidence.")
            figure_entries = describe_figures(subset, dataset, scope)
            for entry in figure_entries:
                lines.append(f"![Figure {figure_counter}. {entry['caption']}]({entry['path']})")
                lines.append(f"*Figure {figure_counter}. {entry['caption']}*")
                lines.append("")
                figure_counter += 1
        lines.append("")

    lines.append("## Global Overview")
    lines.append("")

    global_scope = results_df[results_df["scope"] == "Global"]
    non_global = results_df[(results_df["scope"] != "Global") & (results_df["sign"] != 0)]

    if global_scope.empty:
        lines.append("Global models were not retained after filtering.")
    else:
        sig_global = global_scope[global_scope["sign"] != 0].sort_values("inclusion_weight", ascending=False)
        if sig_global.empty:
            lines.append(
                "The global models did not yield variables with 95% confidence intervals that exclude zero, underlining regional heterogeneity."
            )
        else:
            parts = []
            for _, row in sig_global.iterrows():
                direction = "increased" if row["sign"] == 1 else "reduced"
                parts.append(
                    f"{row['label']} {direction} outbreak risk (beta={row['estimate']:.2f}, 95% CI [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}], inclusion weight {row['inclusion_weight']:.2f})"
                )
            lines.append("Global estimates highlight: " + "; ".join(parts) + ".")

    if not non_global.empty:
        lines.append("")
        lines.append("Cross-continent patterns show that:")
        for dataset in DATASETS:
            dataset_df = non_global[non_global["dataset"] == dataset]
            if dataset_df.empty:
                continue
            positives = dataset_df[dataset_df["sign"] == 1].groupby("label")["scope"].nunique().sort_values(ascending=False)
            negatives = dataset_df[dataset_df["sign"] == -1].groupby("label")["scope"].nunique().sort_values(ascending=False)
            segments = []
            if not positives.empty:
                label = positives.index[0]
                count = int(positives.iloc[0])
                segments.append(f"{label} showed positive associations in {count} scope(s)")
            if not negatives.empty:
                label = negatives.index[0]
                count = int(negatives.iloc[0])
                segments.append(f"{label} was protective in {count} scope(s)")
            if segments:
                lines.append(f"- {DATASET_TITLES[dataset]}: {'; '.join(segments)}.")

    lines.extend(
        [
            "",
            "## Tables and Figures",
            "",
            "- Model-averaged estimates: `outputs/tables/model_averaged_estimates.csv` and `outputs/tables/model_averaged_estimates.md`.",
            "- Model weights: `outputs/tables/model_weights.csv`.",
            "- Descriptive summary: `outputs/tables/descriptive_stats.csv`.",
            "- Coefficient and inclusion-weight plots are saved in `outputs/figures/` for each scope and shock timing.",
            "",
            "All analyses exclude Oceania and remove non-plausible predictors according to `AGENTS.md`.",
        ]
    )

    return "\n".join(lines)


def main():
    sns.set_theme(style="whitegrid")
    ensure_directories()

    all_records = []
    retained_summary = []

    for dataset, path in DATASETS.items():
        filtered_df, dataset_records, original_count = process_dataset(dataset, path)
        all_records.extend(dataset_records)
        retained_summary.append((dataset, len(filtered_df), original_count))
        output_path = FILTERED_DIR / f"{dataset}_filtered.csv"
        filtered_df.to_csv(output_path, index=False)

    results_df, weights_df = compute_model_averages(all_records)

    if not results_df.empty:
        results_df.to_csv(TABLE_DIR / "model_averaged_estimates.csv", index=False)
        table_view = results_df[
            ["dataset", "scope", "label", "estimate", "std_error", "ci_lower", "ci_upper", "inclusion_weight"]
        ].copy()
        for col in ["estimate", "std_error", "ci_lower", "ci_upper", "inclusion_weight"]:
            table_view[col] = table_view[col].round(3)
        with open(TABLE_DIR / "model_averaged_estimates.md", "w", encoding="utf-8") as handle:
            handle.write(dataframe_to_markdown(table_view))
    else:
        results_df.to_csv(TABLE_DIR / "model_averaged_estimates.csv", index=False)
        with open(TABLE_DIR / "model_averaged_estimates.md", "w", encoding="utf-8") as handle:
            handle.write("No models passed the filtering criteria.")

    if not weights_df.empty:
        weights_df.to_csv(TABLE_DIR / "model_weights.csv", index=False)
    else:
        weights_df.to_csv(TABLE_DIR / "model_weights.csv", index=False)

    for dataset in DATASETS:
        for scope in SCOPES:
            subset = results_df[(results_df["dataset"] == dataset) & (results_df["scope"] == scope)]
            if subset.empty:
                continue
            plot_coefficients(subset, dataset, scope, FIGURE_DIR)
            plot_inclusion_weights(subset, dataset, scope, FIGURE_DIR)

    descriptive_df = build_descriptive_stats()
    if descriptive_df is not None and not descriptive_df.empty:
        descriptive_df.to_csv(TABLE_DIR / "descriptive_stats.csv", index=False)

    results_text = assemble_results_text(descriptive_df, results_df)
    with open(TEXT_DIR / "results.md", "w", encoding="utf-8") as handle:
        handle.write(results_text)

    print("Model filtering summary:")
    for dataset, retained, original in retained_summary:
        print(f"  {dataset}: retained {retained} of {original} models")
    print("Outputs written to the 'outputs' directory.")


if __name__ == "__main__":
    main()
