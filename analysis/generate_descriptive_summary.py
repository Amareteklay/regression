"""Generate descriptive statistics and visualisations from full_panel.csv without pandas."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

THRESHOLD_SHARE_GT1 = 0.10  # 10% of positive values exceeding 1 -> keep counts


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_full_panel(data_path: Path) -> Dict[str, object]:
    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        paired_columns = {
            col[:-5]: col
            for col in header
            if col.endswith("__bin") and col[:-5] in header
        }

        total_rows = 0
        years = set()
        countries = set()
        continents = set()
        continent_stats = defaultdict(
            lambda: {"rows": 0, "outbreaks": 0, "cases": 0.0, "deaths": 0.0}
        )
        outbreak_total = 0
        cases_total = 0.0
        deaths_total = 0.0

        pair_stats = {
            base: {
                "nonzero_count": 0,
                "gt1_count": 0,
                "values_sum": 0.0,
                "values_sq_sum": 0.0,
            }
            for base in paired_columns
        }

        for row in reader:
            total_rows += 1
            year = row.get("Year")
            if year:
                years.add(year)
            country = row.get("Country")
            if country:
                countries.add(country)
            continent = row.get("Continent") or "Unknown"
            continents.add(continent)
            stats = continent_stats[continent]
            stats["rows"] += 1

            outbreak_value = safe_float(row.get("Outbreak", ""))
            if outbreak_value > 0:
                stats["outbreaks"] += 1
                outbreak_total += 1

            cases_value = safe_float(row.get("CasesTotal", ""))
            stats["cases"] += cases_value
            cases_total += cases_value

            deaths_value = safe_float(row.get("Deaths", ""))
            stats["deaths"] += deaths_value
            deaths_total += deaths_value

            for base, stat in pair_stats.items():
                value = safe_float(row.get(base, ""))
                if value > 0:
                    stat["nonzero_count"] += 1
                if value > 1:
                    stat["gt1_count"] += 1
                stat["values_sum"] += value
                stat["values_sq_sum"] += value * value

    summary: Dict[str, object] = {
        "threshold_share_gt1": THRESHOLD_SHARE_GT1,
        "overall": {
            "rows": total_rows,
            "years": sorted(years),
            "countries": len(countries),
            "continents": len(continents),
            "outbreak_events": outbreak_total,
            "cases_total": cases_total,
            "deaths_total": deaths_total,
        },
        "by_continent": [],
        "variable_selection": {},
    }

    for continent, stats in sorted(continent_stats.items()):
        rows = stats["rows"] or 1
        summary["by_continent"].append(
            {
                "continent": continent,
                "rows": stats["rows"],
                "outbreak_rate": stats["outbreaks"] / rows,
                "outbreaks": stats["outbreaks"],
                "mean_cases": stats["cases"] / rows,
                "mean_deaths": stats["deaths"] / rows,
            }
        )

    for base, stats_dict in pair_stats.items():
        nonzero = stats_dict["nonzero_count"]
        gt1 = stats_dict["gt1_count"]
        share_gt1 = (gt1 / nonzero) if nonzero else 0.0
        selection = "count" if share_gt1 >= THRESHOLD_SHARE_GT1 else "binary"
        summary["variable_selection"][base] = {
            "use": selection,
            "nonzero_count": nonzero,
            "gt1_count": gt1,
            "share_gt1": share_gt1,
        }

    return summary


def _write_horizontal_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    value_label: str,
    formatter,
    output_path: Path,
) -> None:
    if not values:
        return
    width = 900
    height = 140 + 40 * len(values)
    margin_left, margin_right, margin_top, margin_bottom = 200, 60, 60, 60
    plot_width = width - margin_left - margin_right
    max_value = max(values)
    scale = plot_width / max_value if max_value else 0.0

    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        bar_width = value * scale
        text_y = margin_top + i * 40 + 25
        rect_y = margin_top + i * 40 + 10
        value_text = formatter(value)
        bars.append(
            f"<text x=\"{margin_left - 10}\" y=\"{text_y}\" text-anchor=\"end\" font-size=\"14\">{label}</text>"
            f"<rect x=\"{margin_left}\" y=\"{rect_y}\" width=\"{bar_width:.2f}\" height=\"20\" fill=\"#4682b4\" />"
            f"<text x=\"{margin_left + bar_width + 5}\" y=\"{text_y}\" font-size=\"12\">{value_text}</text>"
        )

    bars_svg = "".join(bars)
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
    <style>
        text {{ font-family: Arial, sans-serif; }}
    </style>
    <rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='black' stroke-width='1'/>
    <text x='{width/2}' y='{margin_top/2}' text-anchor='middle' font-size='18'>{title}</text>
    <text x='{width/2}' y='{height - margin_bottom/2}' text-anchor='middle' font-size='14'>{value_label}</text>
    {bars_svg}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def export_descriptive_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = summary.get("by_continent", [])
    if not entries:
        return

    csv_path = output_dir / "by_continent.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "continent",
            "rows",
            "outbreaks",
            "outbreak_rate",
            "mean_cases",
            "mean_deaths",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow({
                "continent": entry.get("continent", ""),
                "rows": entry.get("rows", 0),
                "outbreaks": entry.get("outbreaks", 0),
                "outbreak_rate": entry.get("outbreak_rate", 0.0),
                "mean_cases": entry.get("mean_cases", 0.0),
                "mean_deaths": entry.get("mean_deaths", 0.0),
            })

    labels = [entry.get("continent", "") for entry in entries]
    outbreak_rates = [entry.get("outbreak_rate", 0.0) * 100 for entry in entries]
    mean_cases = [entry.get("mean_cases", 0.0) for entry in entries]
    mean_deaths = [entry.get("mean_deaths", 0.0) for entry in entries]

    _write_horizontal_bar_chart(
        labels,
        outbreak_rates,
        "Outbreak rate by continent",
        "Outbreak rate (%)",
        lambda value: f"{value:.1f}%",
        output_dir / "outbreak_rate.svg",
    )
    _write_horizontal_bar_chart(
        labels,
        mean_cases,
        "Mean outbreak cases by continent",
        "Cases per country-year",
        lambda value: f"{value:.1f}",
        output_dir / "mean_cases.svg",
    )
    _write_horizontal_bar_chart(
        labels,
        mean_deaths,
        "Mean outbreak deaths by continent",
        "Deaths per country-year",
        lambda value: f"{value:.1f}",
        output_dir / "mean_deaths.svg",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "full_panel.csv"
    summary_path = project_root / "analysis" / "descriptive_summary.json"

    summary = summarize_full_panel(data_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    export_descriptive_outputs(summary, project_root / "outputs" / "descriptive")
    print(f"Wrote descriptive summary to {summary_path}")


if __name__ == "__main__":
    main()
