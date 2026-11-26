"""Render both final reports from completed records: python ames-v3/portfolio-draft/render_final_readmes.py --write."""
import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DRAFT = Path(__file__).resolve().parent
V3 = DRAFT.parent
ROOT = V3.parent
RESULTS = V3 / "results"
sys.path.insert(0, str(V3))
sys.path.insert(0, str(ROOT))

from render_portfolio import currency, evidence, pct, recipe, tr
from src.ames_data import INPUT_SCHEMA


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def final_inputs():
    release = load_json("three-version-holdout.json")
    summary = load_json("summary.json")
    frozen = load_json("frozen-final.json")
    with gzip.open(RESULTS / "split-audits.json.gz", "rt", encoding="utf-8") as handle:
        audits = json.load(handle)
    outer_splits = {f"r{repeat}f{fold}" for repeat in range(2) for fold in range(5)}
    if set(audits) != outer_splits | {"temporal", "final"}:
        raise ValueError("Both reports require all 12 registered split audits.")
    if any(record.get("input_schema") != INPUT_SCHEMA for record in (release, summary, frozen, *audits.values())):
        raise ValueError("Complete the corrected runs and the three-release comparison before rendering.")
    if summary["dataset_sha256"] != release["dataset_sha256"] or summary["splits"] != release["ordered_split_sha256"]:
        raise ValueError("The release and nested reports refer to different sales.")
    for split, audit in audits.items():
        trials = audit["v2_search"]["trials"]
        if set(trials) != {"ridge", "xgboost", "lightgbm", "catboost"}:
            raise ValueError(f"{split} is missing a nested V2 family")
        if any([entry["trial"] for entry in history] != list(range(20)) for history in trials.values()):
            raise ValueError(f"{split} has an incomplete nested trial history")
    expected = {"outer": (1168, 2336), "temporal": (136, 136), "holdout": (292, 292)}
    for name, counts in expected.items():
        if (summary[name]["properties"], summary[name]["predictions"]) != counts:
            raise ValueError(f"The registered {name} predictions are incomplete")
    with gzip.open(RESULTS / "outer-predictions.csv.gz", "rt", encoding="utf-8") as handle:
        outer = list(csv.DictReader(handle))
    with (RESULTS / "final.csv").open(encoding="utf-8") as handle:
        historical = list(csv.DictReader(handle))
    if len(outer) != 2336 or len(historical) != 292 or len({row["Id"] for row in historical}) != 292:
        raise ValueError("The prediction CSVs do not cover the registered evaluation rows")
    sale = next(row for row in historical if row["Id"] == "899")
    return release, summary, frozen, audits, outer, sale


def signed(value, digits):
    return ("−" if value < 0 else "+") + tr(abs(value), digits)


def release_fields(release, sale):
    fields = evidence()
    rows = []
    for current, previous in (("v1", None), ("v2", "v1"), ("v3", "v2")):
        relative, dollars, points = ("—", "—", "—")
        if previous:
            diff = release["differences"][f"{current}_minus_{previous}"]
            relative = pct(diff["rmsle_pct"], signed=True)
            dollars = signed(diff["mae_usd"], 2)
            points = signed(diff["r2_points"], 5)
        cells = (current.upper() + " · " + fields[current + "_model"], fields[current + "_rmsle"],
                 relative, fields[current + "_mae"], dollars, fields[current + "_r2"], points)
        rows.append("| " + " | ".join(cells) + " |")
    change = release["differences"]["v3_minus_v1"]
    mae = change["mae_usd"]
    fields.update(release_table="\n".join(rows), v1_to_v3_mae=currency(abs(mae)),
                  v1_to_v3_mae_pct=pct(abs(change["mae_pct"])),
                  mae_direction="azaldı" if mae < 0 else "arttı" if mae > 0 else "değişmedi",
                  v1_training_rows=tr(release["effective_training_rows"]["v1"], 0),
                  nested_house_v2=tr(float(sale["v2"]), 2))
    return fields


def usd(value):
    return f"${value:,.2f}"


def metric_cells(metrics):
    return f"{metrics['rmsle']:.5f} | {usd(metrics['mae'])} | {metrics['r2']:.5f}"


def partition_table(summary):
    rows = []
    for key, title in (("outer", "Repeated outer CV"), ("temporal", "Forward year 2010"), ("holdout", "Historical holdout")):
        cohort = summary[key]
        for version, name in (("v2", "V2 nested"), ("v3", "V3")):
            rows.append(f"| {title}, {cohort['properties']:,} homes | {name} | {metric_cells(cohort[version])} |")
    return "\n".join(rows)


def ablation_table(metrics):
    effects = (
        ("One-hot → native categories", "onehot_plain", "native_plain"),
        ("Generic → structural missingness", "native_plain", "native_structural"),
        ("Global → neighborhood frontage median, indicator off", "native_structural", "native_neighborhood"),
        ("Global → neighborhood frontage median, indicator on", "native_indicator", "native_full"),
        ("Missingness indicator on, global median", "native_structural", "native_indicator"),
        ("Missingness indicator on, neighborhood median", "native_neighborhood", "native_full"),
    )
    rows = []
    for title, before, after in effects:
        previous = metrics["cand_" + before]["rmsle"]
        current = metrics["cand_" + after]["rmsle"]
        rows.append(f"| {title} | {previous:.5f} | {current:.5f} | {100 * (1 - current / previous):+.2f}% |")
    return "\n".join(rows)


def selection_counts(values):
    return ", ".join(f"{count} × {name}" for name, count in Counter(values).most_common())


def technical_fields(summary, frozen, audits, predictions):
    outer = summary["outer"]
    metrics = outer["model_metrics"]
    ci = outer["paired_bootstrap"]
    error = [abs(float(row["SalePrice"]) - float(row["v3"])) for row in predictions]
    median, p90 = np.quantile(error, [.5, .9])
    worst = max(predictions, key=lambda row: abs(float(row["SalePrice"]) - float(row["v2"])))
    worst_audit = audits[f"r{worst['repeat']}f{worst['fold']}"]
    location, neighborhood = max(outer["by_neighborhood_min20"].items(), key=lambda item: item[1]["v3"]["rmsle"])
    quartiles = outer["by_price_quartile"]
    complement = outer["residual_complementarity"]["v1_xgb vs v2_cat"]
    outer_audits = [audit for name, audit in audits.items() if name.startswith("r")]
    members = ("v1_xgb", "v2_cat", "cand_ridge_full", "cand_tabpfn_v2", "best_new", "equal_blend", "weighted_blend")
    has_foundation = "cand_tabpfn_v2" in metrics
    return {
        "outer_v3_rmsle": outer["v3"]["rmsle"], "outer_v3_mae": usd(outer["v3"]["mae"]),
        "outer_v2_rmsle": outer["v2"]["rmsle"], "outer_v2_mae": usd(outer["v2"]["mae"]),
        "outer_gain": outer["rmsle_gain_pct"], "mae_change": outer["mae_change_pct"],
        "rmsle_ci_low": ci["rmsle_gain_pct_ci95"][0], "rmsle_ci_high": ci["rmsle_gain_pct_ci95"][1],
        "mae_ci_low": ci["mae_change_pct_ci95"][0], "mae_ci_high": ci["mae_change_pct_ci95"][1],
        "promotion": "met" if summary["promote_v3"] else "did not meet",
        "partition_table": partition_table(summary), "v2_selections": selection_counts(a["v2"] for a in outer_audits),
        "worst_id": worst["Id"], "worst_repeat": worst["repeat"], "worst_fold": worst["fold"],
        "worst_actual": usd(float(worst["SalePrice"])), "worst_pred": usd(float(worst["v2"])),
        "worst_model": worst_audit["v2"], "fixed_cat_rmsle": metrics["v2_cat"]["rmsle"],
        "fixed_cat_gain": 100 * (1 - outer["v3"]["rmsle"] / metrics["v2_cat"]["rmsle"]),
        "median_error": usd(median), "p90_error": usd(p90), "neighborhood": location,
        "neighborhood_rmsle": neighborhood["v3"]["rmsle"], "neighborhood_homes": neighborhood["properties"],
        "quartile_one_mae": usd(quartiles["Q1"]["v3"]["mae"]), "quartile_four_mae": usd(quartiles["Q4"]["v3"]["mae"]),
        "ablation_table": ablation_table(metrics),
        "missing_fireplace": summary["development_missingness"]["fireplace_quality_missing"],
        "member_table": "\n".join(f"| {name} | {metric_cells(metrics[name])} |" for name in members if name in metrics),
        "v3_selections": selection_counts(recipe(a) for a in outer_audits), "final_recipe": recipe(frozen),
        "blend_weights": ", ".join(f"{weight:.4f}" for weight in frozen["weights"]),
        "opposite_errors": 100 * complement["opposite_error_sign_fraction"],
        "log_correlation": complement["log_residual_pearson_r"],
        "query_receipt": ("- [Historical query replay and single-house inference check](results/query-batch-check.json)\n"
                          if (RESULTS / "query-batch-check.json").is_file() else ""),
        "foundation_tr": ("Dış değerlendirmede TabPFNv2 ayrıca ölçüldü." if has_foundation
                          else "Dış değerlendirme kaydında TabPFNv2 puanı bulunmuyor."),
        "foundation_en": ("The outer predictions include TabPFNv2." if has_foundation
                          else "The outer predictions contain no TabPFNv2 score."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", required=True, help="Replace both final README files")
    parser.parse_args()
    release, summary, frozen, audits, outer, sale = final_inputs()
    fields = release_fields(release, sale)
    fields.update(technical_fields(summary, frozen, audits, outer))
    targets = ((DRAFT / "README.tr.md.in", ROOT / "README.md"), (DRAFT / "README.v3.en.md.in", V3 / "README.md"))
    rendered = {path: template.read_text(encoding="utf-8").format_map(fields) for template, path in targets}
    for path, content in rendered.items():
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
