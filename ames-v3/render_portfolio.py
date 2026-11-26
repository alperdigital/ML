"""Fill the portfolio's static evidence fields from the release and nested experiment."""
import argparse
import csv
import gzip
import html
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ames-v3/results"
PAGES = ("index.html", "proje-detaylari.html")
FIELD = re.compile(r'(<(?P<tag>[a-z0-9]+)\b[^>]*\bdata-result="(?P<key>[a-z0-9_]+)"[^>]*>)[^<]*(</(?P=tag)>)')
MODEL_NAMES = {"lightgbm": "LightGBM", "catboost": "CatBoost", "xgboost": "XGBoost",
               "ridge": "Ridge", "tabpfn_v2": "TabPFNv2", "weighted_blend": "Ağırlıklı karışım",
               "equal_blend": "Eşit ağırlıklı karışım", "v1_xgb": "V1 XGBoost",
               "v2_cat": "İç aramada seçilen CatBoost", "v2": "İç aramada seçilen V2"}


def tr(value, digits):
    return f"{value:,.{digits}f}".translate(str.maketrans(".,", ",."))


def currency(value):
    return tr(value, 2) + " $"


def pct(value, signed=False):
    sign = "−" if value < 0 else "+" if signed and value > 0 else ""
    return sign + "%" + tr(abs(value), 2)


def label(model):
    return MODEL_NAMES.get(model, model.replace("_", " "))


def recipe(audit):
    name = audit["recipe"]
    return label(audit["best_new"] if name == "best_new" else name)


def evidence():
    report = json.loads((RESULTS / "three-version-holdout.json").read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())
    scores, diff = report["metrics"], report["differences"]
    v1_results = json.loads((ROOT / "ames-v2/v1-v2-comparison.json").read_text())
    v1 = {r["model"]: r for r in v1_results["candidates"] if r["version"] == "v1"}
    v1_winner = v1_results["selected_by_cv"]["v1"]
    v1_other, = set(v1) - {v1_winner}
    v2_search = json.loads((ROOT / "ames-v2/results.json").read_text())
    v2_winner = min(v2_search["candidates"], key=lambda row: row["cv_rmsle"])
    v2_other = next(row for row in v2_search["candidates"] if row["model"] == v2_winner["model"]
                    and row["engineered"] != v2_winner["engineered"])
    with gzip.open(RESULTS / "split-audits.json.gz", "rt") as h:
        splits = json.load(h)
    schema = report.get("input_schema")
    if not schema or any(r.get("input_schema") != schema for r in (v2_search, v1_results, summary, *splits.values())):
        raise ValueError("Publication requires one consistent input schema across all experiments")
    with (RESULTS / "three-version-holdout.csv").open() as h:
        sale = next(r for r in csv.DictReader(h) if r["Id"] == "899")
    d21, d31, d32 = (diff[k] for k in ("v2_minus_v1", "v3_minus_v1", "v3_minus_v2"))
    band = d32["paired_interval"]
    lo, hi = band["rmsle_gain_pct_ci95"]
    def direction(x):
        return "azaldı" if x < 0 else "arttı" if x > 0 else "değişmedi"
    counts = Counter(recipe(audit) for name, audit in splits.items() if name.startswith("r"))
    selection = ", ".join(f"{name} ({n} bölüm)" for name, n in counts.most_common())
    ranks = sum(scores["v3"][m] <= min(scores[v][m] for v in ("v1", "v2")) for m in ("mae","rmsle"))
    ranks += scores["v3"]["r2"] >= max(scores[v]["r2"] for v in ("v1", "v2"))
    m = summary["outer"]["model_metrics"]
    def ablation(before, after):
        change = 100 * (m["cand_"+after]["rmsle"] / m["cand_"+before]["rmsle"] - 1)
        return pct(abs(change)) + (" artırdı" if change > 0 else " azalttı" if change < 0 else " değiştirdi (0 fark)")
    out = {
        "d31_mae": currency(-d31["mae_usd"]), "d31_mae_pct": pct(-d31["mae_pct"]),
        "d32_rmsle_pct": pct(-d32["rmsle_pct"]),
        "gain_interval": f"{pct(lo, True)} ile {pct(hi, True)}",
        "gain_caveat": ("Bu aralık sıfırı içeriyor. RMSLE farkının yönü bu örneklemde kesinleşmiyor."
                        if lo <= 0 <= hi else "Bu aralık, eğitim belirsizliği hariç, sabit tahminler için hesaplandı."),
        "mae_interval": "–".join(tr(x, 0) for x in band["mae_reduction_usd_ci95"]) + " dolar",
        "v2_tradeoff": f"V1’e göre ev başına ortalama mutlak hata {currency(abs(d21['mae_usd']))} {direction(d21['mae_usd'])}.",
        "v2_delta": (f"V1’e göre RMSLE {pct(abs(d21['rmsle_pct']))} {direction(d21['rmsle'])}. "
                     f"R² {tr(abs(d21['r2_points']), 5)} puan {direction(d21['r2_points'])}."),
        "v3_delta": (f"Beş katlı V2’ye göre RMSLE {pct(abs(d32['rmsle_pct']))} {direction(d32['rmsle'])}. "
                     f"Ortalama mutlak hata ev başına {currency(abs(d32['mae_usd']))} {direction(d32['mae_usd'])}."),
        "v3_outcome": (f"Son seçim {recipe(splits['final'])} oldu. V3, ortak 292 evde "
                       + ["üç ölçütte de diğer sürümlerin gerisinde kaldı.", "bir ölçütte öne çıktı.",
                          "iki ölçütte öne çıktı.", "üç ölçütte de en iyi sonucu verdi."][ranks]),
        "v3_selections": f"On dış bölümde seçilen yöntemler {selection}. Son seçim {recipe(splits['final'])}.",
        "house_actual": currency(float(sale["SalePrice"])),
        "v1_cv": tr(v1[v1_winner]["cv_rmsle"], 5),
        "v1_other": label(v1_other),
        "v1_other_holdout": tr(v1[v1_other]["holdout"]["rmsle"], 5),
        "v2_cv": tr(v2_winner["cv_rmsle"], 5),
        "v2_feature_choice": "açık" if v2_winner["engineered"] else "kapalı",
        "v2_other_cv": tr(v2_other["cv_rmsle"], 5),
        "gate_text": ("V3 için önceden belirlenen iki seçim koşulu da karşılandı." if summary["promote_v3"]
                      else "Sonuçlar V3’e geçmek için baştan belirlenen koşulları birlikte karşılamadı."),
        "structural_delta": ablation("native_plain", "native_structural"),
        "frontage_delta": ablation("native_structural", "native_neighborhood"),
    }
    for v, metrics in scores.items():
        out[v+"_model"] = label(report["models"][v])
        out[v+"_rmsle"] = tr(metrics["rmsle"], 5)
        out[v+"_r2"] = tr(metrics["r2"], 5)
        out[v+"_mae"] = currency(metrics["mae"])
        out["house_"+v] = currency(float(sale[v]))
    for cohort in ("outer", "temporal"):
        for v in ("v2", "v3"):
            out[cohort+"_"+v] = tr(summary[cohort][v]["rmsle"], 5)
    return out


def refresh(check=False):
    fields = evidence()
    for name in PAGES:
        page = ROOT / "docs" / name
        original = page.read_text()
        def substitute(m):
            return m.group(1) + html.escape(fields[m["key"]]) + m.group(4)
        updated = FIELD.sub(substitute, original)
        if updated != original:
            if check:
                raise ValueError(f"Portfolio figures differ from experiment records in {page.name}")
            page.write_text(updated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the already rendered site")
    refresh(parser.parse_args().check)
