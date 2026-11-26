"""Guard the ID, price and metric contract between the portfolio and its reports."""
import csv
import json
import math
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScoreTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_body = False
        self.cell = None
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "tbody":
            self.in_body = True
        if self.in_body and tag == "tr":
            self.rows.append([])
        if self.in_body and tag == "td":
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.rows[-1].append("".join(self.cell).strip())
            self.cell = None
        if tag == "tbody":
            self.in_body = False


def turkish_number(value, digits):
    return f"{value:,.{digits}f}".translate(str.maketrans(".,", ",."))


class ReleaseReportTests(unittest.TestCase):
    def test_portfolio_numbers_and_examples_agree_with_experiments(self):
        from render_portfolio import refresh
        refresh(check=True)

    def test_interval_pairs_single_column_prices(self):
        from compare_releases import paired_intervals
        actual = [100, 120, 135]
        old, new = [90, 115, 140], [95, 118, 137]
        expected = paired_intervals(actual, old, new, draws=200)
        self.assertEqual(expected, paired_intervals([[x] for x in actual], old, [[x] for x in new], draws=200))
        with self.assertRaises(ValueError):
            paired_intervals(actual, old[:2], new)
        with self.assertRaisesRegex(ValueError, "zero-error"):
            paired_intervals(actual, actual, new)

    def test_three_published_views_match_full_precision_predictions(self):
        with (ROOT / "ames-v3/results/three-version-holdout.csv").open() as handle:
            predictions = list(csv.DictReader(handle))
        with (ROOT / "data/train.csv").open() as handle:
            raw = {int(row["Id"]): row for row in csv.DictReader(handle)}
        by_id = {int(row["Id"]): row for row in predictions}
        web = json.loads((ROOT / "docs/house-results.json").read_text())
        self.assertEqual(len(predictions), 292)
        self.assertEqual(len(by_id), 292)
        self.assertEqual([row[0] for row in web], sorted(by_id))
        for house, area, year, actual, *prices in web:
            self.assertEqual([area, year, actual], [
                raw[house]["Neighborhood"], int(raw[house]["YrSold"]), float(raw[house]["SalePrice"])])
            self.assertEqual(actual, float(by_id[house]["SalePrice"]))
            self.assertEqual(prices, [round(float(by_id[house][key]), 2) for key in ("v1", "v2", "v3")])

        table = ScoreTable()
        table.feed((ROOT / "docs/index.html").read_text())
        readme = (ROOT / "README.md").read_text().splitlines()
        scores = json.loads((ROOT / "ames-v3/results/three-version-holdout.json").read_text())["metrics"]
        actual = [float(row["SalePrice"]) for row in predictions]
        mean = sum(actual) / len(actual)
        denominator = sum((y - mean) ** 2 for y in actual)
        self.assertEqual(len(table.rows), 3)
        for version, cells in zip(("v1", "v2", "v3"), table.rows):
            estimate = [float(row[version]) for row in predictions]
            paired = list(zip(actual, estimate))
            rmsle = math.sqrt(sum((math.log1p(y) - math.log1p(p)) ** 2 for y, p in paired) / len(paired))
            mae = sum(abs(y-p) for y, p in paired) / len(paired)
            r2 = 1 - sum((y-p)**2 for y, p in paired) / denominator
            for key, value in (("rmsle", rmsle), ("mae", mae), ("r2", r2)):
                self.assertAlmostEqual(value, scores[version][key], places=8)
            labels = [turkish_number(rmsle, 5), turkish_number(mae, 2), turkish_number(r2, 5)]
            self.assertEqual(cells, [labels[0], labels[1] + " $", labels[2]])
            row = next(line for line in readme if line.startswith("| " + version.upper() + " "))
            for label in labels:
                self.assertIn(label, row)
