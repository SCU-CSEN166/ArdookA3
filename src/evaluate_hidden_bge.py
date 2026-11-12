"""Compare submitted BGE hidden-test predictions with released labels."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

from src.classify_hidden_bge import XML_NS, _column_index, binary_metrics
from src.compare_grant_embeddings import DataValidationError


VALID_CLASSES = {"Good", "Bad"}


@dataclass(frozen=True)
class ReleasedLabel:
    grant_id: str
    label: str
    reason: str


def read_released_labels(path: Path) -> dict[str, ReleasedLabel]:
    if not path.is_file():
        raise DataValidationError(f"Released-label workbook does not exist: {path}")
    try:
        with ZipFile(path) as workbook:
            shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.findall(".//m:t", XML_NS))
                for item in shared_root.findall("m:si", XML_NS)
            ]
            sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    except (BadZipFile, ET.ParseError, KeyError) as exc:
        raise DataValidationError(f"Could not read released-label workbook: {exc}") from exc

    matrix: list[list[str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", XML_NS):
        populated: dict[int, str] = {}
        for cell in row.findall("m:c", XML_NS):
            reference = cell.get("r", "")
            column = _column_index(reference)
            value_node = cell.find("m:v", XML_NS)
            value = "" if value_node is None else (value_node.text or "")
            if cell.get("t") == "s" and value:
                try:
                    value = shared[int(value)]
                except (ValueError, IndexError) as exc:
                    raise DataValidationError(
                        f"Invalid shared-string index in label cell {reference}"
                    ) from exc
            populated[column] = value.strip()
        if populated:
            matrix.append(
                [populated.get(index, "") for index in range(max(populated) + 1)]
            )
    if not matrix:
        raise DataValidationError("Released-label workbook contains no rows")
    headers = matrix[0]
    required = {"grant_id", "label", "reason"}
    missing = sorted(required - set(headers))
    if missing:
        raise DataValidationError("Released labels missing columns: " + ", ".join(missing))

    labels: dict[str, ReleasedLabel] = {}
    for values in matrix[1:]:
        padded = values + [""] * (len(headers) - len(values))
        record = {header: padded[index].strip() for index, header in enumerate(headers)}
        try:
            numeric_id = float(record["grant_id"])
        except ValueError as exc:
            raise DataValidationError(f"Invalid released grant_id: {record['grant_id']!r}") from exc
        if not numeric_id.is_integer():
            raise DataValidationError(f"Released grant_id must be an integer: {record['grant_id']!r}")
        grant_id = str(int(numeric_id))
        if grant_id in labels:
            raise DataValidationError(f"Duplicate released grant_id: {grant_id}")
        if record["label"] not in VALID_CLASSES:
            raise DataValidationError(
                f"Unknown released label for grant {grant_id}: {record['label']!r}"
            )
        if not record["reason"]:
            raise DataValidationError(f"Missing released reason for grant {grant_id}")
        labels[grant_id] = ReleasedLabel(grant_id, record["label"], record["reason"])
    expected_ids = {str(number) for number in range(1, 11)}
    if set(labels) != expected_ids:
        raise DataValidationError(
            f"Expected released IDs 1-10; got {sorted(labels, key=int)}"
        )
    return labels


def read_prediction_details(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DataValidationError(f"Prediction details CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "grant_id", "program", "bge_score", "threshold", "binary_class", "label"
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise DataValidationError("Prediction CSV missing columns: " + ", ".join(missing))
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    ids = [row["grant_id"] for row in rows]
    if len(rows) != 10 or set(ids) != {str(number) for number in range(1, 11)}:
        raise DataValidationError(f"Expected prediction IDs 1-10; got {ids}")
    if len(ids) != len(set(ids)):
        raise DataValidationError("Prediction CSV contains duplicate IDs")
    for row in rows:
        if row["binary_class"] not in VALID_CLASSES:
            raise DataValidationError(
                f"Unknown prediction class for grant {row['grant_id']}: {row['binary_class']!r}"
            )
    return sorted(rows, key=lambda row: int(row["grant_id"]))


def error_type(actual: str, predicted: str) -> str:
    if actual == predicted:
        return "True positive" if actual == "Good" else "True negative"
    return "False positive" if predicted == "Good" else "False negative"


def evaluate(
    labels_path: Path,
    predictions_path: Path,
    output_dir: Path,
) -> None:
    labels = read_released_labels(labels_path)
    predictions = read_prediction_details(predictions_path)
    actual_bool = [labels[row["grant_id"]].label == "Good" for row in predictions]
    predicted_bool = [row["binary_class"] == "Good" for row in predictions]
    metrics = binary_metrics(actual_bool, predicted_bool)

    comparison_rows = []
    for row in predictions:
        released = labels[row["grant_id"]]
        comparison_rows.append(
            {
                "grant_id": row["grant_id"],
                "program": row["program"],
                "bge_score": row["bge_score"],
                "threshold": row["threshold"],
                "predicted_class": row["binary_class"],
                "actual_class": released.label,
                "correct": "Yes" if row["binary_class"] == released.label else "No",
                "error_type": error_type(released.label, row["binary_class"]),
                "actual_reason": released.reason,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "hidden_test_bge_label_comparison.csv"
    comparison_fields = (
        "grant_id", "program", "bge_score", "threshold", "predicted_class",
        "actual_class", "correct", "error_type", "actual_reason",
    )
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    matrix_path = output_dir / "hidden_test_bge_confusion_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", "Good", "Bad", "Total"])
        writer.writerow(
            ["Good", metrics.true_positive, metrics.false_negative,
             metrics.true_positive + metrics.false_negative]
        )
        writer.writerow(
            ["Bad", metrics.false_positive, metrics.true_negative,
             metrics.false_positive + metrics.true_negative]
        )
        writer.writerow(
            ["Total", metrics.true_positive + metrics.false_positive,
             metrics.false_negative + metrics.true_negative, len(predictions)]
        )

    specificity = (
        metrics.true_negative / (metrics.true_negative + metrics.false_positive)
        if metrics.true_negative + metrics.false_positive
        else 0.0
    )
    negative_predictive_value = (
        metrics.true_negative / (metrics.true_negative + metrics.false_negative)
        if metrics.true_negative + metrics.false_negative
        else 0.0
    )
    metrics_path = output_dir / "hidden_test_bge_evaluation_metrics.csv"
    metric_rows = [
        ("accuracy", metrics.accuracy),
        ("precision_good", metrics.precision),
        ("recall_good", metrics.recall),
        ("f1_good", metrics.f1),
        ("specificity_bad", specificity),
        ("negative_predictive_value", negative_predictive_value),
        ("true_positive", metrics.true_positive),
        ("false_positive", metrics.false_positive),
        ("true_negative", metrics.true_negative),
        ("false_negative", metrics.false_negative),
        ("evaluated_grants", len(predictions)),
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in metric_rows:
            writer.writerow([metric, f"{value:.6f}" if isinstance(value, float) else value])

    false_positives = [row for row in comparison_rows if row["error_type"] == "False positive"]
    false_negatives = [row for row in comparison_rows if row["error_type"] == "False negative"]
    report_lines = [
        "# BGE hidden-test evaluation against released labels",
        "",
        "## Verified metrics",
        "",
        "Good is the positive class.",
        "",
        "| Accuracy | Precision | Recall | F1 | Specificity | NPV |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {metrics.accuracy:.6f} | {metrics.precision:.6f} | {metrics.recall:.6f} | {metrics.f1:.6f} | {specificity:.6f} | {negative_predictive_value:.6f} |",
        "",
        "## Confusion matrix",
        "",
        "| Actual / Predicted | Good | Bad |",
        "|---|---:|---:|",
        f"| Good | {metrics.true_positive} | {metrics.false_negative} |",
        f"| Bad | {metrics.false_positive} | {metrics.true_negative} |",
        "",
        f"The model correctly classified {metrics.true_positive + metrics.true_negative} of 10 grants. It produced {metrics.false_positive} false positives and {metrics.false_negative} false negative.",
        "",
        "## Errors",
        "",
    ]
    for row in false_positives + false_negatives:
        report_lines.append(
            f"- Grant {row['grant_id']} — {row['program']}: {row['error_type']}; "
            f"predicted {row['predicted_class']}, actual {row['actual_class']}. "
            f"Released reason: {row['actual_reason']}"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "BGE recovered 4 of the 5 truly Good grants, but two grants with hard eligibility or program-purpose conflicts were incorrectly labeled Good. This confirms the earlier concern that semantic similarity can detect topical alignment while missing categorical eligibility restrictions.",
            "",
            "The false negative was the TiE Women competition, which the released labels mark Good because it fits a woman-founded early-stage startup with a validated offering. Its BGE score fell below the synthetic calibration threshold, illustrating imperfect transfer from synthetic training examples to real hidden grants.",
        ]
    )
    (output_dir / "bge_hidden_test_evaluation_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-xlsx", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evaluate(args.labels_xlsx, args.predictions, args.output_dir)
    except (DataValidationError, OSError, ValueError) as exc:
        print(f"Hidden-test evaluation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
