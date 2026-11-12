"""BGE-only binary classification for the professor's 10 hidden grants.

The binary threshold is calibrated on the synthetic Train split, never on the
hidden grants. Deadlines and current status are deliberately ignored as directed
by the hidden-test guide.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import logging
import math
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

from src.compare_grant_embeddings import (
    DataValidationError,
    ModelInferenceError,
    build_model_text,
    default_model_factory,
    read_csv_records,
    validate_and_join_synthetic,
)


LOGGER = logging.getLogger("bge_hidden_test")
MODEL_IDENTIFIER = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
REQUIRED_HIDDEN_COLUMNS = {
    "Grant ID",
    "Program",
    "Type",
    "Core Requirements",
}
POSITIVE_FITS = {"Excellent", "Strong"}
XML_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


@dataclass(frozen=True)
class HiddenPrediction:
    grant_id: str
    program: str
    score: float
    label: str
    binary_class: str
    reason: str


def _column_index(cell_reference: str) -> int:
    letters = "".join(character for character in cell_reference if character.isalpha())
    if not letters:
        raise DataValidationError(f"Invalid Excel cell reference: {cell_reference!r}")
    index = 0
    for letter in letters.upper():
        index = index * 26 + (ord(letter) - ord("A") + 1)
    return index - 1


def read_hidden_xlsx(path: Path) -> list[dict[str, str]]:
    """Read the first worksheet using the OOXML standard library only."""
    if not path.is_file():
        raise DataValidationError(f"Hidden-test workbook does not exist: {path}")
    try:
        with ZipFile(path) as workbook:
            names = set(workbook.namelist())
            if "xl/worksheets/sheet1.xml" not in names:
                raise DataValidationError("Hidden workbook has no first worksheet")
            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in names:
                root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
                shared_strings = [
                    "".join(node.text or "" for node in item.findall(".//m:t", XML_NS))
                    for item in root.findall("m:si", XML_NS)
                ]
            sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    except (BadZipFile, ET.ParseError, KeyError) as exc:
        raise DataValidationError(f"Could not read hidden-test workbook: {exc}") from exc

    matrix: list[list[str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", XML_NS):
        populated: dict[int, str] = {}
        for cell in row.findall("m:c", XML_NS):
            reference = cell.get("r", "")
            column = _column_index(reference)
            cell_type = cell.get("t")
            if cell_type == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.findall(".//m:t", XML_NS)
                )
            else:
                value_node = cell.find("m:v", XML_NS)
                value = "" if value_node is None else (value_node.text or "")
                if cell_type == "s" and value:
                    try:
                        value = shared_strings[int(value)]
                    except (ValueError, IndexError) as exc:
                        raise DataValidationError(
                            f"Invalid shared-string index in cell {reference}"
                        ) from exc
            populated[column] = value.strip()
        if populated:
            width = max(populated) + 1
            matrix.append([populated.get(index, "") for index in range(width)])

    if not matrix:
        raise DataValidationError("Hidden workbook contains no data rows")
    headers = matrix[0]
    missing_columns = sorted(REQUIRED_HIDDEN_COLUMNS - set(headers))
    if missing_columns:
        raise DataValidationError(
            "Hidden workbook is missing required columns: " + ", ".join(missing_columns)
        )

    records: list[dict[str, str]] = []
    for values in matrix[1:]:
        padded = values + [""] * (len(headers) - len(values))
        record = {header: padded[index].strip() for index, header in enumerate(headers)}
        if not any(record.values()):
            continue
        raw_id = record.get("Grant ID", "")
        try:
            numeric_id = float(raw_id)
        except ValueError as exc:
            raise DataValidationError(f"Invalid hidden Grant ID: {raw_id!r}") from exc
        if not numeric_id.is_integer():
            raise DataValidationError(f"Hidden Grant ID must be an integer: {raw_id!r}")
        record["Grant ID"] = str(int(numeric_id))
        records.append(record)

    ids = [record["Grant ID"] for record in records]
    if len(records) != 10 or set(ids) != {str(number) for number in range(1, 11)}:
        raise DataValidationError(
            f"Expected exactly hidden Grant IDs 1-10; got {ids}"
        )
    if len(ids) != len(set(ids)):
        raise DataValidationError("Hidden workbook contains duplicate Grant IDs")
    missing_values = [
        f"{record['Grant ID']}:{field}"
        for record in records
        for field in sorted(REQUIRED_HIDDEN_COLUMNS)
        if not record.get(field, "").strip()
    ]
    if missing_values:
        raise DataValidationError(
            "Hidden workbook has missing required values: " + ", ".join(missing_values)
        )
    return sorted(records, key=lambda record: int(record["Grant ID"]))


def binary_metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> BinaryMetrics:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Labels and predictions must have the same nonzero length")
    tp = sum(expected and predicted for expected, predicted in zip(labels, predictions))
    fp = sum(not expected and predicted for expected, predicted in zip(labels, predictions))
    tn = sum(not expected and not predicted for expected, predicted in zip(labels, predictions))
    fn = sum(expected and not predicted for expected, predicted in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        accuracy=(tp + tn) / len(labels),
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def select_f1_threshold(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float, BinaryMetrics]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("Scores and labels must have the same nonzero length")
    numeric_scores = [float(score) for score in scores]
    if not all(math.isfinite(score) for score in numeric_scores):
        raise ModelInferenceError("Cannot calibrate a threshold from non-finite scores")
    unique = sorted(set(numeric_scores))
    epsilon = 1e-12
    candidates = [unique[0] - epsilon]
    candidates.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    candidates.append(unique[-1] + epsilon)

    evaluated = []
    for threshold in candidates:
        metrics = binary_metrics(labels, [score >= threshold for score in numeric_scores])
        evaluated.append((metrics.f1, metrics.accuracy, metrics.precision, threshold, metrics))
    _, _, _, threshold, metrics = max(evaluated, key=lambda item: item[:4])
    return threshold, metrics


def _as_vectors(encoded: Any) -> list[list[float]]:
    values = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    return [[float(component) for component in vector] for vector in values]


def _bge_scores(model: Any, mission: str, rows: Sequence[Mapping[str, str]]) -> list[float]:
    kwargs = {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    query = BGE_QUERY_PREFIX + mission
    query_vector = _as_vectors(model.encode([query], **kwargs))[0]
    documents = [build_model_text(row) for row in rows]
    document_vectors = _as_vectors(model.encode(documents, **kwargs))
    scores = [
        float(sum(left * right for left, right in zip(query_vector, document_vector)))
        for document_vector in document_vectors
    ]
    if not all(math.isfinite(score) for score in scores):
        raise ModelInferenceError("BGE produced a non-finite hidden-test score")
    return scores


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _write_csv(path: Path, predictions: Sequence[HiddenPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("grant_id", "label", "reason"))
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "grant_id": prediction.grant_id,
                    "label": prediction.label,
                    "reason": prediction.reason,
                }
            )


def _write_details(path: Path, predictions: Sequence[HiddenPrediction], threshold: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("grant_id", "program", "bge_score", "threshold", "binary_class", "label")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "grant_id": prediction.grant_id,
                    "program": prediction.program,
                    "bge_score": f"{prediction.score:.9f}",
                    "threshold": f"{threshold:.9f}",
                    "binary_class": prediction.binary_class,
                    "label": prediction.label,
                }
            )


def _write_report(
    path: Path,
    predictions: Sequence[HiddenPrediction],
    threshold: float,
    calibration: BinaryMetrics,
    calibration_positive: int,
    inference_seconds: float,
) -> None:
    relevant_count = sum(prediction.label == "Relevant" for prediction in predictions)
    lines = [
        "# BGE hidden-test binary classification report",
        "",
        "## Status",
        "",
        "**COMPLETED with real BGE inference on all 10 hidden grants.**",
        "",
        "The professor's expected hidden labels were not included, so hidden-test accuracy, precision, recall, and F1 are not yet available. The table below contains predictions, not verified outcomes.",
        "",
        "## Method",
        "",
        f"- Model: `{MODEL_IDENTIFIER}` only; no MiniLM inference was used.",
        "- Model input fields: `Program`, `Type`, and `Core Requirements` only.",
        "- Deadlines and current status were ignored exactly as instructed.",
        "- BGE's retrieval instruction was applied to the mission query only.",
        "- Embeddings were normalized and scored with normalized dot product (cosine similarity).",
        "- Binary target used for calibration: synthetic Train row is Relevant when `Eligibility label == Eligible` and `Thematic fit` is Excellent or Strong. Availability was ignored.",
        f"- Calibration set: 160 synthetic Train rows, including {calibration_positive} positive rows.",
        f"- Frozen decision threshold: **{threshold:.9f}**. Score at or above the threshold maps to Relevant/Good; lower maps to Not relevant/Bad.",
        "- Threshold selection maximized positive-class F1 on the synthetic Train split, with accuracy, precision, and then the higher threshold as deterministic tie-breakers.",
        "",
        "## Synthetic Train calibration metrics",
        "",
        "These are development/calibration metrics, not hidden-test metrics.",
        "",
        "| Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {calibration.accuracy:.6f} | {calibration.precision:.6f} | {calibration.recall:.6f} | {calibration.f1:.6f} | {calibration.true_positive} | {calibration.false_positive} | {calibration.true_negative} | {calibration.false_negative} |",
        "",
        "## Hidden predictions",
        "",
        f"BGE classified {relevant_count} of 10 grants as Relevant/Good and {10 - relevant_count} as Not relevant/Bad.",
        "",
        "| ID | Program | BGE score | Binary class | Submission label |",
        "|---:|---|---:|---|---|",
    ]
    for prediction in predictions:
        lines.append(
            f"| {prediction.grant_id} | {prediction.program.replace('|', '/')} | "
            f"{prediction.score:.6f} | {prediction.binary_class} | {prediction.label} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "The binary output is produced solely by thresholding BGE semantic similarity. BGE is not a verified rules engine and may miss hard eligibility restrictions that appear in the requirements. Therefore, these predictions satisfy the requested BGE-only experiment but should not be treated as verified grant eligibility decisions.",
            "",
            f"The synthetic calibration was weak: accuracy was {calibration.accuracy:.6f}, precision was {calibration.precision:.6f}, and there were {calibration.false_positive} false positives. This is evidence that a single BGE similarity threshold does not model the combined eligibility-and-fit decision reliably.",
            "",
            "The threshold was calibrated on synthetic labels tied to a hypothetical profile. That calibration may not transfer to the professor's real hidden examples. Once expected labels are released, compute accuracy, precision, recall, and F1 with Relevant as the positive class and analyze every false positive and false negative.",
            "",
            "## Reproducibility",
            "",
            f"- Python: `{platform.python_version()}`",
            f"- sentence-transformers: `{_package_version('sentence-transformers')}`",
            f"- transformers: `{_package_version('transformers')}`",
            f"- torch: `{_package_version('torch')}`",
            f"- CPU encode/classify time, excluding model download/loading: `{inference_seconds:.6f}` seconds",
            "",
            "```bash",
            "python -m src.classify_hidden_bge \\",
            "  --hidden-xlsx Ardook_hidden_test_grants.xlsx \\",
            "  --synthetic data/raw/Synthetic_Ardook_ML_Ready_200.csv \\",
            "  --labels data/raw/Synthetic_Ardook_Ground_Truth_200.csv \\",
            "  --mission config/hypothetical_mission.txt \\",
            "  --output-dir results/ \\",
            "  --low-disk-mode",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_hidden_test(
    hidden_xlsx: Path,
    synthetic_path: Path,
    labels_path: Path,
    mission_path: Path,
    output_dir: Path,
    device: str = "cpu",
    low_disk_mode: bool = False,
    model_factory: Callable[[str, str], Any] = default_model_factory,
) -> list[HiddenPrediction]:
    hidden_rows = read_hidden_xlsx(hidden_xlsx)
    synthetic_rows, synthetic_headers = read_csv_records(synthetic_path)
    label_rows, label_headers = read_csv_records(labels_path)
    synthetic = validate_and_join_synthetic(
        synthetic_rows, synthetic_headers, label_rows, label_headers
    )
    train_rows = sorted(
        (row for row in synthetic if row["Train/Test split"] == "Train"),
        key=lambda row: row["Grant ID"],
    )
    train_labels = [
        row["Eligibility label"] == "Eligible" and row["Thematic fit"] in POSITIVE_FITS
        for row in train_rows
    ]
    if not mission_path.is_file():
        raise DataValidationError(f"Mission file does not exist: {mission_path}")
    mission = mission_path.read_text(encoding="utf-8").strip()
    if not mission:
        raise DataValidationError("Mission query is blank")

    def infer(factory: Callable[[str, str], Any]) -> tuple[list[float], list[float], float]:
        model = factory(MODEL_IDENTIFIER, device)
        started = time.perf_counter()
        train_scores = _bge_scores(model, mission, train_rows)
        hidden_scores = _bge_scores(model, mission, hidden_rows)
        elapsed = time.perf_counter() - started
        del model
        gc.collect()
        return train_scores, hidden_scores, elapsed

    if low_disk_mode:
        if model_factory is not default_model_factory:
            raise ValueError("low_disk_mode is supported only with the default model factory")
        with tempfile.TemporaryDirectory(prefix="ardook-hidden-bge-") as cache_name:
            cache_path = Path(cache_name)

            def ephemeral_factory(identifier: str, selected_device: str) -> Any:
                return default_model_factory(identifier, selected_device, cache_path)

            train_scores, hidden_scores, elapsed = infer(ephemeral_factory)
    else:
        train_scores, hidden_scores, elapsed = infer(model_factory)

    threshold, calibration = select_f1_threshold(train_scores, train_labels)
    predictions: list[HiddenPrediction] = []
    for row, score in zip(hidden_rows, hidden_scores):
        is_relevant = score >= threshold
        label = "Relevant" if is_relevant else "Not relevant"
        binary_class = "Good" if is_relevant else "Bad"
        operator = ">=" if is_relevant else "<"
        reason = (
            f"BGE similarity {score:.6f} {operator} calibrated threshold "
            f"{threshold:.6f}; classified {label}."
        )
        predictions.append(
            HiddenPrediction(
                grant_id=row["Grant ID"],
                program=row["Program"],
                score=score,
                label=label,
                binary_class=binary_class,
                reason=reason,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "hidden_test_outputs.csv", predictions)
    _write_details(output_dir / "hidden_test_bge_details.csv", predictions, threshold)
    _write_report(
        output_dir / "bge_hidden_test_report.md",
        predictions,
        threshold,
        calibration,
        sum(train_labels),
        elapsed,
    )
    return predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-xlsx", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--low-disk-mode", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        predictions = run_hidden_test(
            args.hidden_xlsx,
            args.synthetic,
            args.labels,
            args.mission,
            args.output_dir,
            args.device,
            args.low_disk_mode,
        )
    except (DataValidationError, ModelInferenceError, OSError, ValueError) as exc:
        LOGGER.error("Hidden-test classification failed: %s", exc)
        return 2
    LOGGER.info(
        "Wrote %d BGE hidden predictions (%d Relevant, %d Not relevant) to %s",
        len(predictions),
        sum(prediction.label == "Relevant" for prediction in predictions),
        sum(prediction.label == "Not relevant" for prediction in predictions),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
