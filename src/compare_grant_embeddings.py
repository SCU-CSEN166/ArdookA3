"""Reproducible MiniLM versus BGE grant-relevance ranking experiment.

Only Program, Type, and Core Requirements are passed to an embedding model.
Outcome and status columns are retained solely for evaluation and diagnostics.
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


LOGGER = logging.getLogger("grant_embedding_comparison")

ALLOWED_MODEL_FIELDS = ("Program", "Type", "Core Requirements")
FORBIDDEN_MODEL_FIELDS = frozenset(
    {
        "Fit",
        "Why It Fits Ardook Data",
        "Key Concern / Disqualifier",
        "Recommended Action",
        "Current Status (as last reported)",
        "Eligibility label",
        "Availability label",
        "Thematic fit",
        "Recommendation label",
        "Scenario family",
        "Eligibility rationale",
        "Hard negative?",
        "Train/Test split",
        "Official Source",
    }
)
FIT_GRADES = {"Excellent": 3, "Strong": 2, "Moderate": 1, "Weak": 0}
EXPECTED_SPLITS = {"Train": 160, "Test": 40}
EXPECTED_ELIGIBILITY = {"Eligible": 120, "Ineligible": 60, "Needs review": 20}

SYNTHETIC_REQUIRED = frozenset(
    {
        "Grant ID",
        "Program",
        "Type",
        "Core Requirements",
        "Eligibility label",
        "Availability label",
        "Thematic fit",
        "Recommendation label",
        "Scenario family",
        "Eligibility rationale",
        "Hard negative?",
        "Train/Test split",
    }
)
LABEL_REQUIRED = frozenset(
    {
        "Grant ID",
        "Program",
        "Eligibility label",
        "Availability label",
        "Thematic fit",
        "Recommendation label",
        "Scenario family",
        "Eligibility rationale",
        "Hard negative?",
        "Train/Test split",
    }
)
ORIGINAL_REQUIRED = frozenset(ALLOWED_MODEL_FIELDS) | {
    "Fit",
    "Current Status (as last reported)",
}


class DataValidationError(ValueError):
    """Raised when an input file violates the experiment contract."""


class ModelInferenceError(RuntimeError):
    """Raised when models cannot be loaded or inference cannot complete."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    identifier: str
    query_instruction: str = ""


@dataclass
class ModelRun:
    spec: ModelSpec
    parameter_count: int | None
    inference_seconds: float
    test_scores: dict[str, float]
    primary_ranks: dict[str, int]
    unfiltered_ranks: dict[str, int]
    original_scores: dict[str, float]
    original_ranks: dict[str, int]
    ndcg_at_5: float
    precision_at_5: float
    spearman: float | None
    unfiltered_top5_ineligible: int
    unfiltered_top5_non_open: int


def read_csv_records(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise DataValidationError(f"Input file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise DataValidationError(f"CSV has no header: {path}")
        rows = []
        for row_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise DataValidationError(f"Extra unnamed values in {path} row {row_number}")
            rows.append({key: (value or "").strip() for key, value in raw.items()})
    return rows, list(reader.fieldnames)


def _require_columns(headers: Sequence[str], required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(headers))
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {', '.join(missing)}")


def _index_unique(rows: Sequence[Mapping[str, str]], name: str) -> dict[str, Mapping[str, str]]:
    missing_rows = [str(i) for i, row in enumerate(rows, start=2) if not row.get("Grant ID", "").strip()]
    if missing_rows:
        raise DataValidationError(f"{name} has blank Grant ID at CSV row(s): {', '.join(missing_rows)}")
    counts = Counter(row["Grant ID"] for row in rows)
    duplicates = sorted(grant_id for grant_id, count in counts.items() if count > 1)
    if duplicates:
        raise DataValidationError(f"{name} has duplicate Grant ID(s): {', '.join(duplicates)}")
    return {row["Grant ID"]: row for row in rows}


def validate_and_join_synthetic(
    synthetic_rows: Sequence[dict[str, str]],
    synthetic_headers: Sequence[str],
    label_rows: Sequence[dict[str, str]],
    label_headers: Sequence[str],
) -> list[dict[str, str]]:
    """Validate the two 200-row files and join labels by Grant ID, never position."""
    _require_columns(synthetic_headers, SYNTHETIC_REQUIRED, "Synthetic ML-ready CSV")
    _require_columns(label_headers, LABEL_REQUIRED, "Ground-truth CSV")
    if len(synthetic_rows) != 200 or len(label_rows) != 200:
        raise DataValidationError(
            f"Expected 200 rows in each synthetic file; got {len(synthetic_rows)} and {len(label_rows)}"
        )
    synthetic_by_id = _index_unique(synthetic_rows, "Synthetic ML-ready CSV")
    labels_by_id = _index_unique(label_rows, "Ground-truth CSV")
    if set(synthetic_by_id) != set(labels_by_id):
        only_ml = sorted(set(synthetic_by_id) - set(labels_by_id))
        only_labels = sorted(set(labels_by_id) - set(synthetic_by_id))
        raise DataValidationError(
            f"Grant ID sets differ (ML-only={only_ml[:5]}, labels-only={only_labels[:5]})"
        )

    shared_ground_truth = sorted((set(label_headers) & set(synthetic_headers)) - {"Grant ID"})
    disagreements: list[str] = []
    for grant_id in sorted(synthetic_by_id):
        ml_row = synthetic_by_id[grant_id]
        gt_row = labels_by_id[grant_id]
        for field in shared_ground_truth:
            if ml_row[field] != gt_row[field]:
                disagreements.append(f"{grant_id}:{field}")
    if disagreements:
        raise DataValidationError(
            "ML-ready and ground-truth values disagree after Grant ID join: "
            + ", ".join(disagreements[:10])
        )

    required_values = SYNTHETIC_REQUIRED
    missing = [
        f"{row['Grant ID']}:{field}"
        for row in synthetic_rows
        for field in sorted(required_values)
        if not row.get(field, "").strip()
    ]
    if missing:
        raise DataValidationError("Missing required synthetic values: " + ", ".join(missing[:10]))

    split_counts = Counter(row["Train/Test split"] for row in synthetic_rows)
    eligibility_counts = Counter(row["Eligibility label"] for row in synthetic_rows)
    if dict(split_counts) != EXPECTED_SPLITS:
        raise DataValidationError(f"Expected split counts {EXPECTED_SPLITS}; got {dict(split_counts)}")
    if dict(eligibility_counts) != EXPECTED_ELIGIBILITY:
        raise DataValidationError(
            f"Expected eligibility counts {EXPECTED_ELIGIBILITY}; got {dict(eligibility_counts)}"
        )
    unknown_fits = sorted(set(row["Thematic fit"] for row in synthetic_rows) - set(FIT_GRADES))
    if unknown_fits:
        raise DataValidationError(f"Unknown Thematic fit label(s): {', '.join(unknown_fits)}")

    # Preserve the ML-ready row order while proving the labels were matched by ID.
    return [dict(row) for row in synthetic_rows]


def validate_original(rows: Sequence[dict[str, str]], headers: Sequence[str]) -> list[dict[str, str]]:
    _require_columns(headers, ORIGINAL_REQUIRED, "Original tracker CSV")
    if len(rows) != 37:
        raise DataValidationError(f"Expected 37 original grants; got {len(rows)}")
    missing = [
        f"row {i}:{field}"
        for i, row in enumerate(rows, start=2)
        for field in ALLOWED_MODEL_FIELDS
        if not row.get(field, "").strip()
    ]
    if missing:
        raise DataValidationError("Original tracker has missing permitted input values: " + ", ".join(missing[:10]))
    return [dict(row, **{"Original ID": f"ORIG-{i:04d}"}) for i, row in enumerate(rows, start=1)]


def build_model_text(row: Mapping[str, str]) -> str:
    """Build model input from the immutable three-field allowlist."""
    parts = [f"{field}: {row.get(field, '').strip()}" for field in ALLOWED_MODEL_FIELDS if row.get(field, "").strip()]
    if not parts:
        raise DataValidationError("Cannot build model text: all permitted fields are blank")
    return "\n".join(parts)


def fit_to_grade(label: str) -> int:
    try:
        return FIT_GRADES[label]
    except KeyError as exc:
        raise DataValidationError(f"Unknown thematic-fit grade: {label!r}") from exc


def rank_scores(ids: Sequence[str], scores: Sequence[float]) -> tuple[list[str], dict[str, int]]:
    if len(ids) != len(scores):
        raise ValueError("IDs and scores must have equal lengths")
    if len(set(ids)) != len(ids):
        raise ValueError("Ranking IDs must be unique")
    pairs = []
    for item_id, score in zip(ids, scores):
        numeric = float(score)
        if not math.isfinite(numeric):
            raise ModelInferenceError(f"Non-finite score for {item_id}: {numeric}")
        pairs.append((item_id, numeric))
    pairs.sort(key=lambda item: (-item[1], item[0]))
    sorted_scores = [score for _, score in pairs]
    if any(a < b for a, b in zip(sorted_scores, sorted_scores[1:])):
        raise AssertionError("Ranking is not sorted in descending score order")
    ordered = [item_id for item_id, _ in pairs]
    return ordered, {item_id: rank for rank, item_id in enumerate(ordered, start=1)}


def ndcg_at_k(relevances: Sequence[int], k: int = 5) -> float | None:
    if len(relevances) < k:
        return None
    actual = list(relevances[:k])
    ideal = sorted(relevances, reverse=True)[:k]
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(actual))
    idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def precision_at_k(relevances: Sequence[int], k: int = 5, threshold: int = 2) -> float | None:
    if len(relevances) < k:
        return None
    return sum(rel >= threshold for rel in relevances[:k]) / k


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def spearman_correlation(scores: Sequence[float], grades: Sequence[int]) -> float | None:
    if len(scores) != len(grades) or len(scores) < 2:
        return None
    x = _average_ranks([float(value) for value in scores])
    y = _average_ranks([float(value) for value in grades])
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ModelInferenceError("Query and document embedding dimensions differ")
    return float(sum(a * b for a, b in zip(left, right)))


def _as_vectors(encoded: Any) -> list[list[float]]:
    values = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    return [[float(component) for component in vector] for vector in values]


def _encode_scores(model: Any, query: str, texts: Sequence[str]) -> list[float]:
    kwargs = {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    query_vector = _as_vectors(model.encode([query], **kwargs))[0]
    document_vectors = _as_vectors(model.encode(list(texts), **kwargs))
    scores = [_dot(query_vector, vector) for vector in document_vectors]
    if not all(math.isfinite(score) for score in scores):
        raise ModelInferenceError("Model produced a non-finite similarity score")
    return scores


def _parameter_count(model: Any) -> int | None:
    try:
        return sum(parameter.numel() for parameter in model.parameters())
    except (AttributeError, TypeError):
        return None


def default_model_factory(identifier: str, device: str, cache_folder: Path | None = None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ModelInferenceError(
            "sentence-transformers is not installed; run `pip install -r requirements.txt`"
        ) from exc
    try:
        return SentenceTransformer(
            identifier,
            device=device,
            cache_folder=str(cache_folder) if cache_folder is not None else None,
        )
    except Exception as exc:  # model libraries raise several download/backend exception types
        raise ModelInferenceError(f"Could not load {identifier}: {exc}") from exc


def _run_model(
    spec: ModelSpec,
    mission: str,
    test_rows: Sequence[Mapping[str, str]],
    primary_rows: Sequence[Mapping[str, str]],
    original_rows: Sequence[Mapping[str, str]],
    model_factory: Callable[[str, str], Any],
    device: str,
) -> ModelRun:
    LOGGER.info("Loading model %s on %s", spec.identifier, device)
    model = model_factory(spec.identifier, device)
    parameter_count = _parameter_count(model)
    query = spec.query_instruction + mission
    test_ids = [row["Grant ID"] for row in test_rows]
    original_ids = [row["Original ID"] for row in original_rows]
    primary_ids = [row["Grant ID"] for row in primary_rows]

    started = time.perf_counter()  # intentionally after model loading
    test_values = _encode_scores(model, query, [build_model_text(row) for row in test_rows])
    original_values = _encode_scores(model, query, [build_model_text(row) for row in original_rows])
    test_scores = dict(zip(test_ids, test_values))
    original_scores = dict(zip(original_ids, original_values))
    unfiltered_order, unfiltered_ranks = rank_scores(test_ids, test_values)
    primary_order, primary_ranks = rank_scores(primary_ids, [test_scores[item_id] for item_id in primary_ids])
    _, original_ranks = rank_scores(original_ids, original_values)
    elapsed = time.perf_counter() - started

    test_by_id = {row["Grant ID"]: row for row in test_rows}
    primary_relevances = [fit_to_grade(test_by_id[item_id]["Thematic fit"]) for item_id in primary_order]
    ndcg = ndcg_at_k(primary_relevances, 5)
    precision = precision_at_k(primary_relevances, 5)
    if ndcg is None or precision is None:
        raise DataValidationError("Fewer than five eligible/open test candidates; @5 metrics unavailable")
    primary_scores = [test_scores[row["Grant ID"]] for row in primary_rows]
    primary_grades = [fit_to_grade(row["Thematic fit"]) for row in primary_rows]
    top5_rows = [test_by_id[item_id] for item_id in unfiltered_order[:5]]
    result = ModelRun(
        spec=spec,
        parameter_count=parameter_count,
        inference_seconds=elapsed,
        test_scores=test_scores,
        primary_ranks=primary_ranks,
        unfiltered_ranks=unfiltered_ranks,
        original_scores=original_scores,
        original_ranks=original_ranks,
        ndcg_at_5=ndcg,
        precision_at_5=precision,
        spearman=spearman_correlation(primary_scores, primary_grades),
        unfiltered_top5_ineligible=sum(row["Eligibility label"] == "Ineligible" for row in top5_rows),
        unfiltered_top5_non_open=sum(row["Availability label"] != "Open" for row in top5_rows),
    )
    del model
    gc.collect()
    return result


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None, digits: int = 6) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _write_outputs(
    output_dir: Path,
    synthetic_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
    primary_rows: Sequence[Mapping[str, str]],
    original_rows: Sequence[Mapping[str, str]],
    runs: Sequence[ModelRun],
    mission: str,
    bge_instruction_enabled: bool,
    source_paths: Mapping[str, Path],
    device: str,
    low_disk_mode: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_by_key = {run.spec.key: run for run in runs}
    mini, bge = run_by_key["minilm"], run_by_key["bge"]

    metric_fields = [
        "model_key", "model_identifier", "parameter_count", "evaluated_test_count",
        "eligible_open_test_count", "ndcg_at_5", "precision_at_5", "spearman",
        "unfiltered_top5_ineligible_count", "unfiltered_top5_non_open_count",
        "inference_seconds", "device", "bge_query_instruction_used",
    ]
    metric_rows = []
    for run in runs:
        metric_rows.append(
            {
                "model_key": run.spec.key,
                "model_identifier": run.spec.identifier,
                "parameter_count": run.parameter_count if run.parameter_count is not None else "",
                "evaluated_test_count": len(test_rows),
                "eligible_open_test_count": len(primary_rows),
                "ndcg_at_5": _fmt(run.ndcg_at_5),
                "precision_at_5": _fmt(run.precision_at_5),
                "spearman": _fmt(run.spearman),
                "unfiltered_top5_ineligible_count": run.unfiltered_top5_ineligible,
                "unfiltered_top5_non_open_count": run.unfiltered_top5_non_open,
                "inference_seconds": _fmt(run.inference_seconds),
                "device": device,
                "bge_query_instruction_used": bge_instruction_enabled if run.spec.key == "bge" else "not applicable",
            }
        )
    _write_csv(output_dir / "model_comparison_metrics.csv", metric_fields, metric_rows)

    primary_sorted = sorted(primary_rows, key=lambda row: mini.primary_ranks[row["Grant ID"]])
    ranking_fields = [
        "Grant ID", "Program", "Type", "Core Requirements", "Thematic fit",
        "Eligibility label", "Availability label", "minilm_score", "minilm_rank",
        "bge_score", "bge_rank",
    ]
    primary_output = []
    for row in primary_sorted:
        item = dict(row)
        grant_id = row["Grant ID"]
        item.update(
            minilm_score=_fmt(mini.test_scores[grant_id], 9), minilm_rank=mini.primary_ranks[grant_id],
            bge_score=_fmt(bge.test_scores[grant_id], 9), bge_rank=bge.primary_ranks[grant_id],
        )
        primary_output.append(item)
    _write_csv(output_dir / "synthetic_test_rankings.csv", ranking_fields, primary_output)

    unfiltered_sorted = sorted(test_rows, key=lambda row: mini.unfiltered_ranks[row["Grant ID"]])
    unfiltered_fields = [
        "Grant ID", "Program", "Type", "Core Requirements", "Thematic fit",
        "Eligibility label", "Availability label", "Hard negative?", "minilm_score",
        "minilm_rank", "bge_score", "bge_rank",
    ]
    unfiltered_output = []
    for row in unfiltered_sorted:
        item = dict(row)
        grant_id = row["Grant ID"]
        item.update(
            minilm_score=_fmt(mini.test_scores[grant_id], 9), minilm_rank=mini.unfiltered_ranks[grant_id],
            bge_score=_fmt(bge.test_scores[grant_id], 9), bge_rank=bge.unfiltered_ranks[grant_id],
        )
        unfiltered_output.append(item)
    _write_csv(output_dir / "synthetic_test_unfiltered_rankings.csv", unfiltered_fields, unfiltered_output)

    original_sorted = sorted(original_rows, key=lambda row: mini.original_ranks[row["Original ID"]])
    original_fields = [
        "Original ID", "Program", "permitted_source_text", "minilm_score", "minilm_rank",
        "bge_score", "bge_rank", "Fit (unverified reference only)",
        "Status (unverified reference only)",
    ]
    original_output = []
    for row in original_sorted:
        item_id = row["Original ID"]
        original_output.append(
            {
                "Original ID": item_id,
                "Program": row["Program"],
                "permitted_source_text": build_model_text(row),
                "minilm_score": _fmt(mini.original_scores[item_id], 9),
                "minilm_rank": mini.original_ranks[item_id],
                "bge_score": _fmt(bge.original_scores[item_id], 9),
                "bge_rank": bge.original_ranks[item_id],
                "Fit (unverified reference only)": row["Fit"],
                "Status (unverified reference only)": row["Current Status (as last reported)"],
            }
        )
    _write_csv(output_dir / "original_grant_rankings.csv", original_fields, original_output)

    report = _build_report(
        synthetic_rows, test_rows, primary_rows, original_rows, runs, mission,
        bge_instruction_enabled, source_paths, device, low_disk_mode,
    )
    (output_dir / "model_comparison_report.md").write_text(report, encoding="utf-8")


def _top_ids(rank_map: Mapping[str, int], limit: int = 5) -> list[str]:
    return [item_id for item_id, _ in sorted(rank_map.items(), key=lambda item: item[1])[:limit]]


def _build_report(
    synthetic_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
    primary_rows: Sequence[Mapping[str, str]],
    original_rows: Sequence[Mapping[str, str]],
    runs: Sequence[ModelRun],
    mission: str,
    bge_instruction_enabled: bool,
    source_paths: Mapping[str, Path],
    device: str,
    low_disk_mode: bool,
) -> str:
    by_key = {run.spec.key: run for run in runs}
    mini, bge = by_key["minilm"], by_key["bge"]
    test_by_id = {row["Grant ID"]: row for row in test_rows}
    original_by_id = {row["Original ID"]: row for row in original_rows}
    delta = bge.ndcg_at_5 - mini.ndcg_at_5
    if delta > 1e-12:
        conclusion = f"BGE ranked better on the primary synthetic metric by {delta:.6f} NDCG@5."
    elif delta < -1e-12:
        conclusion = f"MiniLM ranked better on the primary synthetic metric by {-delta:.6f} NDCG@5."
    else:
        conclusion = "The models tied on primary synthetic NDCG@5; the result is inconclusive."

    lines = [
        "# MiniLM versus BGE grant-relevance experiment",
        "",
        "## Status and conclusion",
        "",
        f"**COMPLETED with real local inference.** {conclusion}",
        "This synthetic-only comparison does not establish real-world grant recommendation performance.",
        "",
        "## Data and assumptions",
        "",
        f"- Synthetic rows: {len(synthetic_rows)} (160 predefined Train, {len(test_rows)} predefined Test).",
        "- Eligibility counts: 120 Eligible, 60 Ineligible, and 20 Needs review; validation found no duplicate IDs or missing required values.",
        f"- Primary candidate pool: {len(primary_rows)} held-out grants after an **oracle eligibility/availability filter** requiring ground-truth `Eligible` and `Open`.",
        f"- Original demonstration rows: {len(original_rows)}; their `Fit` and status values are unverified, mixed/conditional, and not accuracy ground truth.",
        f"- Hypothetical evaluation query: “{mission}” This is a fictional assessment profile, not a verified statement about Ardook.",
        "- Model input is restricted to `Program`, `Type`, and `Core Requirements`, joined with field labels. No outcome, rationale, status, split, source, or human-conclusion field is embedded.",
        "- The allowlist prevents label leakage but also omits potentially useful neutral purpose detail; rankings may therefore reflect sparse or eligibility-oriented descriptions.",
        f"- BGE query-only retrieval instruction used: **{bge_instruction_enabled}**. It was not added to documents or MiniLM.",
        "- Ground truth was joined by `Grant ID`, with unique-ID, schema, missing-field, expected-count, and shared-column agreement checks.",
        "",
        "## Primary held-out metrics",
        "",
        "Excellent=3, Strong=2, Moderate=1, Weak=0. Precision@5 treats Excellent/Strong as relevant. Raw cosine values are not compared across model families.",
        "",
        "| Model | Parameters | Test rows | Eligible/open | NDCG@5 | Precision@5 | Spearman | Inference seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        params = str(run.parameter_count) if run.parameter_count is not None else "unknown"
        lines.append(
            f"| `{run.spec.identifier}` | {params} | {len(test_rows)} | {len(primary_rows)} | "
            f"{run.ndcg_at_5:.6f} | {run.precision_at_5:.6f} | {_fmt(run.spearman)} | {run.inference_seconds:.6f} |"
        )

    lines.extend(["", "### Primary top five", ""])
    for run in runs:
        lines.append(f"**{run.spec.key}:**")
        lines.append("")
        for item_id in _top_ids(run.primary_ranks):
            row = test_by_id[item_id]
            lines.append(
                f"- {item_id} — {row['Program']} — {row['Thematic fit']} "
                f"(score {run.test_scores[item_id]:.6f})"
            )
        lines.append("")

    lines.extend([
        "## Unfiltered diagnostic (not eligibility classification)",
        "",
        "All 40 test rows were ranked without the oracle filter. Counts below show constraint violations in each top five; they are not classifier accuracy.",
        "",
        "| Model | Top-5 Ineligible | Top-5 non-open |",
        "|---|---:|---:|",
    ])
    for run in runs:
        lines.append(
            f"| `{run.spec.identifier}` | {run.unfiltered_top5_ineligible} | {run.unfiltered_top5_non_open} |"
        )
    lines.append("")
    for run in runs:
        lines.append(f"**{run.spec.key} unfiltered top five:**")
        lines.append("")
        for item_id in _top_ids(run.unfiltered_ranks):
            row = test_by_id[item_id]
            lines.append(
                f"- {item_id} — {row['Program']} — eligibility={row['Eligibility label']}, "
                f"availability={row['Availability label']}, fit={row['Thematic fit']}"
            )
        lines.append("")

    lines.extend(["## Real-data demonstration (no accuracy claim)", ""])
    for run in runs:
        lines.append(f"**{run.spec.key} top five:**")
        lines.append("")
        for item_id in _top_ids(run.original_ranks):
            row = original_by_id[item_id]
            lines.append(
                f"- {item_id} — {row['Program']} — unverified tracker fit: {row['Fit']}"
            )
        lines.append("")

    disagreements = sorted(
        test_rows,
        key=lambda row: (
            -abs(mini.unfiltered_ranks[row["Grant ID"]] - bge.unfiltered_ranks[row["Grant ID"]]),
            row["Grant ID"],
        ),
    )[:5]
    lines.extend([
        "## Error and disagreement analysis",
        "",
        "Largest test-set rank disagreements (unfiltered pool):",
        "",
    ])
    meaningful = 0
    for row in disagreements:
        item_id = row["Grant ID"]
        rank_delta = abs(mini.unfiltered_ranks[item_id] - bge.unfiltered_ranks[item_id])
        if rank_delta:
            meaningful += 1
        lines.append(
            f"- {item_id} — {row['Program']}: MiniLM #{mini.unfiltered_ranks[item_id]}, "
            f"BGE #{bge.unfiltered_ranks[item_id]} (|Δ|={rank_delta}); fit={row['Thematic fit']}, "
            f"eligibility={row['Eligibility label']}."
        )
    if meaningful < 5:
        lines.append(f"- Only {meaningful} of these had a nonzero rank disagreement.")

    high_relevance_ineligible = [
        row for row in test_rows
        if row["Eligibility label"] == "Ineligible" and fit_to_grade(row["Thematic fit"]) >= 2
    ]
    lines.extend(["", "High-relevance but ineligible examples:", ""])
    for row in high_relevance_ineligible[:5]:
        item_id = row["Grant ID"]
        lines.append(
            f"- {item_id} — {row['Program']}: {row['Thematic fit']}; MiniLM #{mini.unfiltered_ranks[item_id]}, "
            f"BGE #{bge.unfiltered_ranks[item_id]}. This illustrates why semantic relevance cannot enforce eligibility."
        )
    lines.extend([
        "",
        "Because only three neutral source fields are permitted, some disagreements may arise from short or requirement-heavy descriptions rather than genuine mission mismatch. This is a qualitative hypothesis, not a measured causal explanation. Human review and richer, non-leaking purpose text are needed.",
        "",
        "## Reproducibility",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        f"- Device: `{device}`",
        f"- Ephemeral low-disk model cache: `{low_disk_mode}`",
        f"- sentence-transformers: `{_package_version('sentence-transformers')}`",
        f"- transformers: `{_package_version('transformers')}`",
        f"- torch: `{_package_version('torch')}`",
        f"- numpy: `{_package_version('numpy')}`",
        f"- synthetic input: `{source_paths['synthetic']}`",
        f"- ground truth: `{source_paths['labels']}`",
        f"- original input: `{source_paths['original']}`",
        "",
        "```bash",
        "python -m src.compare_grant_embeddings \\",
        "  --synthetic data/raw/Synthetic_Ardook_ML_Ready_200.csv \\",
        "  --labels data/raw/Synthetic_Ardook_Ground_Truth_200.csv \\",
        "  --original data/raw/Original_Ardook_Grant_Tracker_37.csv \\",
        "  --mission config/hypothetical_mission.txt \\",
        "  --output-dir results/" + (" \\" if low_disk_mode else ""),
        *( ["  --low-disk-mode"] if low_disk_mode else [] ),
        "```",
        "",
        "Model download/loading time is excluded from reported inference seconds. The timed section includes normalized query/document encoding and deterministic ranking for the 40 synthetic test and 37 original records.",
        "",
        "## Limitations",
        "",
        "Synthetic labels encode a fictional profile and scenario design; they may reward dataset artifacts and do not prove benefit, fairness, current availability, or legal eligibility in practice. The oracle filter uses truth labels only to isolate thematic ranking and is not deployable as-is. The original tracker is small, possibly stale, and lacks independently verified labels. A human must verify official sources, deadlines, restrictions, and organizational facts before acting.",
    ])
    return "\n".join(lines) + "\n"


def write_not_run_report(output_dir: Path, error: Exception) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_comparison_report.md"
    text = (
        "# MiniLM versus BGE grant-relevance experiment\n\n"
        "## Status\n\n"
        "**NOT RUN.** Model inference did not complete, so no performance metrics were generated. "
        "Do not interpret any pre-existing result files as output from this attempt.\n\n"
        f"Failure: `{type(error).__name__}: {error}`\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def run_experiment(
    synthetic_path: Path,
    labels_path: Path,
    original_path: Path,
    mission_path: Path,
    output_dir: Path,
    device: str = "cpu",
    bge_query_instruction: bool = True,
    model_factory: Callable[[str, str], Any] = default_model_factory,
    low_disk_mode: bool = False,
) -> list[ModelRun]:
    synthetic_rows, synthetic_headers = read_csv_records(synthetic_path)
    label_rows, label_headers = read_csv_records(labels_path)
    joined_rows = validate_and_join_synthetic(
        synthetic_rows, synthetic_headers, label_rows, label_headers
    )
    original_raw, original_headers = read_csv_records(original_path)
    original_rows = validate_original(original_raw, original_headers)
    if not mission_path.is_file():
        raise DataValidationError(f"Mission file does not exist: {mission_path}")
    mission = mission_path.read_text(encoding="utf-8").strip()
    if not mission:
        raise DataValidationError("Mission query is blank")

    test_rows = sorted(
        (row for row in joined_rows if row["Train/Test split"] == "Test"),
        key=lambda row: row["Grant ID"],
    )
    primary_rows = [
        row for row in test_rows
        if row["Eligibility label"] == "Eligible" and row["Availability label"] == "Open"
    ]
    if len(primary_rows) < 5:
        raise DataValidationError(
            f"Only {len(primary_rows)} eligible/open test candidates; fixed @5 metrics unavailable"
        )
    LOGGER.info(
        "Validated 200 synthetic rows, %d held-out rows, %d oracle-filtered candidates, and 37 original rows",
        len(test_rows), len(primary_rows),
    )

    bge_prefix = "Represent this sentence for searching relevant passages: " if bge_query_instruction else ""
    specs = [
        ModelSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
        ModelSpec("bge", "BAAI/bge-small-en-v1.5", bge_prefix),
    ]
    runs: list[ModelRun] = []
    for spec in specs:
        if low_disk_mode:
            if model_factory is not default_model_factory:
                raise ValueError("low_disk_mode is supported only with the default model factory")
            # Each directory is created by this process and removed after _run_model
            # has released the loaded model, so only one downloaded model occupies disk.
            with tempfile.TemporaryDirectory(prefix=f"ardook-{spec.key}-") as cache_name:
                cache_path = Path(cache_name)

                def ephemeral_factory(identifier: str, selected_device: str) -> Any:
                    return default_model_factory(identifier, selected_device, cache_path)

                runs.append(
                    _run_model(
                        spec, mission, test_rows, primary_rows, original_rows,
                        ephemeral_factory, device,
                    )
                )
        else:
            runs.append(
                _run_model(
                    spec, mission, test_rows, primary_rows, original_rows,
                    model_factory, device,
                )
            )
    candidate_sets = [set(run.primary_ranks) for run in runs]
    if len(candidate_sets) != 2 or candidate_sets[0] != candidate_sets[1]:
        raise AssertionError("Models did not rank identical primary candidate IDs")
    _write_outputs(
        output_dir, joined_rows, test_rows, primary_rows, original_rows, runs, mission,
        bge_query_instruction,
        {"synthetic": synthetic_path, "labels": labels_path, "original": original_path},
        device, low_disk_mode,
    )
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cpu", help="sentence-transformers device (default: cpu)")
    parser.add_argument(
        "--low-disk-mode",
        action="store_true",
        help="Delete each task-created model cache after its run so only one model is stored at a time",
    )
    parser.add_argument(
        "--bge-query-instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use BGE's retrieval instruction on the query only (default: enabled)",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        runs = run_experiment(
            args.synthetic, args.labels, args.original, args.mission, args.output_dir,
            args.device, args.bge_query_instruction, low_disk_mode=args.low_disk_mode,
        )
    except (DataValidationError, ModelInferenceError, OSError) as exc:
        LOGGER.error("Experiment failed: %s", exc)
        if isinstance(exc, ModelInferenceError):
            report = write_not_run_report(args.output_dir, exc)
            LOGGER.error("Wrote transparent NOT RUN report to %s", report)
        return 2
    for run in runs:
        LOGGER.info(
            "%s: NDCG@5=%.6f Precision@5=%.6f Spearman=%s inference=%.3fs",
            run.spec.key, run.ndcg_at_5, run.precision_at_5, _fmt(run.spearman),
            run.inference_seconds,
        )
    LOGGER.info("Results written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

