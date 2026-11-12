import csv
import math
from pathlib import Path

import pytest

from src.compare_grant_embeddings import (
    ALLOWED_MODEL_FIELDS,
    DataValidationError,
    FORBIDDEN_MODEL_FIELDS,
    ModelInferenceError,
    build_model_text,
    fit_to_grade,
    ndcg_at_k,
    precision_at_k,
    rank_scores,
    validate_and_join_synthetic,
    write_not_run_report,
)


def _synthetic_row(grant_id: str, split: str = "Train", eligibility: str = "Eligible"):
    return {
        "Grant ID": grant_id,
        "Program": f"Program {grant_id}",
        "Type": "Grant",
        "Core Requirements": "Build responsible community technology",
        "Eligibility label": eligibility,
        "Availability label": "Open",
        "Thematic fit": "Excellent",
        "Recommendation label": "Prioritize",
        "Scenario family": "test",
        "Eligibility rationale": "private rationale",
        "Hard negative?": "No",
        "Train/Test split": split,
    }


def _valid_200_rows():
    rows = []
    for number in range(1, 201):
        split = "Train" if number <= 160 else "Test"
        if number <= 120:
            eligibility = "Eligible"
        elif number <= 180:
            eligibility = "Ineligible"
        else:
            eligibility = "Needs review"
        rows.append(_synthetic_row(f"SYN-{number:04d}", split, eligibility))
    return rows


def test_schema_validation_rejects_missing_column():
    rows = _valid_200_rows()
    headers = list(rows[0])
    headers.remove("Core Requirements")
    with pytest.raises(DataValidationError, match="Core Requirements"):
        validate_and_join_synthetic(rows, headers, rows, list(rows[0]))


def test_ids_join_by_key_and_reject_duplicate_or_disagreement():
    rows = _valid_200_rows()
    labels = [dict(row) for row in reversed(rows)]
    joined = validate_and_join_synthetic(rows, list(rows[0]), labels, list(labels[0]))
    assert joined[0]["Grant ID"] == "SYN-0001"

    duplicate = [dict(row) for row in labels]
    duplicate[-1]["Grant ID"] = duplicate[0]["Grant ID"]
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_and_join_synthetic(rows, list(rows[0]), duplicate, list(duplicate[0]))

    mismatch = [dict(row) for row in labels]
    mismatch[0]["Thematic fit"] = "Weak"
    with pytest.raises(DataValidationError, match="disagree"):
        validate_and_join_synthetic(rows, list(rows[0]), mismatch, list(mismatch[0]))


def test_forbidden_fields_never_enter_model_text():
    row = {field: f"allowed-{field}" for field in ALLOWED_MODEL_FIELDS}
    for field in FORBIDDEN_MODEL_FIELDS:
        row[field] = f"SECRET-{field}"
    text = build_model_text(row)
    assert all(f"allowed-{field}" in text for field in ALLOWED_MODEL_FIELDS)
    assert "SECRET-" not in text
    assert all(field not in text for field in FORBIDDEN_MODEL_FIELDS)


@pytest.mark.parametrize(
    ("label", "grade"),
    [("Excellent", 3), ("Strong", 2), ("Moderate", 1), ("Weak", 0)],
)
def test_fit_to_grade(label, grade):
    assert fit_to_grade(label) == grade


def test_metrics_on_hand_constructed_example():
    relevances = [3, 0, 2, 1, 0]
    expected_dcg = 3 + 2 / math.log2(4) + 1 / math.log2(5)
    expected_idcg = 3 + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(relevances, 5) == pytest.approx(expected_dcg / expected_idcg)
    assert precision_at_k(relevances, 5) == pytest.approx(0.4)
    assert ndcg_at_k([3, 2, 1, 0], 5) is None
    assert precision_at_k([3, 2, 1, 0], 5) is None


def test_score_ranking_is_descending_and_deterministic_on_ties():
    order, ranks = rank_scores(["b", "a", "c"], [0.5, 0.5, -0.1])
    assert order == ["a", "b", "c"]
    assert ranks == {"a": 1, "b": 2, "c": 3}
    with pytest.raises(ModelInferenceError, match="Non-finite"):
        rank_scores(["x"], [float("nan")])


def test_same_candidate_ids_can_be_asserted_for_both_models():
    candidate_ids = ["SYN-1", "SYN-2", "SYN-3"]
    _, first = rank_scores(candidate_ids, [0.2, 0.7, 0.1])
    _, second = rank_scores(candidate_ids, [0.9, 0.1, 0.3])
    assert set(first) == set(second) == set(candidate_ids)


def test_no_test_labels_used_as_input():
    row_a = _synthetic_row("SYN-1", "Test", "Eligible")
    row_b = dict(row_a)
    row_b.update(
        {
            "Eligibility label": "Ineligible",
            "Availability label": "Closed",
            "Thematic fit": "Weak",
            "Recommendation label": "Exclude",
            "Train/Test split": "Train",
        }
    )
    assert build_model_text(row_a) == build_model_text(row_b)


def test_inference_failure_writes_not_run_without_metrics(tmp_path: Path):
    report = write_not_run_report(tmp_path, ModelInferenceError("download blocked"))
    assert "NOT RUN" in report.read_text(encoding="utf-8")
    assert "no performance metrics were generated" in report.read_text(encoding="utf-8")
    assert not (tmp_path / "model_comparison_metrics.csv").exists()

