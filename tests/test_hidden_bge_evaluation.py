from pathlib import Path

from src.evaluate_hidden_bge import error_type, read_prediction_details


def test_error_types():
    assert error_type("Good", "Good") == "True positive"
    assert error_type("Bad", "Bad") == "True negative"
    assert error_type("Bad", "Good") == "False positive"
    assert error_type("Good", "Bad") == "False negative"


def test_prediction_details_are_joinable_by_id(tmp_path: Path):
    path = tmp_path / "predictions.csv"
    rows = [
        "grant_id,program,bge_score,threshold,binary_class,label",
    ]
    for grant_id in range(10, 0, -1):
        rows.append(
            f"{grant_id},Program {grant_id},0.5,0.4,Good,Relevant"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    parsed = read_prediction_details(path)
    assert [row["grant_id"] for row in parsed] == [str(number) for number in range(1, 11)]
