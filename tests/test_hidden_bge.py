from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from src.classify_hidden_bge import binary_metrics, read_hidden_xlsx, select_f1_threshold


def test_binary_metrics():
    metrics = binary_metrics(
        [True, True, False, False],
        [True, False, True, False],
    )
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert (metrics.true_positive, metrics.false_positive) == (1, 1)
    assert (metrics.true_negative, metrics.false_negative) == (1, 1)


def test_threshold_is_selected_without_hidden_labels():
    threshold, metrics = select_f1_threshold(
        [0.1, 0.2, 0.8, 0.9],
        [False, False, True, True],
    )
    assert 0.2 < threshold < 0.8
    assert metrics.f1 == pytest.approx(1.0)


def test_hidden_workbook_requires_exact_ids_1_to_10(tmp_path: Path):
    workbook = tmp_path / "hidden.xlsx"
    shared = ["Grant ID", "Program", "Type", "Core Requirements"]
    for number in range(1, 10):
        shared.extend([f"Program {number}", "Grant", "Requirements"])
    shared_xml = "".join(f"<si><t>{value}</t></si>" for value in shared)
    rows = [
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>'
        '<c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c></row>'
    ]
    string_index = 4
    for number in range(1, 10):
        rows.append(
            f'<row r="{number + 1}"><c r="A{number + 1}"><v>{number}</v></c>'
            f'<c r="B{number + 1}" t="s"><v>{string_index}</v></c>'
            f'<c r="C{number + 1}" t="s"><v>{string_index + 1}</v></c>'
            f'<c r="D{number + 1}" t="s"><v>{string_index + 2}</v></c></row>'
        )
        string_index += 3
    with ZipFile(workbook, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            + shared_xml
            + "</sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<sheetData>"
            + "".join(rows)
            + "</sheetData></worksheet>",
        )
    with pytest.raises(ValueError, match="IDs 1-10"):
        read_hidden_xlsx(workbook)
