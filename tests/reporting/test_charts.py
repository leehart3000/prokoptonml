"""Tests for prokopton.ml.reporting.charts."""

import xml.etree.ElementTree as ET

import pytest

from prokopton.ml.reporting.charts import Series, line_chart, reliability_chart

SVG = "{http://www.w3.org/2000/svg}"


def test_line_chart_is_valid_svg_with_labelled_points() -> None:
    svg = line_chart(
        "Accuracy by length",
        [50.0, 100.0, 650.0],
        ["50", "100", "Whole"],
        [Series("Known", [0.4, 0.7, 0.99]), Series("Novel", [0.35, 0.65, 0.97])],
        x_label="Length",
        y_label="Accuracy",
        marker_x=100.0,
        marker_label="Minimum (100)",
    )

    root = ET.fromstring(svg)
    assert root.get("aria-label") == "Accuracy by length"
    tips = [title.text for title in root.iter(f"{SVG}title")]
    assert "Known, 100: 70.0%" in tips
    assert "Novel, Whole: 97.0%" in tips
    texts = [text.text for text in root.iter(f"{SVG}text")]
    assert {"Known 99%", "Novel 97%", "Minimum (100)", "Whole"} <= set(texts)
    assert "prefers-color-scheme: dark" in svg


def test_line_chart_end_labels_do_not_overlap() -> None:
    svg = line_chart(
        "Close lines",
        [0.0, 1.0],
        ["a", "b"],
        [Series("One", [0.5, 0.9]), Series("Two", [0.5, 0.9])],
        x_label="x",
        y_label="y",
    )

    root = ET.fromstring(svg)
    ends = [
        float(text.get("y", "0"))
        for text in root.iter(f"{SVG}text")
        if (text.text or "").endswith("90%")
    ]
    assert abs(ends[0] - ends[1]) >= 16


def test_line_chart_supports_two_series_at_most() -> None:
    with pytest.raises(ValueError, match="At most 2"):
        line_chart(
            "Too many",
            [0.0, 1.0],
            ["a", "b"],
            [Series(name, [0.1, 0.2]) for name in "abc"],
            x_label="x",
            y_label="y",
        )


def test_reliability_chart_has_a_point_per_bin() -> None:
    svg = reliability_chart("Reliability", [0.55, 0.95], [0.5, 0.97], [120, 5000])

    root = ET.fromstring(svg)
    assert len(list(root.iter(f"{SVG}circle"))) == 2
    tips = [title.text for title in root.iter(f"{SVG}title")]
    assert "Confidence 95.0%: accuracy 97.0% (5,000 records)" in tips
