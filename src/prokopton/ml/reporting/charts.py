"""Small, self-contained SVG charts for model cards.

The charts are static files (they are shown as images in Markdown), so each
mark carries a ``<title>`` for a native tooltip when the SVG is opened
directly, and every chart is accompanied by a table in the card. Colours
follow a colour-blind-checked palette, with light and dark variants chosen
by the viewer's colour scheme. Text uses text colours, never series colours.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

__all__ = ["SERIES_COLOURS", "Series", "line_chart", "reliability_chart"]

SERIES_COLOURS = (("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"))
"""(light, dark) colours for the first two series."""

_WIDTH = 720
_HEIGHT = 360
_LEFT = 64
_RIGHT = 176
_TOP = 56
_BOTTOM = 52


@dataclass(frozen=True, slots=True)
class Series:
    """One named line: a y value (0-1) for each x position."""

    name: str
    values: Sequence[float]


def _line(css: str, x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<line class="{css}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'
    )


def _text(
    x: float, y: float, content: str, anchor: str = "start", extra: str = ""
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{extra}>'
        f"{escape(content)}</text>"
    )


def _dot(index: int, x: float, y: float, tip: str) -> str:
    return (
        f'<circle class="dot f{index}" cx="{x:.1f}" cy="{y:.1f}" r="4">'
        f"<title>{escape(tip)}</title></circle>"
    )


def _polyline(index: int, points: Sequence[tuple[float, float]]) -> str:
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline class="line s{index}" points="{path}"/>'


_PLOT_RIGHT = _WIDTH - _RIGHT
_PLOT_BOTTOM = _HEIGHT - _BOTTOM
_MIDDLE_X = (_LEFT + _PLOT_RIGHT) / 2
_MIDDLE_Y = (_TOP + _PLOT_BOTTOM) / 2


def _style(series_count: int) -> str:
    light = "".join(f"--s{i}:{SERIES_COLOURS[i][0]};" for i in range(series_count))
    dark = "".join(f"--s{i}:{SERIES_COLOURS[i][1]};" for i in range(series_count))
    marks = "".join(
        f".s{i}{{stroke:var(--s{i});}}.f{i}{{fill:var(--s{i});}}"
        for i in range(series_count)
    )
    return (
        "<style>"
        f"svg{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;{light}}}"
        "@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#f0efec;"
        f"--ink2:#c3c2b7;--grid:#353532;{dark}}}}}"
        "text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;"
        "fill:var(--ink2);font-size:12px}"
        ".title{fill:var(--ink);font-size:15px;font-weight:600}"
        ".grid{stroke:var(--grid);stroke-width:1}"
        ".line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
        ".dot{stroke:var(--surface);stroke-width:2}"
        ".ref{stroke:var(--ink2);stroke-width:1}"
        f"{marks}"
        "</style>"
    )


def _frame(title: str, body: str, series_count: int) -> str:
    heading = _text(_LEFT, 28, title, extra=' class="title"')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} {_HEIGHT}" '
        f'width="{_WIDTH}" height="{_HEIGHT}" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title>{_style(series_count)}"
        f'<rect width="{_WIDTH}" height="{_HEIGHT}" fill="var(--surface)"/>'
        f"{heading}"
        f"{body}</svg>\n"
    )


def _y(value: float) -> float:
    return _TOP + (1.0 - value) * (_PLOT_BOTTOM - _TOP)


def _x(value: float, low: float, high: float) -> float:
    return _LEFT + (value - low) / (high - low) * (_PLOT_RIGHT - _LEFT)


def _y_axis(label: str) -> str:
    parts = []
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = _y(tick)
        parts.append(_line("grid", _LEFT, y, _PLOT_RIGHT, y))
        parts.append(_text(_LEFT - 8, y + 4, f"{tick:.0%}", "end"))
    rotate = f' transform="rotate(-90 16 {_MIDDLE_Y:.1f})"'
    parts.append(_text(16, _MIDDLE_Y, label, "middle", rotate))
    return "".join(parts)


def _legend(names: Sequence[str]) -> str:
    parts = []
    x = float(_LEFT)
    for i, name in enumerate(names):
        parts.append(_line(f"line s{i}", x, 44, x + 16, 44))
        parts.append(_text(x + 22, 48, name))
        x += 30 + 7 * len(name)
    return "".join(parts)


def line_chart(
    title: str,
    x_values: Sequence[float],
    x_labels: Sequence[str],
    series: Sequence[Series],
    *,
    x_label: str,
    y_label: str,
    marker_x: float | None = None,
    marker_label: str = "",
) -> str:
    """Draw lines of 0-1 values against numeric x positions, as SVG text.

    ``marker_x`` draws a thin vertical reference line (e.g. a minimum length).
    """
    if len(series) > len(SERIES_COLOURS):
        msg = f"At most {len(SERIES_COLOURS)} series are supported"
        raise ValueError(msg)
    low, high = min(x_values), max(x_values)
    parts = [_y_axis(y_label), _legend([s.name for s in series])]
    for value, label in zip(x_values, x_labels, strict=True):
        parts.append(_text(_x(value, low, high), _PLOT_BOTTOM + 18, label, "middle"))
    parts.append(_text(_MIDDLE_X, _HEIGHT - 10, x_label, "middle"))
    if marker_x is not None:
        x = _x(marker_x, low, high)
        parts.append(_line("ref", x, _TOP, x, _PLOT_BOTTOM))
        parts.append(_text(x + 4, _TOP + 12, marker_label))
    ends: list[tuple[float, str]] = []
    for i, line in enumerate(series):
        points = [
            (_x(x, low, high), _y(y))
            for x, y in zip(x_values, line.values, strict=True)
        ]
        parts.append(_polyline(i, points))
        for (x, y), label, value in zip(points, x_labels, line.values, strict=True):
            parts.append(_dot(i, x, y, f"{line.name}, {label}: {value:.1%}"))
        ends.append((points[-1][1], f"{line.name} {line.values[-1]:.0%}"))
    # Direct labels at the line ends, kept at least 16px apart
    placed: list[float] = []
    for y, text in sorted(ends):
        position = max([y + 4, *[p + 16 for p in placed]])
        placed.append(position)
        parts.append(_text(_PLOT_RIGHT + 10, position, text))
    return _frame(title, "".join(parts), len(series))


def reliability_chart(
    title: str,
    confidence: Sequence[float],
    accuracy: Sequence[float],
    records: Sequence[int],
) -> str:
    """Plot accuracy against mean confidence per bin, with the ideal diagonal."""
    parts = [_y_axis("Accuracy")]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(_text(_x(tick, 0, 1), _PLOT_BOTTOM + 18, f"{tick:.0%}", "middle"))
    parts.append(_text(_MIDDLE_X, _HEIGHT - 10, "Confidence", "middle"))
    parts.append(_line("ref", _x(0, 0, 1), _y(0), _x(1, 0, 1), _y(1)))
    parts.append(_text(_PLOT_RIGHT + 10, _y(1) + 4, "Ideal"))
    points = [(_x(c, 0, 1), _y(a)) for c, a in zip(confidence, accuracy, strict=True)]
    parts.append(_polyline(0, points))
    for (x, y), c, a, n in zip(points, confidence, accuracy, records, strict=True):
        tip = f"Confidence {c:.1%}: accuracy {a:.1%} ({n:,} records)"
        parts.append(_dot(0, x, y, tip))
    return _frame(title, "".join(parts), 1)
