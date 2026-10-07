"""A tiny, dependency-free SVG candlestick renderer.

The UI and the (optional) external vision providers both need a chart *image*.
Matplotlib is not available and not wanted, so this module emits a plain SVG
string directly.  It is deliberately simple and fully deterministic: the same
series always produces byte-identical output, which keeps tests and artifacts
reproducible.

SVG is text, so it renders in any browser and can be read back without an
image library.
"""

from __future__ import annotations

import os
from typing import List, Union

from ..types import MarketSeries
from ..utils.jsonio import atomic_write_text

__all__ = ["render_chart_svg"]

_UP_COLOUR = "#26a69a"
_DOWN_COLOUR = "#ef5350"
_BG_COLOUR = "#ffffff"
_AXIS_COLOUR = "#dddddd"
_TEXT_COLOUR = "#333333"


def _svg_shell(width: int, height: int, title: str, body: str) -> str:
    """Wrap rendered ``body`` markup in a complete SVG document."""
    safe_title = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    title_markup = ""
    if safe_title:
        title_markup = (
            f'<text x="8" y="20" font-family="sans-serif" font-size="14" '
            f'fill="{_TEXT_COLOUR}">{safe_title}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{_BG_COLOUR}"/>'
        f"{title_markup}{body}</svg>"
    )


def render_chart_svg(
    series: MarketSeries,
    out_path: Union[str, "os.PathLike[str]"],
    title: str = "",
    width: int = 800,
    height: int = 400,
    *,
    padding: int = 40,
) -> str:
    """Render ``series`` as a candlestick SVG and write it to ``out_path``.

    Returns the path written (as a string).  An empty series produces a valid
    placeholder SVG rather than raising, so the pipeline never crashes on a
    missing chart.
    """
    candles = list(series.candles) if series is not None else []
    width = max(1, int(width))
    height = max(1, int(height))
    padding = max(0, int(padding))
    plot_w = max(1, width - 2 * padding)
    plot_h = max(1, height - 2 * padding)

    if not candles:
        body = (
            f'<text x="{padding}" y="{padding + 20}" font-family="sans-serif" '
            f'font-size="14" fill="{_TEXT_COLOUR}">No data</text>'
        )
        atomic_write_text(out_path, _svg_shell(width, height, title, body))
        return str(out_path)

    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    hi = max(highs)
    lo = min(lows)
    span = hi - lo
    if span <= 0:
        span = abs(hi) or 1.0  # flat series: avoid divide-by-zero

    n = len(candles)
    step = plot_w / n
    body_w = max(1.0, step * 0.6)

    def x_of(index: int) -> float:
        return padding + (index + 0.5) * step

    def y_of(price: float) -> float:
        return padding + (hi - price) / span * plot_h

    parts: List[str] = []
    for i, candle in enumerate(candles):
        x = x_of(i)
        colour = _UP_COLOUR if candle.close >= candle.open else _DOWN_COLOUR
        y_high = y_of(candle.high)
        y_low = y_of(candle.low)
        y_open = y_of(candle.open)
        y_close = y_of(candle.close)
        top = min(y_open, y_close)
        body_h = max(1.0, abs(y_close - y_open))
        parts.append(
            f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" '
            f'y2="{y_low:.2f}" stroke="{colour}" stroke-width="1"/>'
        )
        parts.append(
            f'<rect x="{x - body_w / 2:.2f}" y="{top:.2f}" '
            f'width="{body_w:.2f}" height="{body_h:.2f}" fill="{colour}"/>'
        )

    # A light baseline gives the eye a reference without adding dependencies.
    baseline_y = padding + plot_h
    parts.append(
        f'<line x1="{padding}" y1="{baseline_y:.2f}" x2="{padding + plot_w:.2f}" '
        f'y2="{baseline_y:.2f}" stroke="{_AXIS_COLOUR}" stroke-width="1"/>'
    )

    atomic_write_text(out_path, _svg_shell(width, height, title, "".join(parts)))
    return str(out_path)
