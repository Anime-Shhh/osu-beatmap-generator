#!/usr/bin/env python
"""Render docs/v3/architecture.png (the README's pipeline figure).

    python docs/v3/make_architecture_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).with_name("architecture.png")
plt.rcParams.update({"font.family": "DejaVu Sans"})

FC = {"io": "#f3f4f6", "rule": "#ffffff", "T": "#dbeafe", "R": "#fef3c7", "G": "#d1fae5", "sel": "#ede9fe"}
EC = {"io": "#6b7280", "rule": "#374151", "T": "#1d4ed8", "R": "#b45309", "G": "#047857", "sel": "#6d28d9"}
STEPS = [
    ("io", "Audio  →  130-d features @ 5 ms  (128 log-mel · spectral flux · RMS)"),
    ("T", "T · timing  (16M)   conv front end → 20 ms transformer 8×384 (RoPE) → 5 ms\nbeat / downbeat activations + 80-bin log-tempo class"),
    ("rule", "Tempo fit  (deterministic)   peak picking → phase-coherence period refinement\n→ weighted least-squares grid → segmentation only on windowed misfit  →  red lines"),
    ("io", "Beat grid   one token per beat · 24 slots per beat (1/16 ∪ 1/12)"),
    ("R", "R · rhythm  (51M)   masked discrete diffusion (MDLM), DiT 12×512, AdaLN-zero\nper beat: onset pattern over 24 slots (human pattern vocabulary)\n→ per onset: kind · duration · spans · new combo · stack;  per beat: SV class"),
    ("rule", "Legalizer  (deterministic)   tokens → timed objects, hold collisions resolved, SV sections"),
    ("G", "G · placement  (49M)   rectified flow, DiT 12×512, self-conditioning\nper object: head (x, y) + 32 unit headings per slider\n(arc length fixed by SV × duration)"),
    ("rule", "Export + checks   native curve fit (linear → arc → Bézier) · .osu v14 · footprint / overlap\nvalidity · rosu-pp star rating"),
    ("sel", "Selection   4 rhythm × 4 placement candidates → valid map closest to the target stars"),
]


def box(ax, x, y, w, h, text, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=FC[kind], ec=EC[kind], lw=1.8 if kind in "TRG" else 1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, linespacing=1.4,
            fontweight="bold" if kind in "TRG" else "normal", color="#111827")


def main():
    fig, ax = plt.subplots(figsize=(11, 10.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 104)
    ax.axis("off")
    X, W = 4, 74
    y, spans = 100.0, []
    for kind, text in STEPS:
        h = 5.4 + 2.6 * text.count("\n")
        box(ax, X, y - h, W, h, text, kind)
        spans.append((y - h, h))
        y -= h + 3.2
    for (y0, _), (y1, h1) in zip(spans, spans[1:]):
        ax.annotate("", xy=(X + W / 2, y1 + h1 + 0.25), xytext=(X + W / 2, y0 - 0.25),
                    arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.4))
    # conditioning rail: stars / AR / OD / CS / HP / era (+ classifier-free guidance) into R, G, selection
    cx = X + W + 9
    cy = spans[3][0]
    ax.add_patch(FancyBboxPatch((X + W + 3, cy - 1.5), 17, spans[3][1] + 3, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc="#fff7ed", ec="#c2410c", lw=1.1))
    ax.text(X + W + 11.5, cy + spans[3][1] / 2, "conditioning\nstars · AR · OD\nCS · HP · era\n(CFG 1.5)",
            ha="center", va="center", fontsize=8.8, linespacing=1.35)
    rail = X + W + 11.5
    ax.plot([rail, rail], [cy - 1.8, spans[8][0] + spans[8][1] / 2], color="#c2410c", lw=1.4)
    for i in (4, 6, 8):
        yy = spans[i][0] + spans[i][1] / 2
        ax.annotate("", xy=(X + W + 0.35, yy), xytext=(rail, yy),
                    arrowprops=dict(arrowstyle="-|>", color="#c2410c", lw=1.4))
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
