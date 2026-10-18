# -*- coding: utf-8 -*-
"""Evaluate Lab 03 — Fluid Level Detection.

Run from the repo root:
    python Assignments/lab03/evaluate_lab03.py

Runs each step of FluidLevelDetector in turn and adds a figure for every step
that works. If a step isn't implemented yet, it stops there and still saves
the figures made so far — so you can run this after finishing each step.
Results are written to a single HTML file next to this script.
"""
import os
import sys
import cv2
import numpy as np
import plotly.graph_objects as go

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
sys.path.insert(0, os.path.join(LAB_DIR, ".."))

from lab03 import FluidLevelDetector, BOTTLE_IMAGE_URL
from helpers.dataloader import DATA_DIR, download_image
from helpers.plotting import save_html, png_source

HOVER = "x: %{x}<br>y: %{y}<extra></extra>"

def plot_image(img, title, roi=None):
    """An image panel (colour, or black-and-white threshold / edge images),
    optionally with the region of interest outlined in red.

    Images are embedded as compressed PNGs (png_source) to keep the report small.
    """
    fig = go.Figure(go.Image(source=png_source(img), hovertemplate=HOVER))
    if roi is not None:
        pts = np.vstack([roi[0], roi[0][:1]])   # close the polygon
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines",
                                 line=dict(color="red", width=2), name="Region of interest"))
    fig.update_layout(title_text=title, height=600)
    return fig

def draw_lines(image_rgb, lines, colour, thickness=2):
    """Return a copy of image_rgb with each (x1, y1, x2, y2) line drawn on it."""
    out = image_rgb.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), colour, thickness)
    return out

def draw_levels(image_rgb, fluid_levels):
    """Return a copy of image_rgb with a red marker at each bottle's fluid level."""
    out = image_rgb.copy()
    width = out.shape[1]
    seg_w = width // 8
    for i, y in enumerate(fluid_levels):
        if y == float('inf'):
            continue
        x_c = int(width * (2 * i + 1) / 6)   # centre of the left / middle / right third
        cv2.line(out, (x_c - seg_w, int(y)), (x_c + seg_w, int(y)), (255, 0, 0), 3)
        cv2.circle(out, (x_c, int(y)), radius=6, color=(255, 0, 0), thickness=-1)
    return out

def evaluate():
    image_rgb = download_image(BOTTLE_IMAGE_URL, save_path=DATA_DIR)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    width = gray.shape[1]

    detector = FluidLevelDetector()
    figs = [plot_image(image_rgb, "Input image — three bottles")]
    step = ""

    try:
        step = "Step 1 (threshold)"
        binary = detector.threshold(gray)
        figs.append(plot_image(binary, "Step 1: Thresholded — bottles become white shapes"))
        print(f"{step} done.")

        step = "Step 2 (close_gaps)"
        closed = detector.close_gaps(binary)
        figs.append(plot_image(closed, "Step 2: After morphological closing — small holes filled"))
        print(f"{step} done.")

        step = "Step 3 (detect_edges)"
        edges = detector.detect_edges(closed)
        figs.append(plot_image(edges, "Step 3: Canny edges"))
        print(f"{step} done.")

        step = "Step 4 (mask_roi)"
        masked = detector.mask_roi(edges)
        figs.append(plot_image(masked, "Step 4: Edges inside the region of interest (red outline)",
                                roi=detector.roi))
        print(f"{step} done.")

        step = "Step 5 (find_lines)"
        lines = detector.find_lines(masked)
        figs.append(plot_image(draw_lines(image_rgb, lines, (0, 0, 255)),
                             f"Step 5: All {len(lines)} Hough line segments (blue)"))
        print(f"{step} done — {len(lines)} line segments found.")

        step = "Step 6 (select_horizontal_lines)"
        horizontal = detector.select_horizontal_lines(lines)
        figs.append(plot_image(draw_lines(image_rgb, horizontal, (0, 255, 0)),
                             f"Step 6: {len(horizontal)} near-horizontal lines (green)"))
        print(f"{step} done — {len(horizontal)} horizontal lines kept.")

        step = "Step 7 (calculate_fluid_levels)"
        levels = detector.calculate_fluid_levels(horizontal, width)
        overlay = draw_levels(draw_lines(image_rgb, horizontal, (0, 255, 0), 1), levels)
        figs.append(plot_image(overlay, "Step 7: Fluid level per bottle (red)"))
        print(f"{step} done — levels [left, middle, right] (y px): {levels}")

        step = "Step 8 (detect_fluid_levels)"
        y_left, y_mid, y_right = detector.detect_fluid_levels(gray)
        print(f"{step} done — levels: [{y_left:.1f}, {y_mid:.1f}, {y_right:.1f}]")
        print(f"  Left vs right difference: {abs(y_left - y_right):.1f} px "
              f"(should be small — same fill level)")
        print(f"  Middle is {min(y_left, y_right) - y_mid:.1f} px higher than the others "
              f"(should be large — it is overfilled)")

    except NotImplementedError:
        print(f"\n{step} is not implemented yet — stopping here.")
        print("Figures for the earlier steps are still saved below.")

    save_html(*figs, output_path=os.path.join(LAB_DIR, "lab03_results.html"))

if __name__ == "__main__":
    evaluate()
