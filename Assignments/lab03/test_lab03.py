# -*- coding: utf-8 -*-
"""Tests for Lab 03 — Fluid Level Detection via the Hough Transform.

One group of tests per step, so a failing test tells you which step to fix.
Each image step is checked against the expected OpenCV result, using the
detector's own settings — so make sure your methods use self.threshold_value,
self.kernel_size, etc. rather than fixed numbers.
"""
import cv2
import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lab03 import FluidLevelDetector, BOTTLE_IMAGE_URL
from helpers.dataloader import DATA_DIR, download_image

SIMILARITY_TOLERANCE = 10   # max px diff between left and right bottles
OVERFILL_THRESHOLD   = 30   # min px diff to confirm middle bottle is higher

@pytest.fixture(scope="module")
def gray():
    image_rgb = download_image(BOTTLE_IMAGE_URL, save_path=DATA_DIR)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

@pytest.fixture(scope="module")
def detector():
    return FluidLevelDetector()

# Non-default settings, used to check each step reads its setting from self.
CUSTOM = dict(threshold_value=200, kernel_size=9)

# ── Step 1: threshold ────────────────────────────────────────────────────

def test_threshold_is_binary(detector, gray):
    binary = detector.threshold(gray)
    assert binary.shape == gray.shape and binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255}), "Thresholded image must contain only 0 and 255."

def test_threshold_matches_reference(gray):
    for settings in ({}, CUSTOM):
        det = FluidLevelDetector(**settings)
        _, expected = cv2.threshold(gray, det.threshold_value, 255, cv2.THRESH_BINARY_INV)
        assert np.array_equal(det.threshold(gray), expected), (
            f"Threshold result is wrong for threshold_value={det.threshold_value} — "
            f"check you use cv2.THRESH_BINARY_INV and self.threshold_value."
        )

# ── Step 2: close_gaps ───────────────────────────────────────────────────

def test_close_gaps_matches_reference(gray):
    for settings in ({}, CUSTOM):
        det = FluidLevelDetector(**settings)
        binary = det.threshold(gray)
        kernel = np.ones((det.kernel_size, det.kernel_size), np.uint8)
        expected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        assert np.array_equal(det.close_gaps(binary), expected), (
            f"Closing result is wrong for kernel_size={det.kernel_size} — "
            f"check you use cv2.MORPH_CLOSE and self.kernel_size."
        )

# ── Step 3: detect_edges ─────────────────────────────────────────────────

def test_detect_edges_matches_reference(gray):
    closed = FluidLevelDetector().close_gaps(FluidLevelDetector().threshold(gray))
    # Edges in a 0/255 binary image are very strong, so only high thresholds
    # change the result — that's what the second setting checks.
    references = []
    for settings in ({}, dict(canny_low=1500, canny_high=3000)):
        det = FluidLevelDetector(**settings)
        expected = cv2.Canny(closed, det.canny_low, det.canny_high)
        references.append(expected)
        assert np.array_equal(det.detect_edges(closed), expected), (
            f"Edge result is wrong for canny_low={det.canny_low}, canny_high={det.canny_high} — "
            f"use self.canny_low and self.canny_high."
        )
    assert not np.array_equal(*references)   # sanity: the two settings really differ

# ── Step 4: mask_roi ─────────────────────────────────────────────────────

def test_mask_roi_zero_outside_roi(detector):
    edges = np.full((560, 660), 255, dtype=np.uint8)   # an image that is all edges
    masked = detector.mask_roi(edges)
    assert masked.shape == edges.shape
    assert masked[:79, :].max() == 0 and masked[202:, :].max() == 0, (
        "Everything above row 80 and below row 200 must be removed."
    )
    assert masked[140, 330] == 255, "Pixels inside the region of interest must be kept."

def test_mask_roi_respects_custom_roi():
    roi = np.array([[[100, 300], [100, 250], [200, 250], [200, 300]]], dtype=np.int32)
    det = FluidLevelDetector(roi=roi)
    masked = det.mask_roi(np.full((560, 660), 255, dtype=np.uint8))
    assert masked[275, 150] == 255, "Inside a custom roi must be kept — use self.roi."
    assert masked[140, 330] == 0, "Outside a custom roi must be removed — use self.roi."

# ── Step 5: find_lines ───────────────────────────────────────────────────

def test_find_lines_format(detector, gray):
    masked = detector.mask_roi(detector.detect_edges(detector.close_gaps(detector.threshold(gray))))
    lines = np.asarray(detector.find_lines(masked))
    assert lines.ndim == 2 and lines.shape[1] == 4, (
        f"Expected an (N, 4) array of (x1, y1, x2, y2), got shape {lines.shape} — "
        f"try lines.reshape(-1, 4)."
    )
    assert len(lines) > 0, "No lines found."
    ys = lines[:, [1, 3]]
    assert ys.min() >= 79 and ys.max() <= 201, "Lines should all lie inside the region of interest."

def test_find_lines_handles_no_lines(detector):
    lines = np.asarray(detector.find_lines(np.zeros((560, 660), dtype=np.uint8)))
    assert len(lines) == 0, "With a blank image, return an empty result (not None)."

# ── Step 6: select_horizontal_lines ──────────────────────────────────────

def test_select_horizontal_keeps_and_removes(detector):
    lines = np.array([
        [0, 50, 100, 50],     # flat — keep
        [0, 50, 100, 55],     # slope 0.05 — keep (threshold 0.1)
        [0, 0, 10, 100],      # steep — remove
        [30, 0, 30, 100],     # vertical (x1 == x2) — remove, without crashing
    ])
    kept = [tuple(int(v) for v in line) for line in detector.select_horizontal_lines(lines)]
    assert kept == [(0, 50, 100, 50), (0, 50, 100, 55)]

def test_select_horizontal_respects_slope_threshold():
    lines = np.array([[0, 50, 100, 55]])   # slope 0.05
    assert len(FluidLevelDetector(slope_threshold=0.1).select_horizontal_lines(lines)) == 1
    assert len(FluidLevelDetector(slope_threshold=0.01).select_horizontal_lines(lines)) == 0, (
        "Use self.slope_threshold rather than a fixed number."
    )

# ── Step 7: calculate_fluid_levels ───────────────────────────────────────

def test_calculate_fluid_levels_groups_and_medians(detector):
    width = 600   # thirds: 0-200 left, 200-400 middle, 400-600 right
    lines = [
        (50, 100, 150, 100), (60, 110, 140, 110), (70, 130, 130, 130),   # left: median 110
        (250, 80, 350, 80),                                               # middle: 80
        (450, 120, 550, 124),                                             # right: centre y 122
    ]
    assert detector.calculate_fluid_levels(lines, width) == [110, 80, 122]

def test_calculate_fluid_levels_empty_bottle_is_inf(detector):
    levels = detector.calculate_fluid_levels([(50, 100, 150, 100)], 600)
    assert levels[0] == 100
    assert levels[1] == float('inf') and levels[2] == float('inf'), (
        "A bottle with no lines must give float('inf')."
    )

# ── Step 8: detect_fluid_levels ──────────────────────────────────────────

@pytest.fixture(scope="module")
def fluid_levels(gray):
    return FluidLevelDetector().detect_fluid_levels(gray)

def test_returns_three_levels(fluid_levels):
    assert len(fluid_levels) == 3, f"Expected 3 fluid levels, got {len(fluid_levels)}."

def test_all_bottles_detected(fluid_levels):
    for label, v in zip(("left", "middle", "right"), fluid_levels):
        assert v != float('inf'), f"Failed to detect fluid level for the {label} bottle."

def test_stores_intermediate_results(gray):
    det = FluidLevelDetector()
    det.detect_fluid_levels(gray)
    for attr in ("thresholded", "closed", "edges", "masked_edges", "lines"):
        assert getattr(det, attr) is not None, f"detect_fluid_levels should store self.{attr}."

def test_fluid_levels(fluid_levels):
    """Left ≈ right; middle is significantly higher (lower y) than both."""
    y_left, y_mid, y_right = fluid_levels
    errors = []

    diff_lr = abs(y_left - y_right)
    if diff_lr > SIMILARITY_TOLERANCE:
        errors.append(
            f"Left and right bottles should be at similar levels. "
            f"Difference was {diff_lr:.1f} px (max allowed: {SIMILARITY_TOLERANCE} px)."
        )
    # In images y=0 is the top, so higher fluid = smaller y value
    for name, y in (("left", y_left), ("right", y_right)):
        if y_mid > y - OVERFILL_THRESHOLD:
            errors.append(
                f"Middle bottle not detected as significantly higher than {name}. "
                f"y_mid={y_mid:.1f}, y_{name}={y:.1f} (requires ≥ {OVERFILL_THRESHOLD} px difference)."
            )
    assert not errors, "Errors:\n" + "\n".join(errors)
