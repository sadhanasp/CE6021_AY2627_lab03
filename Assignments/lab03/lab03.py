# -*- coding: utf-8 -*-
"""Lab 03 — Fluid Level Detection via the Hough Transform.

Detect the fluid level in three bottles. The work is split into small steps,
each one a method below that takes an image (or list of lines) in and returns
the result out:

  1. threshold(image_gray)                       dark bottles -> white shapes
  2. close_gaps(binary)                          fill small holes in the shapes
  3. detect_edges(binary)                        Canny edge detection
  4. mask_roi(edges)                             keep only the region of interest
  5. find_lines(masked_edges)                    Hough line segments
  6. select_horizontal_lines(lines)              keep only near-horizontal lines
  7. calculate_fluid_levels(lines, image_width)  one level per bottle
  8. detect_fluid_levels(image_gray)             put the steps together

Run evaluate_lab03.py after each step to see your progress.
"""
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import DATA_DIR, download_image  # noqa: F401

BOTTLE_IMAGE_URL = "https://cdn.eaglepi.com/uploads/2016/08/NEW_Fill_level_Rounded_Image_660pxh.jpg"

class FluidLevelDetector:
    """Detects fluid levels in a three-bottle inspection image.

    All settings have defaults, so no arguments are needed:

        detector = FluidLevelDetector()
        fluid_levels = detector.detect_fluid_levels(gray_image)

    Any setting can be changed with a keyword argument, e.g.:

        detector = FluidLevelDetector(threshold_value=230, canny_low=40)

    Use these settings (self.threshold_value, self.kernel_size, ...) inside
    your methods rather than typing the numbers in directly.
    """

    def __init__(self, threshold_value=240, **kwargs):
        self.threshold_value = threshold_value
        self.kernel_size     = kwargs.get('kernel_size',     5)
        self.canny_low       = kwargs.get('canny_low',       50)
        self.canny_high      = kwargs.get('canny_high',      150)
        self.slope_threshold = kwargs.get('slope_threshold', 0.1)
        self.hough_rho       = kwargs.get('hough_rho',       1)
        self.hough_theta     = kwargs.get('hough_theta',     np.pi / 180)
        self.hough_threshold = kwargs.get('hough_threshold', 30)
        self.min_line_len    = kwargs.get('min_line_len',    20)
        self.max_line_gap    = kwargs.get('max_line_gap',    20)
        # Region of interest: a rectangle covering the bottle necks (rows 80-200)
        self.roi             = kwargs.get('roi', np.array(
            [[[0, 200], [0, 80], [660, 80], [660, 200]]], dtype=np.int32))

        # Intermediate results, filled in by detect_fluid_levels (step 8)
        self.thresholded      = None
        self.closed           = None
        self.edges            = None
        self.masked_edges     = None
        self.lines            = None
        self.horizontal_lines = []
        self.fluid_levels     = []

    def threshold(self, image_gray):
        """Step 1 — turn the dark bottles into white shapes on a black background.

        Args:
            image_gray (np.ndarray): 2D uint8 greyscale image.

        Returns:
            np.ndarray: 2D uint8 binary image (values 0 or 255).
        """
        raise NotImplementedError("Implement this method")

    def close_gaps(self, binary):
        """Step 2 — morphological closing to fill small holes and gaps.

        Args:
            binary (np.ndarray): 2D uint8 binary image from threshold().

        Returns:
            np.ndarray: 2D uint8 binary image, same shape.
        """
        raise NotImplementedError("Implement this method")

    def detect_edges(self, binary):
        """Step 3 — Canny edge detection.

        Args:
            binary (np.ndarray): 2D uint8 binary image from close_gaps().

        Returns:
            np.ndarray: 2D uint8 edge image (edges are 255, everything else 0).
        """
        raise NotImplementedError("Implement this method")

    def mask_roi(self, edges):
        """Step 4 — keep only the edges inside the region of interest (self.roi).

        Args:
            edges (np.ndarray): 2D uint8 edge image from detect_edges().

        Returns:
            np.ndarray: 2D uint8 edge image, zero outside the region of interest.
        """
        raise NotImplementedError("Implement this method")

    def find_lines(self, masked_edges):
        """Step 5 — find straight line segments with the probabilistic Hough transform.

        Args:
            masked_edges (np.ndarray): 2D uint8 edge image from mask_roi().

        Returns:
            np.ndarray: shape (N, 4) — one row per line, (x1, y1, x2, y2).
            Shape (0, 4) if no lines are found.
        """
        raise NotImplementedError("Implement this method")

    def select_horizontal_lines(self, lines):
        """Step 6 — keep only lines that are close to horizontal.

        A line is kept when |slope| <= self.slope_threshold, where
        slope = (y2 - y1) / (x2 - x1). Vertical lines (x1 == x2) are never kept.

        Args:
            lines: (N, 4) array from find_lines(), one row (x1, y1, x2, y2) per line.

        Returns:
            list of (x1, y1, x2, y2): the lines that pass.
        """
        raise NotImplementedError("Implement this method")

    def calculate_fluid_levels(self, horizontal_lines, image_width):
        """Step 7 — one fluid level (a y value) for each bottle.

        The image is split into three equal-width columns (left, middle,
        right — one per bottle). Each line is assigned to the column its
        centre falls in, and the level for a bottle is the median of the
        centre y values of its lines.

        Args:
            horizontal_lines: list of (x1, y1, x2, y2) from select_horizontal_lines().
            image_width (int): width of the image in pixels.

        Returns:
            list: [y_left, y_mid, y_right]. float('inf') for a bottle with no lines.
        """
        raise NotImplementedError("Implement this method")

    def detect_fluid_levels(self, image_gray):
        """Step 8 — run the whole pipeline using the methods above.

        Stores each intermediate result on self (thresholded, closed, edges,
        masked_edges, lines, horizontal_lines, fluid_levels) so they can be
        displayed afterwards.

        Args:
            image_gray (np.ndarray): 2D uint8 greyscale image of the three bottles.

        Returns:
            list: [y_left, y_mid, y_right] — the fluid level in each bottle.
        """
        raise NotImplementedError("Implement this method")
