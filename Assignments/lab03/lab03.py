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
        _, binary = cv2.threshold(
            image_gray,
            self.threshold_value,
            255,
            cv2.THRESH_BINARY_INV
        )
        return binary

    def close_gaps(self, binary):
        kernel = np.ones(
            (self.kernel_size, self.kernel_size),
            dtype=np.uint8
        )
        return cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel
        )

    def detect_edges(self, binary):
        return cv2.Canny(
            binary,
            self.canny_low,
            self.canny_high
        )

    def mask_roi(self, edges):
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.roi, 255)
        return cv2.bitwise_and(edges, edges, mask=mask)

    def find_lines(self, masked_edges):
        lines = cv2.HoughLinesP(
            masked_edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            return np.empty((0, 4), dtype=np.int32)

        return lines.reshape(-1, 4)

    def select_horizontal_lines(self, lines):
        horizontal = []

        for x1, y1, x2, y2 in lines:
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) <= self.slope_threshold:
                horizontal.append((x1, y1, x2, y2))

        return horizontal

    def calculate_fluid_levels(self, horizontal_lines, image_width):
        groups = [[], [], []]

        for x1, y1, x2, y2 in horizontal_lines:
            x_centre = (x1 + x2) / 2
            y_centre = (y1 + y2) / 2

            column = min(
                int(x_centre // (image_width / 3)), 2
            )
            groups[column].append(y_centre)

        return [
            float(np.median(group)) if group else float('inf')
            for group in groups
        ]

    def detect_fluid_levels(self, image_gray):
        self.thresholded = self.threshold(image_gray)
        self.closed = self.close_gaps(self.thresholded)
        self.edges = self.detect_edges(self.closed)
        self.masked_edges = self.mask_roi(self.edges)
        self.lines = self.find_lines(self.masked_edges)
        self.horizontal_lines = self.select_horizontal_lines(
            self.lines
        )
        self.fluid_levels = self.calculate_fluid_levels(
            self.horizontal_lines, image_gray.shape[1]
        )

        return self.fluid_levels


    