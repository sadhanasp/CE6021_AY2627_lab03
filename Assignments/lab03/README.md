# Lab 03 — Fluid Level Detection via the Hough Transform

## Introduction

The Hough transform is a classic technique for finding straight lines (and other shapes) in an image, even when they are broken up or surrounded by noise. In this lab you apply it to a manufacturing quality-control task: finding the **fluid fill level** in three bottles from a single camera image. The middle bottle has been deliberately overfilled, and your detector should spot it.

The lab is split into eight small steps. Each step is one short method — usually a single OpenCV call — and you can check each one before moving on to the next.

## Dataset / Test Images

| Resource | Details |
|----------|---------|
| Bottle inspection image | Downloaded automatically from a public URL (`BOTTLE_IMAGE_URL`) the first time you run the evaluate script or tests, then cached in `Data/`. A colour image (560 × 660 pixels) of three bottles side by side; the middle bottle is overfilled. |

## How This Lab Works

1. Implement **one step** in `lab03.py`.
2. Run `python evaluate_lab03.py`. It runs every step that is finished, adds a figure for each one to `lab03_results.html`, and stops at the first step that isn't done yet.
3. Open `lab03_results.html`, check the new figure looks right, then move on to the next step.

You can also run `pytest test_lab03.py` at any point — the tests are grouped by step, so the failing tests tell you which step needs attention.

> **Image coordinates:** `x` runs left → right and `y` runs **top → bottom**, so `y = 0` is the top of the image. A *higher* fluid level therefore has a *smaller* `y` value.

## The Pipeline at a Glance

```
greyscale image
  → 1. threshold          bottles become white shapes on black
  → 2. close_gaps         small holes inside the shapes are filled
  → 3. detect_edges       just the outlines are kept
  → 4. mask_roi           only the edges around the bottle necks are kept
  → 5. find_lines         straight line segments are found
  → 6. select_horizontal  only the (nearly) flat lines are kept — the fluid surfaces
  → 7. calculate_levels   one y value per bottle
  → 8. detect_fluid_levels  runs steps 1–7 in order
```

## Background

### Thresholding
A **threshold** turns a greyscale image into a black-and-white (binary) one: every pixel brighter than the threshold becomes one value, everything else the other. The bottles and fluid are *dark* on a *light* background, so we use an **inverted** threshold (`cv2.THRESH_BINARY_INV`) — dark pixels become white (255) and the bright background becomes black (0).

### Morphological closing (lesson `01_05_morphology`, "Opening and closing")
**Closing** is a dilation followed by an erosion. It fills small holes and gaps inside white shapes without making them noticeably bigger. The size of the square **structuring element** (the kernel) controls how big a gap gets filled.

### Canny edge detection (lesson `02_02_canny_hough`, "Canny edge detection")
The **Canny** detector finds thin, reliable edges. It takes two thresholds (low and high) for its hysteresis step and returns an image where edge pixels are 255 and everything else is 0.

### Region of interest
The fluid surfaces are all near the necks of the bottles, so we ignore everything else. The **region of interest (ROI)** is a polygon — here a rectangle covering rows 80–200 — and we zero out every edge pixel outside it. This stops the bottle bases, labels and background from producing unwanted lines.

### The Hough transform (lesson `02_02_canny_hough`, "Lines from edges")
The **Hough transform** finds straight lines by letting every edge pixel "vote" for all the lines that could pass through it; lines with many votes are real lines in the image. OpenCV's **probabilistic** version, `cv2.HoughLinesP`, returns line *segments* as their two end points `(x1, y1, x2, y2)`.

### Slope
The **slope** of a line is `(y2 − y1) / (x2 − x1)` — how much it rises for each step across. A perfectly horizontal line has slope 0. A vertical line has `x1 == x2`, so its slope can't be calculated — skip those lines.

## Your Task

### Class: `FluidLevelDetector`

**Constructor:** `FluidLevelDetector(threshold_value=240, **kwargs)` — all settings have defaults, so `FluidLevelDetector()` works with no arguments. Instantiating the class is already done for you in `evaluate_lab03.py`.

The constructor is **already written for you**. It stores every setting on `self`, and your methods should use these (e.g. `self.threshold_value`) rather than typing the numbers in directly — the tests check this by changing the settings.

| Setting | Default | Used in | Meaning |
|---------|---------|---------|---------|
| `self.threshold_value` | 240 | Step 1 | Brightness threshold for binarising the image. |
| `self.kernel_size` | 5 | Step 2 | Size of the square closing kernel. |
| `self.canny_low`, `self.canny_high` | 50, 150 | Step 3 | Canny hysteresis thresholds. |
| `self.roi` | rectangle, rows 80–200 | Step 4 | Region-of-interest polygon. |
| `self.hough_rho`, `self.hough_theta` | 1, π/180 | Step 5 | Hough distance (px) and angle (radians) resolution. |
| `self.hough_threshold` | 30 | Step 5 | Minimum votes for a line. |
| `self.min_line_len`, `self.max_line_gap` | 20, 20 | Step 5 | Shortest line kept; largest gap joined. |
| `self.slope_threshold` | 0.1 | Step 6 | Largest \|slope\| still counted as horizontal. |

### Methods to Implement

| Step | Method | Input → Output |
|------|--------|----------------|
| 1 | `threshold(image_gray)` | greyscale image → binary image |
| 2 | `close_gaps(binary)` | binary image → binary image |
| 3 | `detect_edges(binary)` | binary image → edge image |
| 4 | `mask_roi(edges)` | edge image → edge image (ROI only) |
| 5 | `find_lines(masked_edges)` | edge image → `(N, 4)` array of lines |
| 6 | `select_horizontal_lines(lines)` | lines → list of horizontal lines |
| 7 | `calculate_fluid_levels(horizontal_lines, image_width)` | lines → `[y_left, y_mid, y_right]` |
| 8 | `detect_fluid_levels(image_gray)` | greyscale image → `[y_left, y_mid, y_right]` |

## Step-by-Step Walkthrough

### Step 1 — `threshold(image_gray)`

**Goal:** dark bottles become white shapes on a black background.

`cv2.threshold` returns **two** values — the threshold used, and the image. You only need the image:

```python
_, binary = cv2.threshold(image_gray, self.threshold_value, 255, cv2.THRESH_BINARY_INV)
return binary
```

**Check:** the Step 1 figure shows white bottle shapes on black.

### Step 2 — `close_gaps(binary)`

**Goal:** fill small holes inside the white shapes.

1. Make a square kernel of ones: `kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)`
2. Apply closing: `cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)` and return the result.

**Check:** compared with Step 1, small black specks inside the shapes have gone.

### Step 3 — `detect_edges(binary)`

**Goal:** keep only the outlines of the shapes.

Return `cv2.Canny(binary, self.canny_low, self.canny_high)`.

**Check:** the Step 3 figure shows thin white outlines on black.

### Step 4 — `mask_roi(edges)`

**Goal:** remove every edge outside the region of interest.

1. Make a blank (all-zero) mask the same size as the edge image: `mask = np.zeros_like(edges)`
2. Fill the region of interest with white: `cv2.fillPoly(mask, self.roi, 255)`
3. Keep only the edges where the mask is white:
   ```python
   return cv2.bitwise_and(edges, edges, mask=mask)
   ```

**Check:** in the Step 4 figure, only the edges inside the red outline remain.

### Step 5 — `find_lines(masked_edges)`

**Goal:** find the straight line segments.

1. Call the Hough transform with the settings from the constructor:
   ```python
   lines = cv2.HoughLinesP(
       masked_edges,
       self.hough_rho, self.hough_theta, self.hough_threshold,
       minLineLength=self.min_line_len,
       maxLineGap=self.max_line_gap,
   )
   ```
2. `HoughLinesP` returns `None` if it finds nothing — in that case return an empty array: `np.empty((0, 4), dtype=np.int32)`.
3. Otherwise, reshape so there is **one row per line** with four columns `(x1, y1, x2, y2)`: `return lines.reshape(-1, 4)`. (Different OpenCV versions return slightly different shapes; `reshape(-1, 4)` makes them all the same.)

**Check:** the Step 5 figure shows blue line segments drawn over the bottles.

### Step 6 — `select_horizontal_lines(lines)`

**Goal:** keep only the (nearly) flat lines — these are the fluid surfaces.

Loop over the lines, work out each one's slope, and keep it if the slope is small enough:

```python
horizontal = []
for x1, y1, x2, y2 in lines:
    if x2 == x1:          # vertical line — slope can't be calculated, skip it
        continue
    slope = (y2 - y1) / (x2 - x1)
    if abs(slope) <= self.slope_threshold:
        horizontal.append((x1, y1, x2, y2))
return horizontal
```

**Check:** the Step 6 figure shows only flat green lines, at the fluid surfaces.

### Step 7 — `calculate_fluid_levels(horizontal_lines, image_width)`

**Goal:** one fluid level (a `y` value) for each bottle.

1. The three bottles sit in three equal-width columns. Each column is `image_width / 3` pixels wide — so `x` from `0` to `image_width / 3` is the left bottle, and so on.
2. Make three empty lists, one per bottle: `groups = [[], [], []]`
3. For each line, find its centre — `x_centre = (x1 + x2) / 2` and `y_centre = (y1 + y2) / 2` — and work out which column it is in:
   ```python
   column = min(int(x_centre // (image_width / 3)), 2)   # 0, 1 or 2
   groups[column].append(y_centre)
   ```
   (The `min(..., 2)` stops a line right on the far edge giving column 3.)
4. Each bottle's level is the **median** of its `y_centre` values (`np.median`). A bottle with no lines gets `float('inf')`:
   ```python
   return [float(np.median(g)) if g else float('inf') for g in groups]
   ```

The median is used rather than the mean because it ignores the odd stray line.

**Check:** the Step 7 figure shows a red marker at the fluid surface in each bottle, with the middle one clearly higher.

### Step 8 — `detect_fluid_levels(image_gray)`

**Goal:** run the whole pipeline. Call your own methods in order, storing each result on `self` so it can be displayed afterwards:

```python
self.thresholded      = self.threshold(image_gray)
self.closed           = self.close_gaps(self.thresholded)
self.edges            = self.detect_edges(self.closed)
self.masked_edges     = self.mask_roi(self.edges)
self.lines            = self.find_lines(self.masked_edges)
self.horizontal_lines = self.select_horizontal_lines(self.lines)
self.fluid_levels     = self.calculate_fluid_levels(self.horizontal_lines, image_gray.shape[1])
return self.fluid_levels
```

`image_gray.shape[1]` is the image width (`shape` is `(height, width)`).

**Check:** the evaluate script prints the three levels, how close the left and right bottles are, and how much higher the middle bottle is.

## What the Pytests Check

Run the tests with:

```bash
pytest test_lab03.py
```

- **Steps 1–3** — each result matches the expected OpenCV output exactly, both with the default settings and with changed settings (so the settings on `self` must be used).
- **Step 4** — everything outside the region of interest is removed, everything inside is kept, and a different `roi` setting is respected.
- **Step 5** — lines come back as an `(N, 4)` array, all inside the region of interest; a blank image gives an empty result rather than `None`.
- **Step 6** — flat lines are kept, steep and vertical lines are removed, and `slope_threshold` is respected.
- **Step 7** — lines are grouped into the correct third, the median `y` is used, and an empty bottle gives `float('inf')`.
- **Step 8** — three levels are returned, all bottles are detected, intermediate results are stored on `self`, the left and right levels agree within **10 px**, and the middle level is at least **30 px** higher than both.

## Evaluation Script

`evaluate_lab03.py` loads the bottle image, runs each step in turn, and saves one figure per completed step to `lab03_results.html`:

- the input image;
- **Steps 1–3** — the thresholded, closed and edge images;
- **Step 4** — the masked edges, with the region of interest outlined in red;
- **Step 5** — all Hough line segments (blue) over the image;
- **Step 6** — the horizontal lines (green);
- **Step 7** — the final fluid level per bottle (red markers).

Hover over any figure to read `x` and `y` positions.

Run it with:

```bash
python evaluate_lab03.py
```

Then open `lab03_results.html` — in VS Code / Codespaces, right-click the file → **Open with Live Server**, or run `python -m http.server 5000` and open the forwarded port.
