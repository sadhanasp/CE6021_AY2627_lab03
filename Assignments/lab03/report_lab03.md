# Lab Report — Lab 03: Fluid Level Detection via the Hough Transform

*Answer each question briefly and factually — a number, a name, or a short phrase is enough.
Each question is marked on whether the required information is present and correct, not on how
it's written, so there is no benefit to writing full sentences or extra explanation.*

**Q1.** In one phrase, what does `detect_fluid_levels()` compute?
Answer:Fluid levels in the left, middle, and right sections of the image.
**Q2.** Which OpenCV function did you use in Step 1 (`threshold`) to binarise the image?
Answer:('cv2.threshold')
**Q3.** What `kernel_size` did you use for the Step 2 (`close_gaps`) morphological closing?
Answer: '5*5 kernel'
**Q4.** What `canny_low` / `canny_high` values did you use for Step 3 (`detect_edges`)?
Answer:'50/150'
**Q5.** In one phrase, what does the Step 4 (`mask_roi`) region-of-interest mask remove from the
edge image?
Answer:'Edges outside the region of interest (ROI)'
**Q6.** Which OpenCV function does Step 5 (`find_lines`) use to find line segments?
Answer: ('cv2.HoughLinesP')
**Q7.** In Step 6 (`select_horizontal_lines`), how is a line's slope calculated from its endpoints
`(x1, y1, x2, y2)`?
Answer: '(y2 - y1) / (x2 - x1)'