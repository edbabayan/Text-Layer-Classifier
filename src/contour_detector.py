import cv2
from src.utils import calculate_iou


def detect_text_regions(image, page_image_coord=None):
    """
    Detect text regions in the input image using edge detection and contour analysis.

    Parameters:
    - image: The input image.
    - page_image_coord: Coordinates of other images on the page to exclude overlapping text regions.

    Returns:
    list: List of text regions represented as (x, y, w, h) coordinates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    page_contours = []

    page_height, page_width = image.shape[:2]

    for contour in contours:
        if cv2.contourArea(contour) > 0:
            x, y, w, h = cv2.boundingRect(contour)
            # Ensure each contour is smaller than the specified size and also smaller than half the image dimensions
            if (w > 5 and h > 5) and (w < page_width / 2 and h < page_height / 2):
                page_contours.append((x, y, w, h))

    filtered_contours = []
    for contour in page_contours:
        if not any(calculate_iou(contour, page_image, input_format='xywh') > 0 for page_image in page_image_coord):
            filtered_contours.append(contour)

    page_contours = filtered_contours if page_image_coord else page_contours
    margin_of_error = 10
    page_contours = sorted(page_contours, key=lambda tup: (tup[1] // margin_of_error, tup[0]))

    return page_contours


def sort_contours_line_by_line(contours):
    sorted_contours = sorted(contours, key=lambda x: x[1])

    lines = []
    current_line = [sorted_contours[0]]
    for contour in sorted_contours[1:]:
        if abs(contour[1] - current_line[-1][1]) <= 7:
            current_line.append(contour)
        else:
            lines.append(sorted(current_line, key=lambda x: x[0]))  # Sort contours in the line by x-coordinate
            current_line = [contour]
    lines.append(sorted(current_line, key=lambda x: x[0]))  # Add the last line

    used_lines = set()
    next_line_idx = 0
    combined_lines = []
    combined = False
    for line_idx, line in enumerate(lines):
        if combined:
            combined = False
            next_line_idx += 1
            continue
        if line_idx not in used_lines and next_line_idx != len(lines) - 1:
            used_lines.add(line_idx)
            current_max_y_plus_h = max(box[1] + box[3] for box in line)
            current_lowest_y = min(box[1] for box in line)
            next_line_idx = line_idx + 1
            next_line = lines[next_line_idx]
            next_lowest_y = min(box[1] for box in next_line)
            if current_lowest_y <= next_lowest_y <= current_max_y_plus_h:
                combined_line = line + next_line
                combined_lines.append(combined_line)
                used_lines.add(next_line_idx)
                combined = True
            else:
                combined_lines.append(line)
        elif line_idx not in used_lines and next_line_idx == len(lines) - 1:
            combined_lines.append(line)
        else:
            pass
    return combined_lines


def join_contours(contours):
    contours_by_line = sort_contours_line_by_line(contours)
    overall_contours = []
    for line in contours_by_line:
        used = set()
        for index_1, contour_1 in enumerate(line):
            if index_1 in used:
                continue
            merged_contour = contour_1
            for index_2, contour_2 in enumerate(line):
                if index_1 != index_2 and index_2 not in used:
                    if is_close(merged_contour, contour_2):
                        merged_contour = combine_contour_coordinates(merged_contour, contour_2)
                        used.add(index_2)
            overall_contours.append(merged_contour)
            used.add(index_1)

    return overall_contours


def is_close(contour1, contour2, threshold=7):
    x1, y1, w1, h1 = contour1
    x2, y2, w2, h2 = contour2
    horizontal_close = (x1 + w1 + threshold >= x2 >= x1 - threshold) or \
                       (x2 + w2 + threshold >= x1 >= x2 - threshold)
    return horizontal_close


def combine_contour_coordinates(contour_1, contour_2):
    x = min(contour_1[0], contour_2[0])
    y = min(contour_1[1], contour_2[1])
    x2_max = max(contour_1[0] + contour_1[2], contour_2[0] + contour_2[2])
    y2_max = max(contour_1[1] + contour_1[3], contour_2[1] + contour_2[3])
    w = x2_max - x
    h = y2_max - y
    return x, y, w, h


