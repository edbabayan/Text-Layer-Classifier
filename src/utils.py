from pathlib import Path
import cv2
import numpy as np
from loguru import logger
from openpyxl import Workbook
from openpyxl.styles import Font


def convert_bboxes_to_pixels(page_image, bboxes_info):
    """
    Convert bounding box coordinates from PDF-relative to image-relative.

    Parameters:
    - page_image: The image of the PDF page.
    - bboxes_info: Information about bounding boxes, including height, width, and coordinates.

    Returns:
    list: List of converted bounding box coordinates in image-relative pixels.
    """
    image_width = page_image.shape[1]
    image_height = page_image.shape[0]

    converted_coordinates = []
    for info in bboxes_info:
        pdf_height = info['height']
        pdf_width = info['width']
        x1, y1, x2, y2 = info['bbox']
        x1 = int(x1 * image_width / pdf_width)
        y1 = int(y1 * image_height / pdf_height)
        x2 = int(x2 * image_width / pdf_width)
        y2 = int(y2 * image_height / pdf_height)
        converted_coordinates.append((x1, y1, x2, y2))
    return converted_coordinates


def draw_rectangles(image_path, rectangles):
    image = cv2.imread(image_path)

    for (x, y, w, h) in rectangles:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


def calculate_iou(box1, box2, input_format="xyxy"):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Parameters:
    - box1: Tuple or List representing the coordinates of the first bounding box.
    - box2: Tuple or List representing the coordinates of the second bounding box.
    - input_format: String specifying the input format. Options: "xyxy" or "xywh".
                    "xyxy" format: (x1, y1, x2, y2)
                    "xywh" format: (x, y, w, h)

    Returns:
    - IoU: Intersection over Union.
    """
    if input_format == "xywh":
        box1 = (box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3])
        box2 = (box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3])

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area

    return iou


def convert_image_coord_representation(page_images):
    """
    Convert image coordinates from (x1, y1, x2, y2) to (x, y, w, h).

    Parameters:
    - page_images: List of image coordinates in (x1, y1, x2, y2) format.

    Returns:
    list: List of image coordinates in (x, y, w, h) format.
    """
    new_coordinates = []
    for image in page_images:
        x = image['bbox'][0]
        y = image['bbox'][1]
        w = image['bbox'][2] - image['bbox'][0]
        h = image['bbox'][3] - image['bbox'][1]
        new_coordinates.append((x, y, w, h))
    return new_coordinates


def convert_text_coord_representation(page_text_coordinates):
    """
    Convert text coordinates from (x1, y1, w, h) to (x1, y1, x2, y2).

    Parameters:
    - page_text_coordinates: List of text coordinates in (x1, y1, w, h) format.

    Returns:
    list: List of text coordinates in (x1, y1, x2, y2) format.
    """
    new_coordinates = []
    for text_coordinate in page_text_coordinates:
        x1 = text_coordinate[0]
        y1 = text_coordinate[1]
        x2 = text_coordinate[2] + text_coordinate[0]
        y2 = text_coordinate[3] + text_coordinate[1]
        new_coordinates.append((x1, y1, x2, y2))
    return new_coordinates


def is_bbox_inside(bbox1, bbox2):
    """
    Check if one bounding box is strictly inside another bounding box.

    Parameters:
    - bbox1: Tuple or List representing the coordinates of the first bounding box.
    - bbox2: Tuple or List representing the coordinates of the second bounding box.

    Returns:
    - True if bbox1 is strictly inside bbox2, False otherwise.
    """
    x1, y1, x2, y2 = bbox1
    x1_inside = bbox2[0] < x1 < bbox2[2]
    y1_inside = bbox2[1] < y1 < bbox2[3]
    x2_inside = bbox2[0] < x2 < bbox2[2]
    y2_inside = bbox2[1] < y2 < bbox2[3]

    return x1_inside and y1_inside and x2_inside and y2_inside


def draw_text_layer_rectangles(image, bounding_boxes, output_path, image_bboxes=None):
    """
    Draw rectangles on an image based on the provided bounding boxes using OpenCV.

    Parameters:
    - image (numpy.ndarray): The input image array.
    - bounding_boxes (list): List of dictionaries containing bounding box coordinates.
                            Each dictionary should have a 'bbox_pixels' key with a list of four coordinates.
    - output_path (str): Path to save the output image with rectangles.
    - image_bboxes (list): List of dictionaries containing image bounding box coordinates.
    - contrast (list): List of tuples where each tuple represents the contrast values for the three RGB channels.

    Returns:
    - None
    """
    for bbox_info in bounding_boxes:
        if len(bbox_info) == 4:
            cv2.rectangle(image, (bbox_info[0], bbox_info[1]), (bbox_info[2], bbox_info[3]), thickness=1,
                          color=(0, 0, 255))

    if image_bboxes:
        for i, bbox_info in enumerate(image_bboxes):
            if len(bbox_info) == 4:
                x, y, w, h = bbox_info
                cv2.rectangle(image, (x, y), (x + w, y + h), thickness=1, color=(0, 255, 0))

    cv2.imwrite(output_path, image)


def is_document_page(image_arr, roi_coordinates_list, saturation_threshold=30, value_threshold=120,
                     contrast_threshold=15):
    # Load the image
    processed_images = []

    for roi_coordinates in roi_coordinates_list:
        # Unpack and convert coordinates
        x, y, w, h = map(int, roi_coordinates)

        # Extract ROI from the image
        roi = image_arr[y:y + h, x:x + w]

        if roi.any():
            # Convert ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            saturation_channel = hsv_roi[:, :, 1]
            value_channel = hsv_roi[:, :, 2]

            # Calculate mean saturation and value
            mean_saturation = np.mean(saturation_channel)
            mean_value = np.mean(value_channel)

            # Convert ROI to grayscale for contrast calculation
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Calculate standard deviation of the grayscale image
            std_deviation = np.std(gray_roi)

            # Check against thresholds for saturation, value, and contrast
            if (
                    mean_saturation >= saturation_threshold or mean_value > value_threshold) or std_deviation > contrast_threshold:
                processed_images.append((x, y, w, h))
            else:
                print(
                    f"ROI at ({x}, {y}) rejected: Saturation={mean_saturation}, Value={mean_value}, Contrast={std_deviation}")

    return processed_images


def confidence_excel_output(eval_results, path_to_save, threshold=0.9):
    """
    Takes document results and path where to save the result. Returns Excel where also includes final confidence scores.
    :param eval_results: List of defaultdicts containing document results
    :param path_to_save: Path to save the Excel file
    :param threshold: Threshold that will decide weather the text layer is usable or not
    :return: Excel output of evaluation
    """

    wb = Workbook()
    ws_summary = wb["Sheet"]
    ws_summary.title = "SUMMARY"

    ws_summary.append(['PDF Name'] + ['Decision'])

    for i, doc_results_dict in enumerate(eval_results):
        for pdf_name, doc_results_list in doc_results_dict.items():
            if not doc_results_list:
                ws_summary.append([pdf_name] + [None] + ['OCR'])
                continue

            ws_doc_data = wb.create_sheet(pdf_name)

            ws_doc_data['A1'] = 'Page Number'
            ws_doc_data['B1'] = 'Confidence'

            for a, doc_result in enumerate(doc_results_list, start=2):
                for page_num, accuracy in doc_result.items():
                    ws_doc_data[f'A{a}'] = page_num
                    ws_doc_data[f'B{a}'] = accuracy

            bold_font = Font(bold=True)
            for cell in ws_doc_data["1:1"]:
                cell.font = bold_font

            result = 'Text layer' if sum(1 for item in doc_results_list if 'Text layer' in item.values()) / len(
                doc_results_list) > threshold else 'OCR'

            ws_summary.append([pdf_name] + [result])

    wb.save(path_to_save)


def calculate_contrast_in_roi_rgb(image, roi_coordinates_list, threshold=50, lower_intensity_threshold=40,
                                  upper_intensity_threshold=230):
    """
    Calculate contrast in a region of interest (ROI) within the RGB image, excluding pixels below lower_intensity_threshold
    and above upper_intensity_threshold.

    Parameters:
    - image: The input RGB image.
    - roi_coordinates_list: List of tuples (x, y, w, h) defining the ROI coordinates.
    - threshold: Contrast threshold. ROIs with contrast higher than this will be processed.
    - lower_intensity_threshold: Pixel intensity threshold below which pixels are considered black.
    - upper_intensity_threshold: Pixel intensity threshold above which pixels are considered white.

    Returns:
    - processed_images: List of tuples (x, y, w, h) for ROIs with contrast higher than the specified threshold.
    """
    processed_images = []

    if roi_coordinates_list:
        for roi_coordinates in roi_coordinates_list:
            x, y, w, h = roi_coordinates

            roi_r = image[y:y + h, x:x + w, 0]
            roi_g = image[y:y + h, x:x + w, 1]
            roi_b = image[y:y + h, x:x + w, 2]

            roi_r = np.clip(roi_r, lower_intensity_threshold, upper_intensity_threshold)
            roi_g = np.clip(roi_g, lower_intensity_threshold, upper_intensity_threshold)
            roi_b = np.clip(roi_b, lower_intensity_threshold, upper_intensity_threshold)

            contrast_r = np.std(roi_r)
            contrast_g = np.std(roi_g)
            contrast_b = np.std(roi_b)

            if contrast_r > threshold and contrast_g > threshold and contrast_b > threshold:
                processed_images.append((x, y, w, h))

    return processed_images


def count_bboxes_within(bbox, bbox_list):
    x1, y1, x2, y2 = bbox
    count = 0
    for box in bbox_list:
        bx1, by1, bx2, by2 = box
        if x1 <= bx1 and y1 <= by1 and x2 >= bx2 and y2 >= by2:
            count += 1
    if count == 2:
        count = 1
    return count


def delete_folder(path: Path):
    for item in path.iterdir():
        logger.warning(f"Deleting {item}")
        if item.is_dir():
            delete_folder(item)  # Recursively delete subdirectory
        else:
            item.unlink()  # Delete file
    path.rmdir()