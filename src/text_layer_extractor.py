import os
import re
import cv2
import fitz
import subprocess
import xml.etree.ElementTree as ET

from lxml import etree
from PIL import Image
from loguru import logger
from collections import defaultdict


def extract_bboxes_from_pdf(xml_path, images_folder_dir):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    namespaces = {'xhtml': 'http://www.w3.org/1999/xhtml'}

    pages_text_layer = defaultdict(list)

    for idx, page in enumerate(root.findall('.//xhtml:page', namespaces=namespaces), start=1):
        # Get page dimensions from XML
        xml_width = float(page.get('width'))
        xml_height = float(page.get('height'))

        image_dir = os.path.join(images_folder_dir, f"image_{idx}.jpeg")
        # Load the corresponding image and get its dimensions
        image = Image.open(image_dir)
        img_width, img_height = image.size

        # Calculate scaling factors
        x_scale = img_width / xml_width
        y_scale = img_height / xml_height

        # Draw a rectangle for each word in the page
        for word in page.findall('.//xhtml:word', namespaces=namespaces):
            # Scale the coordinates from the XML to the image size
            x1 = int(float(word.get('xMin')) * x_scale)
            y1 = int(float(word.get('yMin')) * y_scale)
            x2 = int(float(word.get('xMax')) * x_scale)
            y2 = int(float(word.get('yMax')) * y_scale)

            pages_text_layer[idx].append((x1, y1, x2, y2))
        # Save each page's image with bounding boxes
    return pages_text_layer


def extract_image_bboxes(pdf_path, image_folder_dir):
    logger.info(f'Extracting text layer from {os.path.basename(pdf_path)}')

    """
    Extract text bounding boxes from each page of a PDF using PyMuPDF and resize them to fit the corresponding images.

    Parameters:
    - pdf_path (str): Path to the input PDF file.
    - image_folder_dir (str): Directory where page images are stored.

    Returns:
    - Dictionary mapping page numbers to a list of dictionaries containing adjusted bounding box information.
    """

    image_coordinates = defaultdict(list)

    # Open the PDF document
    pdf_document = fitz.open(pdf_path)
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        # Open the corresponding image file
        image_name = os.path.join(image_folder_dir, f'image_{page_num + 1}.jpeg')
        image = Image.open(image_name)
        img_width, img_height = image.size

        # Get page size from the rendered page
        w, h = fitz.utils.get_pixmap(page).w, fitz.utils.get_pixmap(page).h

        # Calculate scaling factors
        x_scale = img_width / w
        y_scale = img_height / h

        blocks = page.get_text("blocks")

        # Iterate through each block to adjust the bounding boxes
        for block in blocks:
            if block[-1] == 1:  # Block is an image
                x1, y1, x2, y2 = block[:4]
                # Apply scaling to the bounding box
                adjusted_bbox = (x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale)
                image_coordinates[page_num + 1].append({
                    'width': img_width,
                    'height': img_height,
                    'bbox': adjusted_bbox
                })
                logger.info(f'Found image on page {page_num + 1}: {adjusted_bbox}')

    pdf_document.close()

    return image_coordinates


def draw_rectangles(image_path, bounding_boxes, output_path):
    """
    Draw rectangles on an image based on the provided bounding boxes using OpenCV.

    Parameters:
    - image_path (str): Path to the input image file.
    - bounding_boxes (list): List of dictionaries containing bounding box coordinates.
                            Each dictionary should have a 'bbox_pixels' key with a list of four coordinates.
    - output_path (str): Path to save the output image with rectangles.

    Returns:
    - None
    """

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Draw rectangles on the image
    for bbox_info in bounding_boxes:
        if len(bbox_info) == 4:
            cv2.rectangle(image, (bbox_info[0], bbox_info[1]), (bbox_info[2], bbox_info[3]), thickness=1, color=(0, 0, 255))
    # Save the modified image
    cv2.imwrite(output_path, image)


def extract_text_with_bboxes(pdf_path, xml_output_path):
    # Check if PDF file exists
    if not os.path.exists(pdf_path):
        logger.error(f"The specified PDF file does not exist: {pdf_path}")
        return

    # Define the command to extract bounding boxes into an XML file
    command = ['pdftotext', '-bbox', pdf_path, xml_output_path]

    try:
        # Run the command and capture output and errors
        result = subprocess.run(command, check=False, text=True, capture_output=True)

        if result.stderr:
            if "syntax error" in result.stderr.lower():
                logger.warning("Warning encountered: Known character collection issue.")
            else:
                logger.error(f"Other warnings or errors: {result.stderr}")

        # Validate and save the XML content if no syntax errors are reported
        if "syntax error" not in result.stderr.lower():
            with open(xml_output_path, 'r') as file:
                xml_content = file.read()
            validate_and_save_xml(xml_content, xml_output_path)

    except Exception as e:
        # Handle other exceptions that may occur
        logger.error(f"An unexpected error occurred: {e}")


def validate_and_save_xml(xml_content, xml_output_path):
    try:
        # Attempt to parse the XML to check if it's well-formed
        etree.fromstring(xml_content)
        logger.info("XML file has been validated successfully.")
    except etree.XMLSyntaxError as e:
        logger.warning(f"XML syntax error encountered: {e}. Attempting to cleanse the XML content.")
        # Try to cleanse the XML content
        xml_content = cleanse_xml_content(xml_content)
        try:
            # Attempt to re-parse the cleansed XML
            etree.fromstring(xml_content)
            logger.info("XML file has been cleansed and re-validated successfully.")
        except etree.XMLSyntaxError as e:
            # Log the error and save an empty XML as a fallback
            logger.error(f"Failed to create XML file after cleansing due to XML syntax error: {e}")
            xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<root></root>'

    # Save the (potentially modified) XML content
    with open(xml_output_path, 'w') as file:
        file.write(xml_content)
    logger.info("XML file has been created and saved.")


def cleanse_xml_content(xml_content):
    # Implement XML content cleansing to remove non-printable characters
    cleansed_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_content)
    return cleansed_content
