import os
from loguru import logger
from pdf2image import convert_from_path


def extract_images_from_pdf(pdf_path, image_folder):
    """
     Convert PDF pages to images using the specified DPI and save them to the output directory.

     Parameters:
     - pdf_path (str): The path to the input PDF file.
     """
    # Convert PDF to images
    images = convert_from_path(pdf_path)

    # Save each page as an image
    for i, image in enumerate(images):
        image_path = os.path.join(image_folder, f"image_{i + 1}.jpeg")
        image.save(image_path, 'JPEG')


def clean_folder_from_files(folder_path):
    """
    Remove all files from the specified folder.

    Parameters:
    - folder_path: String, the path to the folder to be cleaned.
    """
    if os.path.exists(folder_path):
        # Get the list of files in the specified folder
        files_in_folder = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Remove each file
        for file_name in files_in_folder:
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)

        logger.info(f"All files in the folder at '{folder_path}' have been removed.")
    else:
        logger.warning(f"The folder at '{folder_path}' does not exist.")