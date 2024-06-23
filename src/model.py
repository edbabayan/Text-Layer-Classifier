import os
import cv2
import pandas as pd
from loguru import logger
from openpyxl import Workbook
from collections import defaultdict
from src.contour_detector import detect_text_regions, join_contours
from src.text_layer_extractor import (extract_bboxes_from_pdf,
                                      extract_image_bboxes,
                                      extract_text_with_bboxes)
from src.pdf_cropper import (extract_images_from_pdf,
                             clean_folder_from_files)
from src.utils import (calculate_iou,
                       convert_image_coord_representation,
                       convert_text_coord_representation,
                       is_bbox_inside,
                       draw_text_layer_rectangles,
                       is_document_page,
                       count_bboxes_within,
                       delete_folder)
from src.config import CFG


class TextValidator:
    def __init__(self, draw_text=False, draw_contour=False, threshold: float = 0.9):
        self.per_page_result = None
        self.images_dir = CFG.images_dir
        self.text_layer_output_folder_path = CFG.text_layer_output_folder_path
        self.contour_layer_output_folder_path = CFG.contour_layer_output_folder_path
        self.draw_text = draw_text
        self.draw_contour = draw_contour
        self.threshold = threshold
        self.xml_output_path = CFG.xml_output_path

    def calculate_confidence(self, output_dict, pdf_name):
        return self.evaluate_pages(output_dict=output_dict, pdf_name=pdf_name)

    @staticmethod
    def evaluate_pages(output_dict, pdf_name):
        logger.info(f"Evaluating {pdf_name}")
        result_dict = defaultdict(list)

        if output_dict:
            for page_num, decision in output_dict.items():
                result_dict[pdf_name].append({page_num: decision})
        else:
            result_dict = {pdf_name: []}
        return result_dict

    def predict(self, pdf_path: str):
        logger.info('Creating images from pdf by crop_pdf')
        self.crop_pdf(pdf_path)
        logger.info(f'Images path: {self.images_dir}')
        pdf_name = os.path.basename(pdf_path)
        logger.info(f"Processing {pdf_name}")
        image_bboxes = extract_image_bboxes(pdf_path, self.images_dir)
        extract_text_with_bboxes(pdf_path, self.xml_output_path)
        text_bboxes = extract_bboxes_from_pdf(self.xml_output_path, self.images_dir)
        if not text_bboxes:
            logger.warning(f"No text layer in {pdf_name}")
            return {i: 'OCR' for i in range(1, len(os.listdir(self.images_dir)) + 1)}
        text_bboxes, image_bboxes = self.preprocess(text_bboxes=text_bboxes, image_bboxes=image_bboxes)
        self.draw_text_layer(text_bboxes=text_bboxes, image_bboxes=image_bboxes)
        page_contours = self.detect_text_contours(pdf_name=pdf_name, image_bboxes=image_bboxes)
        self.draw_contour_layer(page_contours=page_contours)
        os.remove(self.xml_output_path)
        delete_folder(self.images_dir)
        return self.predict_for_document(text_bboxes=text_bboxes,
                                         page_contours=page_contours)

    def preprocess(self, text_bboxes, image_bboxes):
        processed_images_dict = defaultdict(list)
        for page_num, _ in enumerate(range(1, len(text_bboxes.keys()) + 1), start=1):
            page_image_path = os.path.join(self.images_dir, f'image_{page_num}.jpeg')
            page_image_arr = cv2.imread(page_image_path)
            image_coordinates = convert_image_coord_representation(image_bboxes[page_num])
            processed_images = is_document_page(page_image_arr, image_coordinates)
            processed_images_dict[page_num] = processed_images
        return text_bboxes, processed_images_dict

    def draw_text_layer(self, text_bboxes, image_bboxes):
        """
        Visualize the model's understanding of the PDF by drawing rectangles around detected text.

        Parameters:
        - text_bboxes (defaultdict): Default dictionary containing text bounding boxes for each page.
        - pdf_document (object): The PDF document object.
        - image_bboxes (defaultdict): Default dictionary containing image bounding boxes for each page.
        """
        if self.draw_text:
            if os.path.exists(CFG.text_layer_output_folder_path):
                folder_contents = os.listdir(CFG.text_layer_output_folder_path)
                if folder_contents:
                    clean_folder_from_files(CFG.text_layer_output_folder_path)
            else:
                CFG.text_layer_output_folder_path.mkdir(exist_ok=True, parents=True)

            for page_num in range(1, len(text_bboxes.keys()) + 1):
                image_name = f"image_{str(page_num)}.jpeg"
                page_image_path = os.path.join(self.images_dir, image_name)
                page_image_arr = cv2.imread(page_image_path)
                output_dir = os.path.join(str(self.text_layer_output_folder_path), image_name)
                draw_text_layer_rectangles(page_image_arr, text_bboxes[page_num], output_dir,
                                           image_bboxes[page_num])

    def crop_pdf(self, pdf_path):
        """
        Crop images from the specified PDF file and save them in the given output folder directory.

        Parameters:
        - pdf_path (str): The path to the input PDF file.
        """
        if os.path.exists(self.images_dir):
            pass
        else:
            os.mkdir(self.images_dir)
        extract_images_from_pdf(pdf_path, self.images_dir)
        logger.info(f"Images extracted from {pdf_path}")

    def detect_text_contours(self, pdf_name, image_bboxes):
        """
        Detect text contours on each page of the PDF.

        Parameters:
        - pdf_name (str): The name of the PDF file.
        - image_bboxes (defaultdict): A defaultdict containing image bounding boxes for each page.
        - pdf_document (object): The PDF document object.

        Returns:
        defaultdict: Detected page contours excluding those within PDF image areas.
        """
        page_contours = defaultdict(list)
        logger.debug(f"Extracting contours from {pdf_name}")
        for page_num, _ in enumerate(range(1, len(image_bboxes.keys()) + 1), start=1):
            image_name = f"image_{str(page_num)}.jpeg"
            page_image_path = os.path.join(self.images_dir, image_name)
            page_image_arr = cv2.imread(page_image_path)
            character_contours = detect_text_regions(page_image_arr, image_bboxes[page_num])
            if character_contours:
                word_contours = join_contours(character_contours)
                page_contour = convert_text_coord_representation(word_contours)
                page_contours[page_num].extend(page_contour)
            else:
                page_contours[page_num].extend([])
        return page_contours

    def draw_contour_layer(self, page_contours):
        """
        Visualize the detected contours on each page of the PDF by drawing rectangles around them.

        Parameters:
        - pdf_document (object): The PDF document object.
        - page_contours (defaultdict): Default dictionary containing contours for each page.
        """
        if self.draw_contour:
            if os.path.exists(CFG.contour_layer_output_folder_path):
                folder_contents = os.listdir(CFG.contour_layer_output_folder_path)
                if folder_contents:
                    clean_folder_from_files(CFG.contour_layer_output_folder_path)
            else:
                CFG.contour_layer_output_folder_path.mkdir(exist_ok=True, parents=True)

            for page_num in range(1, len(page_contours.keys()) + 1):
                image_name = f"image_{str(page_num)}.jpeg"
                page_image_path = os.path.join(self.images_dir, image_name)
                page_image_arr = cv2.imread(page_image_path)
                output_dir = os.path.join(str(self.contour_layer_output_folder_path), image_name)
                draw_text_layer_rectangles(page_image_arr, page_contours[page_num], output_dir)

    def predict_for_document(self, text_bboxes, page_contours):
        accurate_text_index = defaultdict(set)

        for page_num in range(1, len(text_bboxes.keys()) + 1):
            for text_layer_coordinate in text_bboxes[page_num]:
                for index, contour_coordinate in enumerate(page_contours[page_num]):
                    if (is_bbox_inside(contour_coordinate, text_layer_coordinate) or
                            calculate_iou(contour_coordinate, text_layer_coordinate, input_format='xyxy') > 0):
                        accurate_text_index[page_num].add(index)

        confidence_per_page = {}

        for page_num in range(1, len(text_bboxes.keys()) + 1):
            accurate_characters_per_page = len(accurate_text_index[page_num])
            all_characters = len(page_contours[page_num])
            if accurate_characters_per_page == 0 and all_characters == 0:
                confidence_per_page[page_num] = 1.0
            elif all_characters == 0:
                confidence_per_page[page_num] = 0.0
            else:
                confidence_per_page[page_num] = accurate_characters_per_page / all_characters
                if confidence_per_page[page_num] > self.threshold:
                    if not self.check_word_boxes(page_contours[page_num], text_bboxes[page_num]):
                        confidence_per_page[page_num] = 0
        new_dict = {key: 'Text layer' if value > self.threshold else 'OCR' for key, value in
                    confidence_per_page.items()}
        return new_dict

    @staticmethod
    def check_word_boxes(contours, text_bboxes_list):
        wrong_matches = []
        for text_bbox in text_bboxes_list:
            matching_words = count_bboxes_within(text_bbox, contours)
            if matching_words != 0:
                confidence = 1 / matching_words
                wrong_matches.append(confidence)
        if not wrong_matches == 0:
            return True
        mean_value = sum(wrong_matches) / len(wrong_matches)
        return mean_value > 0.75

    def evaluate_excels(self, confidence_excel_dir, ground_truth_excel_dir):
        """
        Evaluate model performance using confidence predictions and ground truth from two Excel files.

        Parameters:
        - confidence_excel_dir (str): Path to the Excel file containing model confidence predictions.
        - ground_truth_excel_dir (str): Path to the Excel file containing ground truth data.

        Returns:
        tuple: A tuple containing two elements:
            - accuracy_list (list): List of dictionaries containing accuracy metrics for each class.
            - average_metrics (dict): Dictionary containing average accuracy metrics.
        """
        wb = Workbook()
        ws_summary = wb["Sheet"]
        ws_summary.title = "SUMMARY"

        ws_summary.append(['PDF Name'] + ['Accuracy'])

        conf_df = pd.read_excel(confidence_excel_dir)

        gt_df = pd.read_excel(ground_truth_excel_dir)

        unique_pdf_names = gt_df['PDF Name'].unique()

        accuracy_list = []

        for pdf_name in unique_pdf_names:
            if pdf_name in conf_df['PDF Name'].values:
                pdf_confidence_df = pd.read_excel(confidence_excel_dir, sheet_name=pdf_name)
                selected_rows = gt_df[gt_df['PDF Name'] == pdf_name]
                summary_df, accuracy = self.match_df(pdf_confidence_df, selected_rows)
                ws_doc_data = wb.create_sheet(pdf_name)
                ws_doc_data.append(['Page Number', 'GT', 'Prediction'])

                for row in summary_df.itertuples(index=False, name=None):
                    ws_doc_data.append(row[1:])
                ws_summary.append([pdf_name, accuracy])
                accuracy_list.append(accuracy)

        if accuracy_list:
            average_accuracy = sum(accuracy_list) / len(accuracy_list)
            ws_summary.append(['Average', average_accuracy])

        wb.save(CFG.model_metrics_path)

    @staticmethod
    def match_df(conf_df, gt_df):
        true_positive = 0
        for index, row in conf_df.iterrows():
            page_number = row['Page Number']
            confidence = row['Confidence']

            if page_number in gt_df['Page'].values:
                if confidence.lower() == gt_df.loc[gt_df['Page'] == page_number, 'Decision'].values[0].lower():
                    true_positive += 1

        merged_df = pd.merge(gt_df, conf_df, left_on='Page', right_on='Page Number', how='left')
        merged_df = merged_df.drop('Page Number', axis=1)
        merged_df = merged_df.rename(columns={'Confidence': 'Prediction'})

        accuracy = true_positive / len(gt_df['Page'])
        return merged_df, accuracy


if __name__ == '__main__':
    pdf_path = "/home/eduard/Downloads/Cascade Application and Disclosures - Eli Brown.pdf"

    conf_results = []

    model = TextValidator(draw_text=True, draw_contour=True)
    output_dict_ = model.predict(pdf_path=pdf_path)
    print(output_dict_)
