from pathlib import Path


class CFG:
    root_dir = Path(__file__).parent.parent.absolute()
    images_dir = root_dir.joinpath('cropped_images')
    src_dir = root_dir.joinpath('src')
    data_dir = root_dir.joinpath('data')
    text_layer_output_folder_path = data_dir.joinpath('text_layer_images')
    contour_layer_output_folder_path = data_dir.joinpath('contour_images')
    model_output_dir = data_dir.joinpath('model_output')
    confidence_output_dir = model_output_dir.joinpath('confidence_metrics.xlsx')
    reporting_path = data_dir.joinpath('model_metrics')
    model_metrics_path = reporting_path.joinpath("model_metrics.xlsx")
    xml_output_path = src_dir.joinpath('output.xml')
