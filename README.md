# Text-Layer-Classifier
The Text Layer Classifier model is designed to determine the optimal method for extracting text from PDF pages. It evaluates each page to decide whether to use Optical Character Recognition (OCR) or the PDF's native text layer for word extraction.


## Installation
```bash
git clone https://github.com/your-repo/text-layer-validation.git
cd text-layer-validation
```
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
To run the model, ensure you set the correct pdf_path in model.py, then execute the following command:
```bash
python3 model.py
```


## Configuration
`draw_text`: boolean, default=False. If True, the model will draw the extracted text on the PDF pages.
`draw_contour`: boolean, default=False. If True, the model will draw the contours of the text boxes on the PDF pages.