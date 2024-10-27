import cv2
import pytesseract
from PIL import Image
import numpy as np

# Set path to Tesseract executable (Windows only)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('test.png')
# Preprocessing function to clean up the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Load the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return thresh_img

# Function to perform OCR
def perform_ocr(preprocessed_img):
    pil_img = Image.fromarray(preprocessed_img)  # Convert to PIL format
    text = pytesseract.image_to_string(preprocessed_img, lang='eng',  \
        config='--psm 13 --oem 1 -c tessedit_char_whitelist=0123456789')  # Perform OCR
    return text

# Main function to run preprocessing and OCR
def main(image_path):
    preprocessed_img = preprocess_image(image_path)  # Preprocess the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(preprocessed_img, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    extracted_text = perform_ocr(invert)  # Perform OCR
    print("Extracted Text:\n", extracted_text)  # Display the text

# Specify the path to your image
image_path = 'test.png'  # Replace with your actual image file path

# Run the main function
main(image_path)
