from cnocr import CnOcr
from cv2 import imread
cn_ocr = CnOcr()

def extract_text_from_image_list(img_list):
    for img in img_list:
        ocr_results.append(cn_ocr.ocr(img))

def extract_text_from_image(img):
    return cn_ocr.ocr(img)

def extract_text_from_file(file_path):
    image = imread(file_path)
    return extract_text_from_image(image)