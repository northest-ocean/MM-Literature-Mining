from cnocr import CnOcr

cn_ocr = CnOcr()

def extract_text_from_image_list(img_list):
    for img in img_list:
        ocr_results.append(cn_ocr.ocr(img))

def extract_text_from_image(img):
    return cn_ocr.ocr(img)