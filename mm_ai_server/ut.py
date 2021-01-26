from mm_ai_get_mask import get_mask_for_single_image
from mm_ai_ocr import extract_text_from_file

import os


if __name__ == "__main__":
    FILE_PATH = "/home/SENSETIME/liumengyang/GT/MM-Literature-Mining/mm_be_server/images/0.png"
    get_mask_for_single_image(FILE_PATH)
    print(extract_text_from_file(FILE_PATH))
