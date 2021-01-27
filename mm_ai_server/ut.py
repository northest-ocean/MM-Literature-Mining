from mm_ai_get_mask import get_mask_for_single_image
from mm_ai_ocr import extract_text_from_file, extract_text_from_image

import os
import mxnet as mx
from cnstd import CnStd
import cv2


if __name__ == "__main__":
    FILE_PATH = "/home/SENSETIME/liumengyang/GT/MM-Literature-Mining/mm_be_server/images_scaled/2.png"
    get_mask_for_single_image(FILE_PATH)
    # std = CnStd()
    # # img_fp = 'examples/taobao.jpg'
    # img = mx.image.imread(FILE_PATH, 1)
    # box_info_list = std.detect(img)
    # for box in box_info_list:
    #     cropped_img = box['cropped_img']
    #     print(extract_text_from_image(cropped_img))
