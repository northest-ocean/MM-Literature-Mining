# -*- coding: utf-8 -*-
# Developer : Nikola Liu
# Date		: 2020.02.18
# Filename	: extract.py
# Tool		: Visual Studio Code

import os
import cv2
import gc
from core import generate_result
from keras_segmentation.predict import predict
from utils import get_single_segmentation, get_segmentations


def extract_from_pdf(file_path):
    pass

def extract_single_image(file_path):
    assert os.path.isfile(file_path), "Your file path is not valid, cannot read image file."
    try:
        image = cv2.imread(file_path)
        assert len(image.shape) == 3 and image.shape[2] == 3, "Your input image is supposed to have 3 channels"
    except Exception as e:
        print("Cannot load image from your file_path, please check it.")
        print(e)

    get_single_segmentation(file_path)
    generate_result(seg_image="./output_seg/" + file_path.split("/")[-1], ori_image=file_path)
        

def extract_from_pdf_dir(input_dir):
    pass

def extract_from_image_dir(input_dir):
    assert os.path.isdir(input_dir), "Your directory path is not valid, cannot read file from it."
    get_segmentations(input_dir)
    for path in os.listdir(input_dir):
        if "jpg" in path:
            seg_image =  "./output_seg/" + path
            ori_image = input_dir + "/" + path
            generate_result(seg_image, ori_image)
    


if __name__ == "__main__":
    extract_single_image("./examples/Material5_images_3.jpg")

