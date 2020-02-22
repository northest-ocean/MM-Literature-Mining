# -*- coding: utf-8 -*-
# Developer : Nikola Liu
# Date		: 2020.02.17
# Filename	: tools.py
# Tool		: Visual Studio Code

import fitz
import os
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras_segmentation.predict import predict_multiple, predict


def pdf2img(pdf_path, image_path="./images"):
    """ Convert PDF pages to PNG images.
    Args:
        pdf_path: Path of PDF file
        image_path: Path of target images

    Returns:
        None
    """

    start_time = datetime.datetime.now()

    print("Your images will be saved in directory: " + image_path)
    pdf_doc = fitz.open(pdf_path)
    for pg in range(pdf_doc.pageCount):
        page = pdf_doc[pg]
        rotate = int(0)
        zoom_x = 1.6666666
        zoom_y = 1.6666666
        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pix = page.getPixmap(matrix=mat, alpha=False)

        if not os.path.exists(image_path):
            os.makedirs(image_path)
        
        pix.writePNG(
            image_path +
            "/" +
            pdf_path.split(".")[1].split("/")[2] +
            "_" +
            "images_%s.jpg" %
            pg)
    end_time = datetime.datetime.now()


def convert(input_path, output_path, target_size):
    """ Convert images to a scaled size(target_size).
    Args:
        input_path: Path of input image
        output_path: Path of resize image
        target_size: Size of target imag

    Returns:
        None

    """
    assert len(target_size) == 2, TypeError(
        "Length of target_size should be 2")
    file_list = os.listdir(input_path)
    for file_name in file_list:
        in_path = input_path + "/" + file_name
        img = cv2.imread(in_path)
        cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        out_path = output_path + "/" + file_name
        cv2.imwrite(out_path, img)



# def show_seg_result(original, seg_result):
#     cv2.imshow("name", seg_result)
#     cv2.waitKey(0)

#     for i in range(original.shape[0]):
#         for j in range(original.shape[1]):
#             if seg_result[i][j][2] == 20:
#                 original[i][j] = 0
#     return original


# def show_dir_seg_result(original_dir, seg_result_dir):
#     seg_results_paths = os.listdir(seg_result_dir)
#     for path in seg_results_paths:
#         seg_path = os.path.join(seg_result_dir, path)
#         original_path = os.path.join(original_dir, path)
#         original = cv2.imread(original_path)
#         seg_result = cv2.imread(seg_path)
#         image = show_seg_result(original, seg_result)
#         cv2.imwrite("./" + path, image)


def line_segmentation(img, flag=1, tag="figure"):
    assert tag in ["figure", "table"], "Invalid tag, tag should be figure or table."
    start_row = 0
    segments = []

    def determine_white_line(row):
        pixel_sum = 0
        for i in range(img.shape[1]):
            if img[row][i][0] >= 230 and img[row][i][1] >= 230 and img[row][i][2] >= 230:
                pass
            elif img[row][i][0] <= 130 or img[row][i][1] <= 130 or img[row][i][2] <= 130:
                return False
            else:
                pixel_sum += 1
            if pixel_sum / img.shape[1] > 0.03:
                return False
        return True

    # The scan_scope must start with white line, if not, it should move to a white line start location.
    if tag == "figure":
        scan_scope = range(img.shape[0])
        if not determine_white_line(0):
            for i in range(img.shape[0]):
                if not determine_white_line(i):
                    continue
                else:
                    scan_scope = range(i, img.shape[0])
                    break
    else:
        scan_scope = range(img.shape[0]-1, -1, -1)
        if not determine_white_line(img.shape[0]-1):
            for i in range(img.shape[0]-1, -1, -1):
                if not determine_white_line(i):
                    continue
                else:
                    scan_scope = range(i, -1, -1)
                    break
    single_line_width = -1
    explore_range = None
    
    for i in scan_scope:
        if determine_white_line(i) and flag == 1:
            continue
        elif determine_white_line(i) and flag == 0:
            flag = 1 
            if tag == "figure":
                segments.append([start_row - 1, i + 1])
                single_line_width = int((i - start_row) * 1.2)
                if single_line_width != -1:
                    explore_range = range(i, i+single_line_width+1)
            else:
                segments.append([i - 1, start_row + 1])
                single_line_width = int((start_row - i) * 1.2)
                if single_line_width != -1:
                    explore_range = range(i, i-single_line_width-1, -1)
        elif not determine_white_line(i) and flag == 1:
            flag = 0
            start_row = i
        else:
            continue

        if explore_range is not None:
            for j in explore_range:
                if not determine_white_line(j):
                    break
                if j == explore_range[-1]:
                    return segments
            explore_range = None
        if single_line_width >= 25:
            return segments
        if segments.__len__() >= 6:
            return segments
    return segments


# def log_line_segmentation(img, segments, output_dir):
#     for index, segment in enumerate(segments):
#         cv2.imwrite(output_dir + "/seg" + str(index) + ".jpg",
#                     img[segment[0]: segment[1], :, :])


def check_model_file():
    model = "./models/resnet_segnet_1"
    if not os.path.isfile(model + ".0"):
        try:
            # download model from github
            os.system("wget -p ./models https://github.com/Alpha-Monocerotis/PDF_FigureTable_Extraction/releases/download/v1.0/resnet_segnet_1.0")
        except Exception as e:
            # print(e)
            print("Download is not completed, please try again later.")
            raise RuntimeError("Model not ready")
    print("Loading model from downloaded weight...")

def get_single_segmentation(file_path):
    if not os.path.isfile("./output_seg/" + file_path.split("/")[-1]):
        check_model_file()
        try:
            predict(
                checkpoints_path="./models/resnet_segnet_1",
                inp=file_path,
                out_fname="./output_seg/" + file_path.split("/")[-1]
            )
        except Exception as e:
            print(e)
            raise RuntimeError("Cannot finish segmentation.")
    else:
        print("Using last segmentation result.")


def get_segmentations(input_dir):
    check_model_file()
    try:
        predict_multiple(
            inp_dir=input_dir,
            out_dir="./output_seg",
            checkpoints_path="./models/resnet_segnet_1"
        )
        print("Segmentation Completed!")
    except Exception as e:
        print("Cannot finish segmentation, please check the Exception below!")
        print(e)

def get_segmented_image():
    paths = os.listdir("./output_seg")
    for path in paths:
        if "png" in path or "jpg" in path:
            seg_result = cv2.imread(os.path.join("./output_seg", path))
            original_image= cv2.imread(os.path.join("./images", path))
            assert seg_result.shape == original_image.shape, "Shape of segmentation result is not compatible with original image"
            generate_result(seg_result, original)
        else:
            print("File format Error: " + path)


if __name__ == "__main__":
    pass