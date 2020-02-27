# -*- coding: utf-8 -*-
# Developer : Nikola Liu
# Date		: 2020.02.18
# Filename	: core.py
# Tool		: Visual Studio Code

import cv2
import os
from tqdm import tqdm
from utils import line_segmentation
from cnocr import CnOcr
import random


def remove_adhesion_area(seg_image, erode_iter=10, kernel=None):
    """ Remove adhension areas with erode and dialate method, if the adhesion area, the  erode_iter should be set higher to seperate different areas. You can also custom a kernel, which can be more effective.
    """
    B, G, R = cv2.split(seg_image)
    # Here we only determine background or block, blocks include tables and figures
    _,RedThresh = cv2.threshold(R,50,255,cv2.THRESH_BINARY)
    
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    eroded = cv2.erode(RedThresh,kernel)
    for i in range(erode_iter):
        eroded = cv2.erode(eroded,kernel)
    origin = cv2.dilate(eroded,kernel)
    for i in range(erode_iter):
        origin = cv2.dilate(origin,kernel)
    cv2.imwrite("x.png", origin)
    return origin


def reconstruct_area(line_segs, ori_image, block, tag, mode):
    """Reconstruct text extraction area"""
    # WARNING Potential Bug need fixed
    if tag == "table":
        split_y = block[1]
    else:
        split_y = block[1] + block[3]
    if mode == "double_left":
        base_area = ori_image[:, 0:ori_image.shape[1] // 2]
        up_line_segs = line_segmentation(ori_image[0:split_y, 0:ori_image.shape[1] // 2], tag="table")
        down_line_segs = line_segmentation(ori_image[split_y:, 0:ori_image.shape[1] // 2])
    elif mode == "double_right":
        base_area = ori_image[:, ori_image.shape[1] // 2:]
        up_line_segs = line_segmentation(ori_image[0:split_y, ori_image.shape[1] // 2:], tag="table")
        down_line_segs = line_segmentation(ori_image[split_y:, ori_image.shape[1] // 2:])
    elif mode == "single":
        base_area = ori_image[:, :]
        up_line_segs = line_segmentation(ori_image[0:split_y,:], tag="table")
        down_line_segs = line_segmentation(ori_image[split_y:,:])
    if tag == "table":
        up_line_segs.reverse()
    print(mode)
    start_y = split_y
    end_y = split_y
    for i in range(len(up_line_segs)):
        if len(up_line_segs[i]) != 0:
            start_y -= up_line_segs[i][0]
            break
    for i in range(len(down_line_segs)-1, -1, -1):
        if len(down_line_segs[i]) != 0:
            end_y += down_line_segs[i][1]
            break
    # print(split_y)
    # print(up_line_segs)
    # print(start_y, end_y)

    lines = line_segmentation(base_area[start_y:end_y], tag="table")
    lines.reverse()
    for index, line in enumerate(lines):
        if len(line) == 0 or line[1] - line[0] >= 25:
            lines.pop(index)
    # print
    return base_area[start_y:end_y][lines[0][0]:lines[-1][-1]]


def check_required_folder_file(file_name):
    # Check required folders and files
    try:
        if not os.path.isdir("./results"):
            os.system("mkdir results")
            print("Building directory: results")
        if not os.path.isdir("./results/" + file_name):
            os.system("mkdir results/" + file_name)
            print("Building directory: " + file_name)
        if not os.path.isfile("./log.txt"):
            os.system("touch log.txt")
            print("Building log: log.txt")
    except Exception as e:
        raise RuntimeError("Cannot secure required folder and file.")

def generate_result(seg_image, ori_image, erode_iter=10, kernel=None):

    # Forbidden passing image variables, only receive image path as string
    assert type(seg_image) == str and type(ori_image) == str, "seg_image and ori_image should be strings of image path."
    file_name = seg_image.split("/")[-1].split("_")[0]
    seg_image = cv2.imread(seg_image)
    ori_image = cv2.imread(ori_image)
    assert len(seg_image.shape) == 3 and seg_image.shape[2] == 3, "Channel number of Segmentation result should be 3"

    def save(title=None, block=None, file_name=None, tag=None):
        """save detected area and save log"""
        check_required_folder_file(file_name)
        # Tag should be determined, otherwise the model may not work properly.
        assert tag is not None, "Tag cannot be None, please check your data or issue on GitHub."
        save_name = ""
        if title is None:
            title = "This " + tag + " cannot find subscript."
        else:
            for i in range(title.__len__()):
                start = 1
                if i == 0:
                    if "Table" in title[0:5]:
                        i = 5
                        start = 5
                    if "Fig" in title[0:5]:
                        i = 3
                        start = 3
                    if "Figure" in title[0:6]:
                        i = 6
                        start = 6
                    if "图" in title[0:2] or "表" in title[0:2]:
                        i = 1
                        start = 1
                title = title.replace("<space>", " ")
                if title[i] not in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", " ","]", ")","-"):
                    # Normally we won"t see Table1.100
                    if i - start > 5:
                        i = start + 4
                    save_name = title[:i]
                    break
        if save_name is None or len(save_name) == 0:
            save_name = str(random.randint(100,1000))
        cv2.imwrite("./results/" + file_name + "/" + save_name  + ".jpg", ori_image[block[1]:block[1]+block[3], block[0]:block[0]+block[2]])
        with open("log.txt", "a") as f:
            f.writelines(title + " " + file_name + "/" + save_name + ".jpg\n")
    

    # Remove adhesion area using morphology methods
    seg_image_binary = remove_adhesion_area(seg_image, erode_iter, kernel)

    # Getting all blocks in binary segmentation image
    blocks = get_all_seg_areas(seg_image_binary)[2]

    # The first block is the background, so start from 1
    for index, block in enumerate(blocks[1:]):
        # Here the coordinate is managed (y, x)
        left_up = (block[1], block[0])
        right_down = (block[1]+block[3], block[0]+block[2])

        # Use the center of area to determine tag and page type
        center_point = (block[1]+block[3]//2, block[0]+block[2]//2)

        mode = "single"
        if block[2] >= seg_image.shape[0] // 2:
            print("Single Page mode.")
        else:
            if abs(center_point[1] - seg_image.shape[1] // 2) < min(abs(center_point[1] -seg_image.shape[1] // 4), abs(center_point[1] - (seg_image.shape[1] * 3 //4))):
                print("Single Page mode.")
            else:
                if abs(center_point[1] - seg_image.shape[1] // 4) < abs(center_point[1] - (seg_image.shape[1] * 3 //4)):
                    mode = "double_left"
                else:
                    mode = "double_right"
                print("Double page mode: " + mode)

            # print("Cannot define Page mode, default mode is single page mode.")
        
        tag = ""
        if abs(seg_image[center_point[0]][center_point[1]][0] - 156) <= 3:
            tag = "table"
            print("Table found!")
            if mode == "single":
                scan_area = ori_image[:block[1]]
            else:
                if mode == "double_left":
                    scan_area = ori_image[:block[1], 0:ori_image.shape[1] // 2]
                elif mode == "double_right":
                    scan_area = ori_image[:block[1], ori_image.shape[1] // 2:]
                else:
                    raise TypeError("Invalid mode!")
        elif abs(seg_image[center_point[0]][center_point[1]][0] - 133) <= 3:
            tag = "figure"
            print("Figure found!")
            if mode == "single":
                scan_area = ori_image[block[1]+block[3]:]
            else:
                if mode == "double_left":
                    scan_area = ori_image[block[1]+block[3]:, 0:ori_image.shape[1] // 2]
                elif mode == "double_right":
                    scan_area = ori_image[block[1]+block[3]:, ori_image.shape[1] // 2:]
                else:
                    raise TypeError("Invalid mode!")
        else:
            raise RuntimeWarning("The category cannot be determined, the extraction can be bad.")

        # Get all possible areas that include title
        line_segs = line_segmentation(scan_area,flag=1 ,tag=tag)

        # If the category is table, the return sequence of areas should be reversed
        if tag == "table":
            line_segs.reverse()
            
        # Extract text by lines, this could improve the accuracy, proved by test.
        line_seg_areas = []
        
        if line_segs is None:
            raise RuntimeError("NO segmentation found!")
        for line_seg in line_segs:
            line_seg_areas.append(scan_area[line_seg[0]:line_seg[1]])
        # cv2.imwrite('x.png', scan_area[line_segs[0][0]:line_segs[-1][-1]])
        title = extract_text_from_image(scan_area[line_segs[0][0]:line_segs[-1][-1]], mode="S")
        if title[0] not in "图表":
            print("\033[1;31mFirst Dection Failed! Reconstruting the detection area...\033[0m")
            new_area = reconstruct_area(line_segs, ori_image, block, tag, mode)
            title = extract_text_from_image(new_area, mode="S")
        save(title, block, file_name, tag)
        



def extract_text_from_image(path=None, mode="C"):
    """ CRNN to recognize text"""
    ocr = CnOcr()
    if mode == "C":
        reses = ocr.ocr_for_single_lines(path)
    elif mode == "S":
        reses = ocr.ocr(path)
    text = ""
    for index, res in enumerate(reses):
        if len(res) == 0 or res[0] not in ("表", "图"):
            pass
        else:
            reses = reses[index:]
            break
    print(reses)
    for res in reses:
        text += "".join(res) + "\n"
    return text

def get_all_seg_areas(image):
    """ Get all areas from binary segmentation result. """
    assert len(image.shape) == 2 or image.shape[2] == 1, "The input should be a binary image..."
    nccomps = cv2.connectedComponentsWithStats(image)
    return nccomps


if __name__ == "__main__":
    seg_image = "/root/Projects/GraduationDesign/result0/Material5_images_3.jpg"
    ori_image = "/root/Projects/GraduationDesign/imgs/Material5_images_3.jpg"
    generate_result(seg_image, ori_image)

    


