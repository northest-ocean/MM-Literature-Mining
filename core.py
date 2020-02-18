# -*- coding: utf-8 -*-
# Developer : Nikola Liu
# Date		: 2020.02.18
# Filename	: core.py
# Tool		: Visual Studio Code

import cv2
from tqdm import tqdm
from UnionFind import UnionFind
from utils import line_segmentation
from cnocr import CnOcr
def generate_result():
    index = 0
    # TODO(NikolaLiu@icloud.com): Adding function about extract each block and related image



def extract_text_from_image(path=None):
    """ CRNN to recognize text"""
    ocr = CnOcr()
    reses = ocr.ocr(path)
    text = ""
    for res in reses:
        text += ''.join(res)
    return text

def get_all_seg_areas(image):
    # TODO(NikolaLiu@icloud.com): As for a segmentation result, we can scan it to get each areas.\
    assert len(image.shape) == 2 or image.shape[2] == 1, "The input should be a binary image..."

    # directions = [(0, 1), (1, 0), (1, 1)]
    # DFS Method, StackOverflow !!!
    # def render_image(x, y):
    #     # print("dealing with %d %d" % (x, y))
    #     image[x][y] == 0
    #     for i, j in directions:
    #         if x + i < image.shape[0] and y + j < image.shape[1] and image[x+i][y+j] == 255:
    #             render_image(x+i, y+j)

    # def get_index(x, y):return x*image.shape[1]+y
    # background_count = 0
    # # IDEA: Optimized UnionFind Method, avoid stackoverflow by removing recursion. [2020.02.18]
    # uf = UnionFind(image.shape[0] * image.shape[1])
    # for i in tqdm(range(image.shape[0])):
    #     for j in range(image.shape[1]):
    #         if image[i][j] == 255:
    #             # for direction in directions:
    #             if i+1 < image.shape[0]  and image[i+1][j] == 255: #and new_y < col  is true 优化---------------
    #                 uf.union(get_index(i, j), get_index(i+1, j))
    #             if  j+1 < image.shape[1] and image[i][j+1] == 255:
    #                 uf.union(get_index(i, j), get_index(i, j+1))
    #             if  j+1 < image.shape[1] and i+1 < image.shape[0] and image[i+1][j+1] == 255:
    #                 uf.union(get_index(i, j), get_index(i+1, j+1))
    #         else:
    #             background_count += 1

    nccomps = cv2.connectedComponentsWithStats(image)
    return nccomps



def process_area(images, x, y):
    # TODO(NikolaLiu@icloud.com): Scan column and row respectively, then get the split possibility. You should also consider the situation that hollows in an area could also be the sign of splitting. 
    pass


if __name__ == "__main__":
    seg_image = cv2.imread('/root/Projects/GraduationDesign/result0/Material28_images_96.jpg')
    ori_image = cv2.imread('/root/Projects/GraduationDesign/imgs/Material28_images_96.jpg')

    B, G, R = cv2.split(seg_image)
    _,RedThresh = cv2.threshold(R,50,255,cv2.THRESH_BINARY)
    #OpenCV定义的结构矩形元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    eroded = cv2.erode(RedThresh,kernel)
    for i in range(10):
        eroded = cv2.erode(eroded,kernel)
    origin = cv2.dilate(eroded,kernel)
    for i in range(10):
        origin = cv2.dilate(origin,kernel)
    # cv2.imwrite('./x.png', get_all_seg_areas(origin)[1] * 100)
    blocks = get_all_seg_areas(origin)[2]
    print(blocks)
    for index, block in enumerate(blocks[1:]):
        # cv2.imwrite('./figure' + str(index) + '.png', 
        # ori_image[block[1]:block[1]+block[3], block[0]:block[0]+block[2]])
        if block[2] >= seg_image.shape[0] // 2:
            print("Single Page mode.")
        else:
            print("Cannot define Page mode, default mode is single page mode.")
        # Here the coordinate is managed (y, x)
        left_up = (block[1], block[0])
        right_down = (block[1]+block[3], block[0]+block[2])
        center_point = (block[1]+block[3]//2, block[0]+block[2]//2)
        tag = ""
        if abs(seg_image[center_point[0]][center_point[1]][0] - 156) <= 3:
            tag = "table"
            print("Table found!")
            scan_area = ori_image[:block[1]]
        elif abs(seg_image[center_point[0]][center_point[1]][0] - 133) <= 3:
            tag = "figure"
            print("Figure found!")
            scan_area = ori_image[block[1]+block[3]:]
        
        line_segs = line_segmentation(scan_area,flag=1 ,tag=tag)[0]
        # cv2.imwrite('./lineseg.png', scan_area)
        print(extract_text_from_image(scan_area[line_segs[0]:line_segs[1]]))

