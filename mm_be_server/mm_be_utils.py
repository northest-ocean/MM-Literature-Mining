
import cv2
import base64
import asyncio
import os
import json
import numpy as np
from tqdm import tqdm

# from mm_core.mm_core_utils import check_required_folder_file_for_save, get_all_seg_areas, remove_adhesion_area
# from mm_core.mm_core_logger import logger
# from mm_ai_sever.mm_ai_ocr import extract_text_from_image
from mm_be_pdf_conversion import get_page_image_pair

# async def async_send_ocr_request(code):
#     await send_request(code)


# def get_ocr_result_from_ai_server(image_list):
#     loop = asyncio.get_event_loop()
#     encoded_list = []
#     for image in image_list:
#         encoded_list.append(image_to_base64(image))
#     tasks = [async_send_ocr_request(code) for code in encoded_list]

    

# def save(title=None, block=None, file_name=None, tag=None, ori_image=None):
#         """save detected area and save log"""
#         check_required_folder_file_for_save(file_name)
#         # Tag should be determined, otherwise the model may not work properly.
#         assert tag is not None, "Tag cannot be None, please check your data or issue on GitHub."
#         save_name = ""
#         if title is None:
#             title = "This " + tag + " cannot find subscript."
#         else:
#             for i in range(title.__len__()):
#                 start = 1
#                 if i == 0:
#                     if "Table" in title[0:5]:
#                         i = 5
#                         start = 5
#                     if "Fig" in title[0:5]:
#                         i = 3
#                         start = 3
#                     if "Figure" in title[0:6]:
#                         i = 6
#                         start = 6
#                     if "图" in title[0:2] or "表" in title[0:2]:
#                         i = 1
#                         start = 1
#                 title = title.replace("<space>", " ")
#                 if title[i] not in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", " ","]", ")","-"):
#                     # Normally we won"t see Table1.100
#                     if i - start > 5:
#                         i = start + 4
#                     save_name = title[:i]
#                     break
#         if save_name is None or len(save_name) == 0:
#             save_name = str(random.randint(100,1000))
#         cv2.imwrite("./results/{}/{}.jpg".format(file_name, save_name), ori_image[block[1]:block[1]+block[3], block[0]:block[0]+block[2]])
#         with open("./results/{}/log.txt".format(file_name), "a") as f:
#             f.writelines(title + " " + file_name + "/" + save_name + ".jpg\n")



# def generate_result(seg_image, ori_image, erode_iter=10, kernel=None):

#     # Forbidden passing image variables, only receive image path as string
#     assert type(seg_image) == str and type(ori_image) == str, "seg_image and ori_image should be strings of image path."
#     file_name = seg_image.split("/")[-1].split("_")[0]
#     seg_image = cv2.imread(seg_image)
#     ori_image = cv2.imread(ori_image)
#     assert len(seg_image.shape) == 3 and seg_image.shape[2] == 3, "Channel number of Segmentation result should be 3"
    

#     # Remove adhesion area using morphology methods
#     seg_image_binary = remove_adhesion_area(seg_image, erode_iter, kernel)

#     # Getting all blocks in binary segmentation image
#     blocks = get_all_seg_areas(seg_image_binary)[2]

#     # The first block is the background, so start from 1
#     for index, block in enumerate(blocks[1:]):
#         # Here the coordinate is managed (y, x)
#         left_up = (block[1], block[0])
#         right_down = (block[1]+block[3], block[0]+block[2])

#         # Use the center of area to determine tag and page type
#         center_point = (block[1]+block[3]//2, block[0]+block[2]//2)

#         mode = "single"
#         if block[2] >= seg_image.shape[0] // 2:
#             print("Single Page mode.")
#         else:
#             if abs(center_point[1] - seg_image.shape[1] // 2) < min(abs(center_point[1] -seg_image.shape[1] // 4), abs(center_point[1] - (seg_image.shape[1] * 3 //4))):
#                 print("Single Page mode.")
#             else:
#                 if abs(center_point[1] - seg_image.shape[1] // 4) < abs(center_point[1] - (seg_image.shape[1] * 3 //4)):
#                     mode = "double_left"
#                 else:
#                     mode = "double_right"
#                 print("Double page mode: " + mode)

#             # print("Cannot define Page mode, default mode is single page mode.")
        
#         tag = ""
#         if abs(seg_image[center_point[0]][center_point[1]][0] - 156) <= 3:
#             tag = "table"
#             logger.info("Table figure !")
#             if mode == "single":
#                 scan_area = ori_image[:block[1]]
#             else:
#                 if mode == "double_left":
#                     scan_area = ori_image[:block[1], 0:ori_image.shape[1] // 2]
#                 elif mode == "double_right":
#                     scan_area = ori_image[:block[1], ori_image.shape[1] // 2:]
#                 else:
#                     raise TypeError("Invalid mode!")
#         elif abs(seg_image[center_point[0]][center_point[1]][0] - 133) <= 3:
#             tag = "figure"
#             logger.info("Found figure !")
#             if mode == "single":
#                 scan_area = ori_image[block[1]+block[3]:]
#             else:
#                 if mode == "double_left":
#                     scan_area = ori_image[block[1]+block[3]:, 0:ori_image.shape[1] // 2]
#                 elif mode == "double_right":
#                     scan_area = ori_image[block[1]+block[3]:, ori_image.shape[1] // 2:]
#                 else:
#                     raise TypeError("Invalid mode!")
#         else:
#             logger.warn("The category cannot be determined, the extraction can be bad.")

#         # Get all possible areas that include title
#         line_segs = line_segmentation(scan_area,flag=1 ,tag=tag)

#         # If the category is table, the return sequence of areas should be reversed
#         if tag == "table":
#             line_segs.reverse()
            
#         # Extract text by lines, this could improve the accuracy, proved by test.
#         line_seg_areas = []
        
#         if line_segs is None:
#             raise RuntimeError("NO segmentation found!")
#         for line_seg in line_segs:
#             line_seg_areas.append(scan_area[line_seg[0]:line_seg[1]])
#         # title = extract_text_from_image(scan_area[line_segs[0][0]:line_segs[-1][-1]])
#         title = get_ocr_result_from_ai_server(line_seg_areas)

#         if title[0] not in "图表":
#             logger.warn("First Dection Failed! Reconstruting the detection area...")
#             new_area = reconstruct_area(line_segs, ori_image, block, tag, mode)
#             title = extract_text_from_image(new_area)
        # save(title, block, file_name, tag, ori_image)

def image_to_base64(image):
    return base64.b64encode(image)

def base64_to_image(base64_code):
    return base64.b64decode(base64_code)

def fetch_dataset():
    dataset_json = dict()
    os.makedirs("./dataset", exist_ok=True)
    with open("../urls.txt", 'r') as f:
        for line in f.readlines():
            tag, url = line.split(" ")
            abs_path_for_file = os.path.join(os.getcwd(), "dataset/{}".format(url.split("/")[-1][:-1]))
            dataset_json[tag] = abs_path_for_file
            # print(abs_path_for_file)
            if os.path.exists(abs_path_for_file):
                print("Skipping {}".format(abs_path_for_file))
                continue
            # print(url)
            # os.system("wget -P./dataset {}".format(url))
            
    with open("./dataset.json", 'w') as f:
        json.dump(dataset_json, f)


def build_training_dataset(augmentation=False):
    DATASET_JSON = "./dataset.json"
    ANNOTATION_JSON = "./annotations.json"
    dataset_obj = None
    with open(DATASET_JSON, 'r') as f:
        dataset_obj = json.loads(f.read())
    
    annotations_obj = None
    with open(ANNOTATION_JSON, 'r') as f:
        annotations_obj = json.loads(f.read())
    
    if augmentation:
        os.makedirs("./aug_dataset/train_ori", exist_ok=True)
        os.makedirs("./aug_dataset/train_ann", exist_ok=True)
        os.makedirs("./aug_dataset/val_ori", exist_ok=True)
        os.makedirs("./aug_dataset/val_ann", exist_ok=True)
        index = 0
        for key, val in tqdm(dataset_obj.items()):
            page_seg_pairs = get_page_image_pair(val)
            anns = []
            for page, _, _ in page_seg_pairs:
                page_rect = list(map(lambda x:float(x), page.page_rect().as_tuple()))
                size = (page_rect[3] - page_rect[1], page_rect[2] - page_rect[0])
                size = tuple(map(lambda x:round(x), size))
                if size == (0, 0):
                    size = anns[-1].shape
                anns.append(np.random.randint(255, size=size))
            for i in range(len(anns)):
                anns[i][:, :] = 0
            images = list(map(lambda x:x[1], page_seg_pairs))
            resize_flag = False
            for fig in annotations_obj[key]["figures"]:
                region_bb = list(map(lambda x:int(x), fig["region_bb"]))
                page = fig["page"] - 1
                figure_type = fig["figure_type"]
                page_height = round(float(fig["page_height"]))
                page_width = round(float(fig["page_width"]))
                if not resize_flag:
                    for i in range(len(anns)):
                        anns[i] = np.resize(anns[page], (page_height, page_width))
                    resize_flag = True
                # print(anns[page].shape)
                anns[page][region_bb[1]:region_bb[3], region_bb[0]:region_bb[2]] = 1 if figure_type == "Table" else 2
                # print(anns[page][region_bb[1]:region_bb[3], region_bb[0]:region_bb[2]])
            for ann, image in zip(anns, images):
                if ann.shape[0] != image.shape[0] or ann.shape[1] != image.shape[1]:
                    print("Warning: ", ann.shape, image.shape)
                    image = cv2.resize(image, ann.shape[::-1], interpolation=cv2.INTER_AREA)
                    # raise ValueError("Fuck")
                cv2.imwrite("./aug_dataset/train_ann/{}.png".format(index), ann)
                cv2.imwrite("./aug_dataset/train_ori/{}.png".format(index), image)
                index += 1

    



if __name__ == "__main__":
    build_training_dataset(True)