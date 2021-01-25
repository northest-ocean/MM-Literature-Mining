
import cv2

from mm_core.mm_core_utils import check_required_folder_file_for_save, get_all_seg_areas, remove_adhesion_area
from mm_core.mm_core_logger import logger
from mm_ai_sever.mm_ai_ocr import extract_text_from_image
def save(title=None, block=None, file_name=None, tag=None, ori_image):
        """save detected area and save log"""
        check_required_folder_file_for_save(file_name)
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
        cv2.imwrite("./results/{}/{}.jpg".format(file_name, save_name), ori_image[block[1]:block[1]+block[3], block[0]:block[0]+block[2]])
        with open("./results/{}/log.txt".format(file_name), "a") as f:
            f.writelines(title + " " + file_name + "/" + save_name + ".jpg\n")



def generate_result(seg_image, ori_image, erode_iter=10, kernel=None):

    # Forbidden passing image variables, only receive image path as string
    assert type(seg_image) == str and type(ori_image) == str, "seg_image and ori_image should be strings of image path."
    file_name = seg_image.split("/")[-1].split("_")[0]
    seg_image = cv2.imread(seg_image)
    ori_image = cv2.imread(ori_image)
    assert len(seg_image.shape) == 3 and seg_image.shape[2] == 3, "Channel number of Segmentation result should be 3"
    

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
            logger.info("Table figure !")
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
            logger.info("Found figure !")
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
            logger.warn("The category cannot be determined, the extraction can be bad.")

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
        title = extract_text_from_image(scan_area[line_segs[0][0]:line_segs[-1][-1]])
        if title[0] not in "图表":
            logger.warn("First Dection Failed! Reconstruting the detection area...")
            new_area = reconstruct_area(line_segs, ori_image, block, tag, mode)
            title = extract_text_from_image(new_area)
        save(title, block, file_name, tag, ori_image)