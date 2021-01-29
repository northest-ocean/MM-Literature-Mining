import cv2
import numpy as np
import os
import json
import asyncio
import time
import poppler
import random

from pdf2image import convert_from_path
from multiprocessing import Process, cpu_count

# async def pdf2img_worker(pdf_path, image_path, page, idx):
#     rotate = int(0)
#     zoom_x = 1.6666666
#     zoom_y = 1.6666666
#     mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
#     pix = page.getPixmap(matrix=mat, alpha=False)
#     if not os.path.exists(image_path):
#         os.makedirs(image_path, exist_ok=True)
#     pix.writePNG( image_path + "/{}.png".format(idx))


# def pdf2img(pdf_path, image_path="./images"):
#     start = time.time()
#     pdf_doc = fitz.open(pdf_path)
#     pages = []
#     for pg in range(pdf_doc.pageCount):
#         pages.append(pdf_doc[pg])
#     loop = asyncio.get_event_loop()
#     tasks = [pdf2img_worker(pdf_path, image_path, page, idx) for idx, page in enumerate(pages)]
#     loop.run_until_complete(asyncio.wait(tasks))
#     loop.close()
#     print("Time consumed: {}".format(time.time()-start))

def pdf2image(input_path, output_path='./images', size=(612, 792)):
    file_name = output_path.split('/')[-1][:-4]
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    convert_from_path(input_path, 79, output_path, fmt="PNG", output_file=file_name, thread_count=cpu_count(), size=size)


def scale_image_worker(image_list):
    for in_path, out_path, target_size in image_list:
        img = cv2.imread(in_path)
        cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path, img)

def scale_image(input_path, output_path, target_size):
    assert len(target_size) == 2, TypeError(
        "Length of target_size should be 2")
    os.makedirs('./images_scaled', exist_ok=True)
    file_list = os.listdir(input_path)
    dispatch_list = {i:[] for i in range(cpu_count())}
    for idx, file_name in enumerate(file_list):
        in_path = input_path + "/" + file_name
        out_path = output_path + "/" + file_name
        dispatch_list[idx % cpu_count()].append([in_path, out_path, target_size])
    process_pool = [Process(target=scale_image_worker, args=(dispatch_list[i], )) for i in range(cpu_count())]
    for process in process_pool:
        process.start()
    for process in process_pool:
        process.join()

def get_binary(src, threshold, maxVal):
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src[i, j] = maxVal if src[i, j] > threshold else 0
    return src

def remove_text_area(page, image, bin=False):
    image_array = np.array(image)
    for text_box in page.text_list():
        # print(text_box.text, )
        text_scope_x_start, text_scope_y_start, text_scope_x_end, text_scope_y_end = map(lambda x:int(x), text_box.bbox.as_tuple())
        image_array[text_scope_y_start:text_scope_y_end, text_scope_x_start:text_scope_x_end] = 0
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            gray_image[i, j] = 255 - gray_image[i, j]

    # cv2.imwrite("./x.png", gray_image)
    # gray_image = cv2.imread("./x.png", cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite("./gray.png", gray_image)
    kernel = np.ones((2, 2), dtype=np.int8)
    _image = gray_image
    _image = cv2.dilate(gray_image,kernel,iterations=2)
    _image = cv2.erode(_image,kernel,iterations=2)

    return _image, np.array(image)


def get_page_image_pair(input_path):
    filename = input_path.split("/")[-1][:-4]
    document = poppler.load(input_path)
    page_images = convert_from_path(input_path, dpi=300)
    page_image_pair = []
    for idx, page_image in enumerate(page_images):
        page_doc = document.create_page(idx)
        page_rect = list(map(lambda x:round(x), page_doc.page_rect().as_tuple()))
        size = (page_rect[2] - page_rect[0], page_rect[3] - page_rect[1])
        size = tuple(map(lambda x:int(x), size))
        # page_image = page_image.resize(size)
        # text_seg_image, original_image = remove_text_area(page_doc, page_image)
        text_seg_image, original_image = np.array(page_image), np.array(page_image)
        page_image_pair.append((page_doc, original_image, text_seg_image))
    return page_image_pair


def load_dataset_images(dataset_json):
    dataset_obj = None
    annotations_obj = None
    with open("./dataset.json", 'r') as f:
        dataset_obj = json.loads(f.read())
    with open("./annotations.json", 'r') as f:
        annotations_obj = json.loads(f.read())
    
    if dataset_obj is None or annotations_obj is None:
        return
    for key, val in dataset_obj.items():
        # print(annotations_obj[key], key)
        if annotations_obj[key]["figures"]:
            pdf2image(val, "./images/{}".format(key), (annotations_obj[key]["figures"][0]["page_width"], annotations_obj[key]["figures"][0]["page_height"]))
        else:
            pdf2image(val, "./images/{}".format(key))

if __name__ == "__main__":
    pass