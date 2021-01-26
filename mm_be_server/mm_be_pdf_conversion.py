import fitz
import cv2
import os
import asyncio
import time

from multiprocessing import Process, cpu_count

async def pdf2img_worker(pdf_path, image_path, page, idx):
    rotate = int(0)
    zoom_x = 1.6666666
    zoom_y = 1.6666666
    mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
    pix = page.getPixmap(matrix=mat, alpha=False)
    if not os.path.exists(image_path):
        os.makedirs(image_path, exist_ok=True)
    pix.writePNG( image_path + "/{}.png".format(idx))


def pdf2img(pdf_path, image_path="./images"):
    start = time.time()
    pdf_doc = fitz.open(pdf_path)
    pages = []
    for pg in range(pdf_doc.pageCount):
        pages.append(pdf_doc[pg])
    loop = asyncio.get_event_loop()
    tasks = [pdf2img_worker(pdf_path, image_path, page, idx) for idx, page in enumerate(pages)]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
    print("Time consumed: {}".format(time.time()-start))


def scale_image_worker(image_list):
    for in_path, out_path, target_size in image_list:
        img = cv2.imread(in_path)
        cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path, img)

def scale_image(input_path, output_path, target_size):
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

if __name__ == "__main__":
    pdf2img("/home/SENSETIME/liumengyang/GT/Papers/2004.14723.pdf")
    scale_image("./images", './images_scaled', (960, 704))