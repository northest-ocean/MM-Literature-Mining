import fitz
import cv2

def pdf2img(pdf_path, image_path="./images"):
    """ Convert PDF pages to PNG images.
    Args:
        pdf_path: Path of PDF file
        image_path: Path of target images
    Returns:
        None
    """
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
    file_list = os.listdir(input_path)
    for file_name in file_list:
        in_path = input_path + "/" + file_name
        img = cv2.imread(in_path)
        cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        out_path = output_path + "/" + file_name
        cv2.imwrite(out_path, img)