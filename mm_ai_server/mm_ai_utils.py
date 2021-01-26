import os
import sys
import traceback

from keras_segmentation.predict import predict_multiple, predict
from mm_core.mm_core_logger import logger
from mm_core.mm_core_utils import check_model_file

def get_single_segmentation(file_path):
    logger.info("IN get_single_segmentation")
    if not os.path.isfile("./.cached_masks/" + file_path.split("/")[-1]):
        logger.warn("No cached mask for current data, fetching mask from model...")
        check_model_file()
        try:
            predict(
                checkpoints_path="./.models/resnet_segnet_1",
                inp=file_path,
                out_fname="./.cached_masks/" + file_path.split("/")[-1]
            )
            logger.info("Saving mask in {}".format("./.cached_masks/" + file_path.split("/")[-1]))
        except Exception as e:
            logger.error("Cannot fetch mask from model, error reason: " + e.__str__() + traceback.format_exc())
            return -1
    else:
        logger.warn("Found cached mask for current data, using cached mask...")
    logger.info("OUT get_single_segmentation")
    return 0


def get_segmentations(input_dir):
    logger.info("IN get_segmentations")
    check_model_file()
    try:
        predict_multiple(
            inp_dir=input_dir,
            out_dir="./.cached_masks",
            checkpoints_path="./.models/resnet_segnet_1"
        )
        logger.info("Segmentation Completed!")
    except Exception as e:
        logger.error("Exception occured while fetching mask for data folder, error reason: " + e.__str__())
        return -1
    logger.info("OUT get_segmentations")
    return 0

def get_segmented_image():
    paths = os.listdir("./.cached_masks")
    for path in paths:
        if "png" in path or "jpg" in path:
            seg_result = cv2.imread(os.path.join("./.cached_masks", path))
            original_image= cv2.imread(os.path.join("./.images", path))
            if seg_result.shape != original_image.shape:
                logger.error("Shape of segmentation result is not identical with original image")
            generate_result(seg_result, original)
        else:
            print("File format Error: " + path)

if __name__ == "__main__":
    check_model_file()