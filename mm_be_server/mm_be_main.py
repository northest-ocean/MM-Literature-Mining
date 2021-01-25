from mm_be_common import COMMON_DEFINE
get_single_segmentation(file_path)
generate_result(seg_image="./output_seg/" + file_path.split("/")[-1], ori_image=file_path)