from keras_segmentation.models.segnet import resnet50_segnet
from keras_segmentation.predict import evaluate
model = resnet50_segnet(n_classes=3 , input_height=792 - 792 % 32, input_width=612 - 612 % 32)

model.train(
    train_images =  "/home/SENSETIME/liumengyang/GT/MM-Literature-Mining/mm_be_server/aug_dataset/train_ori",
    train_annotations = "/home/SENSETIME/liumengyang/GT/MM-Literature-Mining/mm_be_server/aug_dataset/train_ann",
    checkpoints_path = "./resnet50_segnet_1" , epochs=10, batch_size=1,
)

# print(evaluate(
#             inp_images_dir="/home/SENSETIME/liumengyang/GT/MM-Literature-Mining/mm_be_server/aug_dataset/val_ori", annotations_dir="/home/SENSETIME/liumengyang/GT/MM-Literature-Mining/mm_be_server/aug_dataset/val_ann",
#             checkpoints_path="./.models/resnet50_segnet_1"))