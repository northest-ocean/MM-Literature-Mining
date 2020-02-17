# PDF Figure and Table Extraction (Graduate Design)

## This project is based on project [CNOCR](https://github.com/breezedeus/cnocr) and [Keras_segmentation](https://github.com/divamgupta/image-segmentation-keras). Thanks for their excellent work!

## Fundamentals
1. Using ResNet50_Segnet to detect Image and Table area in a PDF file, especially for scientific papers
2. Using CNOCR which is based on CRNN to extract tags and related information. (More NLP work is required to further this project)
3. Self-developed algorithm to deal with the adhesion areas (Developing ...)

## Getting Started

### Installing
```
git clone https://github.com/Alpha-Monocerotis/PDF_FigureTable_Extraction
pip install -r requirements.txt
```

### Download the pretrained model (Currently only support Resnet50-Segnet)

If you cannot have a good connection, you can download it by yourself in repo's release or use wget(Linux).

```
wget -p ./models https://github.com/Alpha-Monocerotis/PDF_FigureTable_Extraction/releases/download/v1.0/resnet_segnet_1.0
```