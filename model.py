import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import pathlib
from pathlib import Path
import pandas as pd
from sartorius_dataset import SartoriusDataset


TRAIN_CSV = "train.csv"
TRAIN_PATH = "train"
TEST_PATH = "test"

BATCH_SIZE = 2

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

# Amount of epochs
NUM_EPOCHS = 8

localpath = pathlib.Path().resolve()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device} device")


df_train = pd.read_csv(f"{localpath}/{TRAIN_CSV}")

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (cells) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)