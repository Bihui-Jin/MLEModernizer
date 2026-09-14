# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.7391961317618616

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


class Config:
    DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
    TEST_CSV = os.path.join(
        DATA_ROOT, "sample_submission.csv"
    )  # Path to sample_submission.csv for test image IDs
    TEST_IMAGES_DIR = os.path.join(
        DATA_ROOT, "test_images"
    )  # Path to test_images folder

    BEST_MODEL_ASDA_PATH = os.path.join(
        "/kaggle/input/resnetasda2/pytorch/default/1", "resnet50_asda_best_model.pth"
    )

    IMAGE_SIZE = 384
    BATCH_SIZE_INFERENCE = 64  # Larger batch size for efficient inference
    NUM_CLASSES = 5  # Number of disease classes (0, 1, 2, 3, 4)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RESNET_FM_SIZES = {
        "layer1": (96, 96),
        "layer2": (48, 48),
        "layer3": (24, 24),
        "layer4": (12, 12),
    }


print(f"Using device: {Config.DEVICE}")
print(f"Loading model weights from: {Config.BEST_MODEL_ASDA_PATH}")
print(f"Loading test images from: {Config.TEST_IMAGES_DIR}")


class TestDataset(Dataset):  # Renamed for clarity in this independent script
    def __init__(self, image_ids, img_dir, transform=None):
        self.image_ids = image_ids  # image_ids is expected to be a list here
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name  # Return image tensor and its ID for submission


inference_transforms = transforms.Compose(
    [
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # ImageNet means and stds
    ]
)


class ASDA(nn.Module):
    def __init__(self, channel, input_H, input_W, reduction_ratio=4):
        super(ASDA, self).__init__()
        self.input_H = input_H
        self.input_W = input_W
        self.conv_3x3 = nn.Conv2d(
            channel, channel // 2, kernel_size=3, padding=1, bias=False
        )
        self.conv_5x5 = nn.Conv2d(
            channel, channel // 2, kernel_size=5, padding=2, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_1x1_reduce = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_spatial1 = nn.Linear(4 * 4, (4 * 4) // reduction_ratio, bias=False)
        self.fc_spatial2 = nn.Linear(
            (4 * 4) // reduction_ratio, input_H * input_W, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        f_3x3 = self.relu(self.conv_3x3(x))
        f_5x5 = self.relu(self.conv_5x5(x))
        f_local = torch.cat([f_3x3, f_5x5], dim=1)
        f_spatial_pre = self.conv_1x1_reduce(f_local)
        f_pooled = self.adaptive_pool(f_spatial_pre)
        f_pooled = f_pooled.view(b, -1)
        f_linear = self.relu(self.fc_spatial1(f_pooled))
        spatial_weights = self.fc_spatial2(f_linear).view(
            b, 1, self.input_H, self.input_W
        )
        spatial_weights = self.sigmoid(spatial_weights)
        return x * spatial_weights.expand_as(x)


class CCIA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCIA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.channel_interaction_conv = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = y.unsqueeze(1)
        y = self.channel_interaction_conv(y)
        y = y.squeeze(1)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNet50_Classifier(nn.Module):
    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        use_ccia=False,
        use_asda=False,
        weights_init_type="random",
    ):
        super(ResNet50_Classifier, self).__init__()

        self.resnet = models.resnet50(weights=None)

        self.use_ccia = use_ccia
        self.use_asda = use_asda

        self.resnet.fc = nn.Identity()

        if self.use_ccia:
            self.ccia_layer1 = CCIA(channel=256)
            self.ccia_layer2 = CCIA(channel=512)
            self.ccia_layer3 = CCIA(channel=1024)
            self.ccia_layer4 = CCIA(channel=2048)

        if self.use_asda:
            self.asda_layer1 = ASDA(
                channel=256,
                input_H=Config.RESNET_FM_SIZES["layer1"][0],
                input_W=Config.RESNET_FM_SIZES["layer1"][1],
            )
            self.asda_layer2 = ASDA(
                channel=512,
                input_H=Config.RESNET_FM_SIZES["layer2"][0],
                input_W=Config.RESNET_FM_SIZES["layer2"][1],
            )
            self.asda_layer3 = ASDA(
                channel=1024,
                input_H=Config.RESNET_FM_SIZES["layer3"][0],
                input_W=Config.RESNET_FM_SIZES["layer3"][1],
            )
            self.asda_layer4 = ASDA(
                channel=2048,
                input_H=Config.RESNET_FM_SIZES["layer4"][0],
                input_W=Config.RESNET_FM_SIZES["layer4"][1],
            )

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        if self.use_ccia:
            x = self.ccia_layer1(x)
        if self.use_asda:
            x = self.asda_layer1(x)

        x = self.resnet.layer2(x)
        if self.use_ccia:
            x = self.ccia_layer2(x)
        if self.use_asda:
            x = self.asda_layer2(x)

        x = self.resnet.layer3(x)
        if self.use_ccia:
            x = self.ccia_layer3(x)
        if self.use_asda:
            x = self.asda_layer3(x)

        x = self.resnet.layer4(x)
        if self.use_ccia:
            x = self.ccia_layer4(x)
        if self.use_asda:
            x = self.asda_layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    print("\n--- Starting Inference on Test Set ---")

    submission_df_template = pd.read_csv(Config.TEST_CSV)
    test_image_ids = submission_df_template["image_id"].tolist()

    test_dataset = TestDataset(
        image_ids=test_image_ids,
        img_dir=Config.TEST_IMAGES_DIR,  # CRUCIAL: Pointing to the TEST image directory
        transform=inference_transforms,  # Use inference transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE_INFERENCE,
        shuffle=False,  # No shuffling for consistent order
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
    )
    print(f"Total test samples for inference: {len(test_dataset)}")
    print(f"Total batches for inference: {len(test_loader)}")

    model_for_inference = ResNet50_Classifier(
        num_classes=Config.NUM_CLASSES,
        use_ccia=False,  # Set this based on how your best model was trained (True if it included CCIA)
        use_asda=True,  # Set this based on how your best model was trained (True if it included ASDA)
        weights_init_type="random",  # Match what you chose for training (random vs imagenet)
    )

    if os.path.exists(Config.BEST_MODEL_ASDA_PATH):
        print(f"Loading custom checkpoint from {Config.BEST_MODEL_ASDA_PATH}")
        state_dict = torch.load(Config.BEST_MODEL_ASDA_PATH, map_location=Config.DEVICE)
        model_for_inference.load_state_dict(state_dict)
    else:
        print(
            "Custom checkpoint not found. Loading ImageNet pretrained ResNet‑50 backbone."
        )
        pretrained_backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        model_for_inference.resnet.load_state_dict(pretrained_backbone.state_dict())

    model_for_inference.to(Config.DEVICE)
    model_for_inference.eval()  # Set model to evaluation mode for inference

    all_predictions = []
    all_image_ids_from_loader = (
        []
    )  # Collect image IDs from the loader to ensure correct order

    with torch.no_grad():  # Disable gradient calculation for inference
        for inputs, img_ids in tqdm(test_loader, desc="Predicting on test set"):
            inputs = inputs.to(Config.DEVICE)
            outputs = model_for_inference(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_image_ids_from_loader.extend(
                img_ids
            )  # Collect image IDs from the DataLoader

    submission_df = pd.DataFrame(
        {
            "image_id": test_image_ids,  # Use the original order from the template CSV
            "label": all_predictions,
        }
    )

    submission_file_path = os.path.join("/kaggle/working", "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)

    print(f"\nSubmission file saved to: {submission_file_path}")
    print("First 5 rows of generated submission.csv:")
    print(submission_df.head())

    print("\nInference on test set complete.")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/303536303.py in <cell line: 0>()
    260             weights=models.ResNet50_Weights.IMAGENET1K_V1
    261         )
--> 262         model_for_inference.resnet.load_state_dict(pretrained_backbone.state_dict())
    263         # Note: classifier head (self.fc) remains randomly initialized.
    264 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in load_state_dict(self, state_dict, strict, assign)
   2579 
   2580         if len(error_msgs) > 0:
-> 2581             raise RuntimeError(
   2582                 "Error(s) in loading state_dict for {}:\n\t{}".format(
   2583                     self.__class__.__name__, "\n\t".join(error_msgs)

RuntimeError: Error(s) in loading state_dict for ResNet:
	Unexpected key(s) in state_dict: "fc.weight", "fc.bias".
