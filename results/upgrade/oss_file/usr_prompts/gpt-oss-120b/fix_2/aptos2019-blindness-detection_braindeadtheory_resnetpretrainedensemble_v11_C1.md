# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.89353670176906

# 6. Current score

0.01711

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.01711) has done: 'The fix updates the data paths to use the absolute “/kaggle/input” location, replaces missing custom ResNet weight files with standard ImageNet‑pretrained weights, and adjusts the prediction routine to use a regular tqdm loop. These changes remove the file‑not‑found and undefined‑model errors, ensure the models are instantiated correctly, and finally write a proper `submission.csv` with the required columns. No core modeling logic is altered beyond loading the pretrained weights, keeping the original architecture and ensemble approach intact.'

# 9. Code solution

## === cell 0
import os, glob, json, pandas as pd, numpy as np

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):
        self.eye_frame = pd.read_csv(csv_file)
        self.filetype = filetype
        self.transform = transform
        self.base_path = os.path.join(
            "/kaggle/input",
            "aptos2019-blindness-detection",
            "train_images" if filetype == "train" else "test_images",
        )

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.base_path, self.eye_frame.loc[idx, "id_code"] + ".png"
        )
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.filetype == "train":
            label = int(self.eye_frame.loc[idx, "diagnosis"])
            return image, label
        else:
            img_id = self.eye_frame.loc[idx, "id_code"]
            return image, img_id




## === cell 2
test_dataset = APTOSDataset(
    csv_file="/kaggle/input/aptos2019-blindness-detection/test.csv",
    filetype="test",
    transform=transform,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=24, shuffle=False, num_workers=4, pin_memory=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 3
def build_resnet(model_name, pretrained=True):
    if model_name == "resnet152":
        model = torchvision.models.resnet152(pretrained=pretrained)
    else:  # resnet101
        model = torchvision.models.resnet101(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # 5 classes
    return model.to(device)


model0 = build_resnet("resnet152", pretrained=True)
model1 = build_resnet("resnet101", pretrained=True)
model2 = build_resnet("resnet101", pretrained=True)
model3 = build_resnet("resnet101", pretrained=True)




## === cell 4
def compute_predictions(model, mode, data_loader, device):
    """
    mode = 'train' or 'test'
    Returns:
        - For test: (DataFrame with id_code & diagnosis, list of logits tensors, list of ids)
        - For train: (list of predictions, accuracy)
    """
    model.eval()
    if mode == "train":
        preds = []
        correct, total = 0, 0
        for inputs, labels in tqdm(data_loader, desc="Train Predict"):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().numpy())
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        accuracy = correct / total * 100
        return preds, accuracy

    else:  # test
        logits_list = []
        img_ids = []
        for inputs, ids in tqdm(data_loader, desc="Test Predict"):
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            logits_list.append(outputs.cpu())
            img_ids.extend(ids)
        all_logits = torch.cat(logits_list, dim=0)
        _, preds = torch.max(all_logits, 1)
        preds = preds.numpy()
        df = pd.DataFrame({"id_code": img_ids, "diagnosis": preds})
        return df, all_logits, img_ids




## === cell 5
with torch.set_grad_enabled(False):
    print("Computing test predictions for all models...")
    pred_df0, out0, ids0 = compute_predictions(model0, "test", test_loader, device)
    pred_df1, out1, ids1 = compute_predictions(model1, "test", test_loader, device)
    pred_df2, out2, ids2 = compute_predictions(model2, "test", test_loader, device)
    pred_df3, out3, ids3 = compute_predictions(model3, "test", test_loader, device)

avg_logits = (out0 + out1 + out2 + out3) / 4.0
_, final_preds = torch.max(avg_logits, 1)
final_preds = final_preds.numpy()

submission = pd.DataFrame({"id_code": np.array(ids0), "diagnosis": final_preds})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
