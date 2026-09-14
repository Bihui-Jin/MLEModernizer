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

3.12

# 3. Installed packages

geopandas==0.14.4
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
sklearn-pandas==2.2.0
timm==1.0.19
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

0.8917552996898972

# 6. Current score

-0.05762

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13717) has done: 'I add a safe fallback when loading the pretrained checkpoints: if the file is missing the code instead create the same model with ImageNet‑pretrained weights, ensuring the model list is never empty. I also simplify the ensemble aggregation by stacking the softmax outputs (removing the unnecessary `unsqueeze`) and keep the original weighting scheme. Finally, I make sure the final prediction variable exists before building the submission file.'
- What this solution (achieved -0.12604) has done: 'I fixed the path‑resolution function so it safely handles missing directories, ensured the model list is always populated (falling back to ImageNet‑pretrained weights when checkpoints are absent), and guarded the weight calculation against division‑by‑zero. These changes let the pipeline run end‑to‑end and create a proper `submission.csv` file.'
- What this solution (achieved 0.0) has done: 'I add a simple fallback that predicts the most common training diagnosis when no fine‑tuned checkpoints are found. This keeps the original ensemble logic unchanged for the usual case, but when all models fall back to ImageNet weights (which give near‑random predictions) the baseline constant prediction is much more sensible and raises the quadratic weighted kappa from a negative value toward the target. I also expose the train CSV path so the mode can be computed.'
- What this solution (achieved -0.15519) has done: 'I adjust the fallback logic so that when no fine‑tuned checkpoints are found the code still runs inference with the ImageNet‑pretrained models (using uniform ensemble weights). This gives predictions far better than a constant class and moves the quadratic weighted kappa toward the target. I also fix the uniform‑weight calculation to use the actual number of available models.'
- What this solution (achieved 0.0) has done: 'I adjust the fallback logic so the code uses the constant‑most‑common‑class prediction only when no fine‑tuned checkpoints are successfully loaded (i.e., `loaded_keys` is empty). This prevents the ensemble of ImageNet‑pretrained models—which give near‑random outputs—from degrading performance, moving the quadratic weighted kappa from a large negative value toward a positive score closer to the target. The change is limited to the decision flag in cell 4 and a brief comment explaining the rationale.'
- What this solution (achieved 0.15556) has done: 'I adjust the fallback logic so that, even when no fine‑tuned checkpoints are found, the code still runs inference with the ImageNet‑pretrained models using uniform ensemble weights. This removes the constant‑class baseline that gave a score of 0.0 and leverages the pretrained models to produce more informative predictions, moving the quadratic weighted kappa toward the target. The changes are confined to the model‑loading cell (to disable the baseline flag) and the inference cell (to always run inference unless no models exist).'
- What this solution (achieved -0.02449) has done: 'I adjust the weight computation so that, even when no fine‑tuned checkpoints are loaded, the ensemble uses the pre‑computed validation scores (normalized) instead of uniform weights. This gives higher‑scoring models more influence, which should improve the quadratic weighted kappa and move the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved -0.04828) has done: 'I adjust the weighting logic so that when none of the fine‑tuned checkpoints are found the ensemble uses uniform weights instead of the validation‑score‑based weights (which bias random ImageNet‑pretrained predictions). This small change keeps the overall pipeline intact while eliminating a source of harmful bias, moving the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.0) has done: 'I make the inference step fall back to the most‑common class whenever no fine‑tuned checkpoints were successfully loaded (i.e., `loaded_keys` is empty). This prevents the ensemble of ImageNet‑pretrained models—which give near‑random outputs—from degrading performance and moves the quadratic weighted kappa from a negative value toward the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved -0.05762) has done: 'I adjust the weighting logic so that even when no fine‑tuned checkpoints are loaded the ensemble still uses the validation‑score‑based weights (instead of falling back to a constant prediction). Then I remove the constant‑baseline shortcut in the inference cell and always run the ensemble, which should give predictions better than always predicting the most common class and therefore move the quadratic weighted kappa upward toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
loaded_keys = []  # keep track of which models were actually loaded with checkpoints

model_paths = {
    "efficientnet_b1": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext50_32x4d.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}


def resolve_path(path):
    """Return a valid checkpoint path if possible, otherwise None.
    Handles missing directories and common typos."""
    if os.path.isfile(path):
        return path
    corrected = path.replace("efficent", "efficient")
    if os.path.isfile(corrected):
        return corrected
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        return None
    basename = os.path.basename(path).lower().replace("efficent", "efficient")
    for f in os.listdir(dirname):
        if basename.split(".")[0] in f.lower():
            return os.path.join(dirname, f)
    return None


for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    checkpoint_path = resolve_path(path)
    try:
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model = timm.create_model(model_name, pretrained=False, num_classes=5)
            model.load_state_dict(state_dict)
            loaded = True
        else:
            raise FileNotFoundError
    except Exception:
        model = timm.create_model(model_name, pretrained=True, num_classes=5)
        loaded = False
    model.to(device)
    model.eval()
    models_list.append(model)
    if loaded:
        loaded_keys.append(model_key)

most_common_class = pd.read_csv(train_csv_file)["diagnosis"].mode()[0]

use_baseline = False




## === cell 5
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.9127,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}

if loaded_keys:
    total_score = sum(validation_scores[k] for k in loaded_keys)
    weights = {k: validation_scores[k] / total_score for k in loaded_keys}
else:
    total_score = sum(validation_scores[k] for k in model_paths.keys())
    weights = {k: validation_scores[k] / total_score for k in model_paths.keys()}




## === cell 6
all_outputs = []
num_test = len(test_dataset)  # total number of test samples

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        weighted_outputs = []
        for model_key, model in zip(model_paths.keys(), models_list):
            if model_key not in weights:
                continue  # skip models without a valid weight
            out = nn.functional.softmax(model(images), dim=1)
            weighted_outputs.append(weights[model_key] * out)
        if not weighted_outputs:
            out = nn.functional.softmax(models_list[0](images), dim=1)
            weighted_outputs.append(out)
        stacked = torch.stack(weighted_outputs)  # (M, B, 5)
        summed = torch.sum(stacked, dim=0)  # (B, 5)
        all_outputs.extend(summed.cpu().numpy())

    all_outputs = np.array(all_outputs)  # shape (num_test, 5)
    final_predictions = np.argmax(all_outputs, axis=1).astype(int)




## === cell 7
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"],
        "diagnosis": final_predictions,
    }
)
submission_df.to_csv("submission.csv", index=False)
