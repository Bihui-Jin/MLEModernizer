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

0.8598304030554627

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.38082) has done: 'I guard the model‑loading step so missing .pth files no longer crash the notebook, and if no models can be loaded I fall back to a deterministic baseline that predicts class 0 for every test image. This guarantees the pipeline runs to completion and writes a valid `submission.csv`. The changes are limited to error handling and a simple fallback inference; the core data‑handling logic remains unchanged.'
- What this solution (achieved 0.0) has done: 'I make the model‑loading logic smarter: for every architecture listed it now tries to load a provided checkpoint, but if the file is missing it falls back to the pretrained ImageNet weights instead of a single generic fallback model. This lets the ensemble use many strong pretrained nets whose validation scores we already have. I also change the final prediction step to use the expected class (probability‑weighted average) rounded to the nearest integer, which is more appropriate for the ordinal Quadratic Weighted Kappa metric than a plain argmax. These minimal adjustments keep the overall pipeline unchanged while improving the quality of the predictions toward the target score.'
- What this solution (achieved -0.26296) has done: 'I replace the expected‑value rounding step with a plain argmax over the weighted ensemble probabilities. Using the class with highest combined probability is the standard prediction for a classification task and typically yields a higher quadratic weighted kappa than rounding a weighted average, moving the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.0) has done: 'I replace the argmax‑based decision with a probability‑weighted expected value (rounded to the nearest integer and clamped to the valid label range). This simple change aligns the prediction step with the ordinal nature of the Quadratic Weighted Kappa metric and is expected to raise the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.21387) has done: 'I adjust the inference step so that the final class prediction uses the argmax of the weighted ensemble probabilities instead of the expected‑value rounding. This minor change aligns the prediction with the usual classification rule and is expected to move the Quadratic Weighted Kappa score closer to the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.00533) has done: 'I replace the argmax‑based decision with a probability‑weighted expected value (rounded to the nearest integer and clamped to the valid label range). This aligns the prediction step with the ordinal nature of the Quadratic Weighted Kappa metric and is expected to raise the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.0) has done: 'I add a lightweight fallback that predicts the most frequent training label whenever no fine‑tuned checkpoints are available (the current setup only loads ImageNet‑pretrained nets, which give near‑random outputs and a very low QWK). I also switch the ensemble decision from an expected‑value rounding to a simple argmax, which is more appropriate when the models are not calibrated. These minimal changes keep the original pipeline intact while providing a deterministic, higher‑scoring baseline that moves the metric toward the target.'

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
        image = Image.open(img_name)
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
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



## === cell 4
model_paths = {
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/seresnext101_32x4d.pth"
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



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
has_finetuned = False  # tracks whether any checkpoint was actually loaded

for model_key, model_name in model_names.items():
    checkpoint_path = model_paths.get(model_key, None)
    try:
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            model = timm.create_model(model_name, pretrained=False, num_classes=5)
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
            has_finetuned = True
        else:
            model = timm.create_model(model_name, pretrained=True, num_classes=5)
        model.to(device)
        model.eval()
        models_list.append((model_key, model))
    except Exception as e:
        print(f"Skipping model {model_key}: {e}")

if not models_list:
    fallback_name = "resnet18"
    fallback_model = timm.create_model(fallback_name, pretrained=True, num_classes=5)
    fallback_model.to(device)
    fallback_model.eval()
    models_list.append((fallback_name, fallback_model))



## === cell 6
validation_scores = {
    "resnet18": 0.879,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.897,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}
total_score = sum(validation_scores.values())
weights = {k: v / total_score for k, v in validation_scores.items()}



## === cell 7
all_outputs = []

class_indices = torch.arange(5, device=device, dtype=torch.float32)

if not has_finetuned:
    train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    most_common = pd.read_csv(train_csv_file)["diagnosis"].mode()[0]
    final_predictions = np.full(len(test_dataset), int(most_common), dtype=int)
else:
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            model_outputs = []
            for model_key, model in models_list:
                w = weights.get(
                    model_key, 1.0
                )  # default weight 1.0 for any unexpected model
                logits = model(images)
                probs = nn.functional.softmax(logits, dim=1) * w
                model_outputs.append(probs)
            combined = torch.stack(model_outputs, dim=0).sum(dim=0)  # (batch, 5)

            preds = torch.argmax(combined, dim=1).long().clamp(0, 4)
            all_outputs.append(preds.cpu().numpy())

    if not all_outputs:
        final_predictions = np.zeros(len(test_dataset), dtype=int)
    else:
        final_predictions = np.concatenate(all_outputs, axis=0)



## === cell 8
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)
submission_df.to_csv("submission.csv", index=False)
