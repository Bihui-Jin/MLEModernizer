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

3.7

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

0.5027069774014674

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.24768) has done: 'We import a standard ResNet‑101 from torchvision, replace the undefined `resnet101` reference, and build a simple validation transform (resize → center‑crop → tensor → normalize). The model loads the provided checkpoint with `strict=False` to avoid key mismatches. In the inference loop we apply this transform to each image individually, get the arg‑max prediction, and write a proper `submission.csv` matching the required column names.'
- What this solution (achieved -0.02133) has done: 'I replace the hard‑argmax prediction with a soft‑max weighted expected rating (and round it) so that the output better reflects the model’s confidence; this usually raises the quadratic weighted kappa and moves the score toward the target. The rest of the pipeline stays unchanged.'
- What this solution (achieved -0.04511) has done: 'I add a lightweight test‑time augmentation (horizontal flip) to the inference loop and average the model’s logits before converting them to probabilities. This simple TTA usually yields a modest boost in prediction quality, moving the quadratic weighted kappa score closer to the target without changing the core model or training logic. The rest of the pipeline remains unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I increase the input resolution to the standard 224×224 size expected by ResNet‑101 and add two extra test‑time augmentations (vertical flip and combined horizontal + vertical flip). The logits from all four versions of each image are averaged before converting to a soft‑max probability, then to an expected rating and finally rounded to the nearest integer class. These modest changes keep the core model and training unchanged while giving the model more visual detail and more stable predictions, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.0) has done: 'The script failed because essential libraries (`os`, `torch`, `torch.nn`, `torchvision.transforms`, `PIL.Image`, `ImageOps`, and `pandas`) were never imported, leading to `NameError` exceptions. I added the missing imports at the top, kept the existing model loading, transforms, and inference logic unchanged, and ensured the submission file is correctly written as `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image, ImageOps

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = models.resnet101(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

possible_dirs = [
    "../input/weights",
    "/kaggle/input/weights",
    "./weights",
]

ckpt_path = None
for d in possible_dirs:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(".pth"):
                ckpt_path = os.path.join(d, f)
                break
    if ckpt_path:
        break

if ckpt_path and os.path.exists(ckpt_path):
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
else:
    print("No checkpoint found – using ImageNet‑pretrained weights only.")

print("Model loaded and set to evaluation mode.")

Transforms = {
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    110.63666788 / 255.0,
                    103.16065604 / 255.0,
                    96.29023126 / 255.0,
                ],
                std=[
                    38.7568578 / 255.0,
                    37.88248729 / 255.0,
                    40.02898126 / 255.0,
                ],
            ),
        ]
    )
}



## === cell 1
test_csv_path = os.path.join("../input/aptos2019-blindness-detection", "test.csv")
test_ids = pd.read_csv(test_csv_path, usecols=[0])

predictions = []

class_indices = torch.arange(5, dtype=torch.float32, device=device)

for idx, row in test_ids.iterrows():
    image_id = row[0]
    img_path = os.path.join(
        "../input/aptos2019-blindness-detection/test_images/",
        f"{image_id}.png",
    )
    img = Image.open(img_path).convert("RGB")

    imgs = [
        img,
        ImageOps.mirror(img),  # horizontal flip
        ImageOps.flip(img),  # vertical flip
        ImageOps.mirror(ImageOps.flip(img)),  # both flips
    ]

    logits_sum = torch.zeros(1, 5, device=device)

    with torch.no_grad():
        for aug in imgs:
            inp = Transforms["val"](aug).unsqueeze(0).to(device)  # [1,3,224,224]
            logits = model(inp)  # [1,5]
            logits_sum += logits

        avg_logits = logits_sum / len(imgs)  # average over TTA
        probs = torch.softmax(avg_logits, dim=1)  # [1,5]
        exp_rating = torch.sum(probs * class_indices, dim=1).item()
        pred_label = int(round(exp_rating))
        pred_label = max(0, min(4, pred_label))

    predictions.append(pred_label)

print(f"Generated predictions for {len(predictions)} images")

sample_submission_path = os.path.join(
    "../input/aptos2019-blindness-detection", "sample_submission.csv"
)
submission = pd.read_csv(sample_submission_path)
submission["diagnosis"] = predictions
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
