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

3.9

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.9237673810862916

# 6. Current score

-0.0352

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.04217) has done: 'The script was missing all required imports, a device definition, and proper handling of missing weight files, which caused the runtime errors. I added the missing imports, defined `device`, and organized the cells so the code runs start‑to‑finish and writes a valid `submission.csv`. No core model logic was changed, preserving the original approach while making the pipeline executable.'
- What this solution (achieved 0.01679) has done: 'I replace the regression‑based class conversion with a direct argmax of the classifier logits (the model’s first head). Using the classifier’s predictions is a standard way to obtain discrete grades and should raise the QWK score substantially toward the target. The change is limited to the inference loop, preserving the overall architecture and training logic.'
- What this solution (achieved -0.01653) has done: 'I replace the faulty regression‑to‑class conversion with a direct argmax on the classifier logits, which avoids the dimension error and aligns the prediction method with the model’s intended output. This change also tends to improve the quadratic weighted kappa score while keeping the overall architecture untouched. The rest of the pipeline remains the same, and a proper submission CSV be written.'
- What this solution (achieved 0.03818) has done: 'I keep the existing model and data pipeline unchanged, but adjust the inference step to combine the classifier’s soft‑max confidence with the regression head. When the classifier is uncertain (max probability < 0.6) we fall back to the regression output rounded to the nearest integer; otherwise we keep the classifier’s argmax. This simple calibration often raises the quadratic weighted kappa toward the target without altering the core architecture.'
- What this solution (achieved 0.0) has done: 'I add a simple fallback that uses the most frequent diagnosis from the training data whenever the custom weights are unavailable (which is the case now). This keeps the existing model code untouched but ensures a sensible prediction (the majority class) instead of random outputs, moving the QWK score much closer to the target. Additionally, I store the fallback flag and compute the mode from `train.csv` early in the script.'
- What this solution (achieved 0.0) has done: 'I adjust the inference step to always use the classifier’s arg‑max prediction (removing the confidence‑threshold fallback to the regression head). This simple change aligns the prediction method with the model’s intended classification output and should raise the quadratic weighted kappa score toward the target while keeping the core architecture untouched.'
- What this solution (achieved 0.0) has done: 'I keep the overall model and data pipeline unchanged, but improve the inference logic.  
When the pretrained three‑stage weights are available, the code now use test‑time augmentation (horizontal flip) and average the classifier logits and regression outputs. Then it pick the class with the highest soft‑max probability, falling back to the rounded regression prediction when the classifier confidence is low (max prob < 0.6). This small change is expected to raise the quadratic weighted kappa score toward the target while preserving the core architecture.'
- What this solution (achieved -0.02634) has done: 'The fix adds all missing imports, defines the computation device, and corrects the inference logic to always use the classifier’s arg‑max prediction (removing the low‑confidence fallback). These changes resolve the NameError issues, ensure a proper submission.csv is written, and modestly improve the model’s scoring behavior while keeping the core architecture unchanged.'
- What this solution (achieved -0.03287) has done: 'I enhance the inference step to use the classifier’s probability distribution rather than a simple arg‑max, and combine it with a confidence‑based fallback to the regression head (including flip augmentation). Using the expected value of the probabilities and a low‑confidence threshold provides a smoother, better‑calibrated prediction that should raise the quadratic weighted kappa toward the target while preserving the existing model architecture.'
- What this solution (achieved -0.03045) has done: 'Implemented a simpler, classifier‑focused inference: always use the soft‑max argmax of the averaged logits (including TTA flip) to predict the diagnosis, removing the low‑confidence fallback to the regression head. This keeps the core model unchanged while aligning predictions with the intended classification output, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.0352) has done: 'I keep the same model architecture and data pipeline, but improve the inference logic. Instead of using the classifier’s arg‑max, I compute the expected class from the soft‑max probabilities and round it, which better matches the quadratic weighted kappa metric. When the classifier’s confidence (max probability) is low (‑ < 0.6), I fall back to the regression head’s prediction (averaged over the original and flipped image). This small change preserves the core model while providing a more calibrated prediction that should raise the score toward the target.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import timm
from torchvision import transforms
import torchvision.transforms.functional as FT
from PIL import Image, ImageChops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p={:.4f}".format(self.p.data.tolist()[0])
            + ", eps={})".format(self.eps)
        )


class Regressor(nn.Module):
    """Simple regression model used for single‑stage inference."""

    def __init__(self):
        super(Regressor, self).__init__()
        backbone = timm.create_model(
            "tf_efficientnet_b5_ns", pretrained=True, num_classes=0
        )
        backbone.global_pool = GeM(flatten=True)
        feat_dim = backbone.num_features
        self.backbone = backbone
        self.regressor = nn.Linear(feat_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    """Three‑stage architecture with classifier, regressor and ordinal heads."""

    def __init__(self):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)
        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )
        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )
        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )
        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)
        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)

        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 1
test_ids_path = "../input/aptos2019-blindness-detection/test.csv"
train_csv_path = "../input/aptos2019-blindness-detection/train.csv"

test_ids_df = pd.read_csv(test_ids_path)
test_ids = test_ids_df["id_code"].values.squeeze()

if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    majority_class = int(train_df["diagnosis"].mode()[0])
else:
    majority_class = 0

input_size = 380


class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            FT.adjust_brightness,
            FT.adjust_contrast,
            FT.adjust_saturation,
            FT.adjust_hue,
        ]
        random.shuffle(distortions)
        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    adjust_factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    adjust_factor = random.uniform(0.7, 1.3)
                image = d(image, adjust_factor)
        return image


class cropTo4_3(object):
    def __call__(self, image):
        w, h = image.size
        if (w / h) >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w
        left = int((w - new_w) / 2)
        top = int((h - new_h) / 2)
        right = left + new_w
        bottom = top + new_h
        return image.crop((left, top, right, bottom))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image


transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()
weights_path = "../input/weights/B4_3stage_18epoch_finetune3.pkl"
weights_loaded = False
if os.path.exists(weights_path):
    try:
        state_dict = torch.load(weights_path, map_location=device)
        net.load_state_dict(state_dict)
        print("Custom weights loaded.")
        weights_loaded = True
    except Exception as e:
        print(
            f"Failed to load custom weights ({e}), using default pretrained backbone."
        )
else:
    print("Weights file not found, using default pretrained backbone.")

net = net.to(device)
net.eval()

if not weights_loaded:
    fallback_regressor = Regressor().to(device)
    fallback_regressor.eval()
    print("Fallback Regressor model instantiated for inference.")




## === cell 2
submission_list = []

class_indices = torch.arange(5, device=device, dtype=torch.float32)  # [0,1,2,3,4]

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        print(f"Processing {i + 1}/{len(test_ids)} – {idx}")
        img_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
        if not os.path.exists(img_path):
            pred = majority_class
            submission_list.append([idx, pred])
            continue

        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        if weights_loaded:
            c_out, r_out, _ = net(img)
            img_flipped = torch.flip(img, dims=[3])
            c_out_f, r_out_f, _ = net(img_flipped)

            c_out_avg = (c_out + c_out_f) / 2.0
            r_out_avg = (r_out + r_out_f) / 2.0

            probs = F.softmax(c_out_avg, dim=1)  # shape [1,5]
            max_prob, _ = probs.max(dim=1)  # confidence

            expected = torch.sum(probs * class_indices, dim=1)  # shape [1]

            if max_prob.item() >= 0.6:
                pred_val = expected.item()
            else:
                pred_val = r_out_avg.item()  # already scaled to 0‑4.5

            pred = int(round(pred_val))
        else:
            reg_out = fallback_regressor(img)
            pred = int(round(reg_out.item()))

        pred = max(0, min(4, pred))  # clamp to valid range
        submission_list.append([idx, pred])

submission_array = np.array(submission_list)




## === cell 3
df = pd.DataFrame(submission_array, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission file saved to {output_path}")
