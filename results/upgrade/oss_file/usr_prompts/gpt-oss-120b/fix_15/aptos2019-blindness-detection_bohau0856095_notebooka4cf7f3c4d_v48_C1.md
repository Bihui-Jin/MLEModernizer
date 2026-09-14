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

0.9062655211787224

# 6. Current score

0.00866

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the script robust by (1) falling back to CPU when CUDA isn’t available, (2) handling missing weight files gracefully (using the randomly‑initialized model instead), (3) safely loading the optional timm wheel, and (4) ensuring the prediction loop always creates a non‑empty DataFrame and writes a proper submission.csv. These fixes resolve the runtime errors while preserving the original model architecture and inference logic.'
- What this solution (achieved 0.06089) has done: 'I enable ImageNet‑pretrained weights for the EfficientNet backbone (changing only the pretrained flag) so the model starts from a sensible feature extractor instead of random initialization. This small tweak preserves the original architecture and inference flow while giving much better predictions, moving the validation score toward the target. No other logic is altered.'
- What this solution (achieved 0.01196) has done: 'I keep the overall model and training pipeline unchanged but improve inference by using the built‑in classifier head instead of the regressor output. The classifier provides five logits that map directly to the five DR grades, so taking the arg‑max yields a more appropriate discrete prediction than the thresholded regression output. I also wrap the inference in a `torch.no_grad()` block for safety and slightly cleaner code. This small change should raise the quadratic weighted kappa score toward the target while preserving the original architecture.'
- What this solution (achieved -0.05803) has done: 'I make the inference robust by (1) fixing the dataset and image paths so the images are actually loaded (preventing the fallback‑to‑zero predictions that caused the near‑zero score) and (2) using the regression head `r_out` instead of the untrained classifier logits, rounding it to the nearest integer class. This small change keeps the original model architecture intact while providing more meaningful predictions, moving the quadratic weighted kappa score toward the target.'
- What this solution (achieved 0.08909) has done: 'I switch the inference to use the model’s classifier logits (the five‑class head) instead of the regression output, taking the arg‑max as the predicted diagnosis. This simple change keeps the architecture unchanged but yields more appropriate discrete predictions, which should raise the quadratic weighted kappa from the current negative value toward the target score.'
- What this solution (achieved -0.20461) has done: 'I fix the image preprocessing size so the EfficientNet‑B4 backbone receives the expected 380 × 380 input (the previous non‑square resize harms feature extraction). This small change keeps the model architecture and inference unchanged while improving the quality of predictions, moving the quadratic weighted kappa score closer to the target.'
- What this solution (achieved 0.1569) has done: 'I keep the overall architecture and data handling unchanged but improve the inference step by turning the classifier logits into a probability distribution, computing the expected rating, and rounding it to the nearest integer (clamped to 0‑4). When the classifier confidence (maximum probability) is low I fall back to the regression head, which provides a continuous prediction that I also round. This small post‑processing tweak usually yields predictions that better reflect the underlying severity scores and moves the quadratic weighted kappa toward the target without altering the core model.'
- What this solution (achieved -0.0224) has done: 'I keep the model architecture and training logic unchanged, but improve the inference post‑processing.  
Instead of relying only on the classifier logits (and falling back to the regression head when confidence is low), I combine the classifier soft‑max probabilities with the ordinal‑head probabilities (via the existing `ordinal2class_prob` function). The averaged distribution gives a more calibrated expectation of the rating, which after rounding should raise the quadratic weighted kappa toward the target. The fallback to the regression head is kept only for very low confidence cases.'
- What this solution (achieved -0.03668) has done: 'I adjust the inference logic to use the classifier’s arg‑max prediction (the most confident class) instead of the averaged ordinal‑classifier expectation, and only fall back to the regression output when the classifier confidence is very low. This change keeps the model architecture unchanged while providing a sharper, more appropriate discrete prediction, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.02551) has done: 'I replace the prediction logic in the inference loop with a small, calibrated approach: compute the classifier’s expected rating (probability‑weighted sum) and combine it with the regression output by averaging, then round and clamp to the valid class range. This keeps the model architecture unchanged while providing a more nuanced prediction that should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.00559) has done: 'I adjust the inference post‑processing to rely solely on the classifier’s softmax logits, selecting the class with the highest probability (arg‑max) instead of averaging it with the regression output. This keeps the model architecture unchanged while providing a more appropriate discrete prediction, which should raise the quadratic weighted kappa score toward the target.'
- What this solution (achieved -0.05924) has done: 'I keep the original model and data handling unchanged and only modify the inference post‑processing. Instead of using the raw classifier arg‑max, I combine the classifier’s soft‑max expected rating with the regression output, average them and round to the nearest integer (clamped to 0‑4). This small calibration step is expected to produce predictions that better reflect the ordinal nature of the problem and move the quadratic weighted kappa score toward the target without altering the core architecture.'
- What this solution (achieved -0.04317) has done: 'I improve the post‑processing of the model outputs during inference.  
Instead of averaging the classifier’s expected rating with the regression value, I combine the classifier soft‑max probabilities with the ordinal‑head probabilities (via `ordinal2class_prob`) to form a more calibrated distribution, then take its expected value and round it to the final class. This keeps the original architecture and training unchanged while providing a richer, ensemble‑style prediction that should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.00866) has done: 'I adjust the inference post‑processing to rely on the classifier and ordinal heads only—both of which already produce calibrated class probabilities—by computing their expected class values and averaging them (dropping the noisy regression output). This small change keeps the model architecture intact while providing a clearer, more reliable prediction, which should raise the quadratic weighted kappa toward the target.'

# 9. Code solution

## === cell 0
import subprocess, sys, os

try:
    wheel_path = "../input/weights/timm-0.3.1-py3-none-any.whl"
    if os.path.isfile(wheel_path):
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path])
except Exception as e:
    print(f"Optional timm wheel not installed: {e}")



## === cell 1
import random, math, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2
from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



## === cell 2
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    """
    Convert regression output (continuous) to discrete class by counting thresholds.
    """
    prediction = torch.zeros(out.size(0), dtype=torch.long)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5).to(device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5)).to(device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob




## === cell 3
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


class ThreeStage_Model(nn.Module):
    def __init__(self):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )
        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )
        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
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




## === cell 4
def locate_path(*candidates):
    for p in candidates:
        if os.path.isfile(p) or os.path.isdir(p):
            return p
    return None


test_csv_path = locate_path(
    "/kaggle/input/aptos2019-blindness-detection/test.csv",
    "../input/aptos2019-blindness-detection/test.csv",
    "./input/aptos2019-blindness-detection/test.csv",
)
if test_csv_path is None:
    raise FileNotFoundError("test.csv not found in any known location.")

test_ids_df = pd.read_csv(test_csv_path)
test_ids = test_ids_df["id_code"].values

input_size = 380
transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()
weights_path = locate_path(
    "/kaggle/input/weights/B4_3stage_36epoch_CLAHE.pkl",
    "../input/weights/B4_3stage_36epoch_CLAHE.pkl",
    "./input/weights/B4_3stage_36epoch_CLAHE.pkl",
)
if weights_path and os.path.isfile(weights_path):
    try:
        net.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded pretrained weights.")
    except Exception as e:
        print(f"Failed to load weights ({e}); using ImageNet‑pretrained backbone.")
else:
    print("Weight file not found; using ImageNet‑pretrained backbone.")
net = net.to(device)
net.eval()



## === cell 5
submission = []
test_images_dir = locate_path(
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "../input/aptos2019-blindness-detection/test_images",
    "./input/aptos2019-blindness-detection/test_images",
)
if test_images_dir is None:
    raise FileNotFoundError("test_images directory not found in any known location.")

class_idx_tensor = torch.arange(5, dtype=torch.float32, device=device)

with torch.no_grad():
    for idx in test_ids:
        try:
            image_path = os.path.join(test_images_dir, f"{idx}.png")
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            c_out, r_out, o_out = net(img_tensor)

            cls_prob = F.softmax(c_out, dim=1).squeeze(0)  # [5]
            cls_exp = torch.sum(cls_prob * class_idx_tensor).item()

            ord_prob = ordinal2class_prob(o_out).squeeze(0)  # [5]
            ord_exp = torch.sum(ord_prob * class_idx_tensor).item()

            combined_exp = (cls_exp + ord_exp) / 2.0

            pred_class = int(round(combined_exp))
            pred_class = max(0, min(4, pred_class))  # clamp to valid range

            submission.append([idx, pred_class])
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            submission.append([idx, 0])  # safe fallback

df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, rows: {len(df)}")
