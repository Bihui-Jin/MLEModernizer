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

0.9030625594169336

# 6. Current score

0.10436

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.19517) has done: 'I fixed the shape mismatch caused by EfficientNet‑B4 outputting 1792 features while the classifier, regressor and ordinal heads expected 1000, and also made weight loading tolerant to these changes. This lets the model run inference and produce a non‑empty `submission.csv` file.'
- What this solution (achieved -0.0881) has done: 'I replace the regression‑based class conversion with the model’s built‑in classifier head, which outputs logits for the five severity levels. By taking the arg‑max of the softmax probabilities we obtain a direct prediction that aligns better with the Quadratic Weighted Kappa metric, moving the score toward the target without altering the core architecture.'
- What this solution (achieved -0.078) has done: 'The fix ensures the regression head output retains a batch dimension before converting it to class probabilities, preventing the `IndexError`. The `regress2class_prob` function now safely reshapes 1‑D tensors and handles single‑sample batches. The inference loop is updated to pass the raw `r_out` tensor (without squeezing) to this function, allowing successful probability aggregation and generation of a non‑empty submission file. This resolves the runtime error and produces a valid `submission.csv`.'
- What this solution (achieved 0.19669) has done: 'I simplify the prediction step by using only the classifier head’s softmax probabilities (which are already aligned to the five severity classes) instead of averaging with the ordinal and regression heads. This reduces noisy or mis‑calibrated contributions from the other heads, moving the quadratic weighted kappa upward toward the target while preserving the core model architecture. The change is limited to the inference cell and leaves all other code unchanged.'
- What this solution (achieved -0.15512) has done: 'I enhance the inference step by combining the classifier, regression‑derived, and ordinal probabilities (instead of using only the classifier) and add a simple horizontal‑flip test‑time augmentation, then average these predictions before taking the argmax. This leverages more of the model’s learned heads and provides a modest boost toward the target score while preserving the original architecture and training logic.'
- What this solution (achieved 0.09698) has done: 'I simplify the inference step to rely only on the classifier head, which is the most reliable source of predictions, and keep the horizontal‑flip test‑time augmentation. By discarding the regression and ordinal heads (which were adding noisy probabilities) the predicted class probabilities become better calibrated for the quadratic weighted kappa metric, moving the score upward toward the target while preserving the rest of the pipeline unchanged.'
- What this solution (achieved -0.07739) has done: 'I keep the overall model and data pipeline unchanged but improve the inference step by leveraging all three heads (classifier, regression, ordinal) instead of only the classifier. For each TTA image I compute class probabilities from the classifier (softmax), from the regression output (via `regress2class_prob`) and from the ordinal output (via `ordinal2class_prob`), then average these three distributions before the final TTA averaging. This adds more calibrated information from the trained heads and is expected to raise the quadratic weighted kappa toward the target while preserving the core architecture.'
- What this solution (achieved -0.02689) has done: 'Implemented a simpler inference that relies solely on the classifier head’s softmax probabilities (the most reliable source observed in earlier attempts). This reduces noisy contributions from regression and ordinal heads, keeping the existing TTA flip averaging, and is expected to move the quadratic weighted kappa score upward toward the target.'
- What this solution (achieved 0.02451) has done: 'Implemented two fixes to get a non‑empty, valid submission and move the score toward the target:  
1. `regress2class_prob` now safely handles scalar tensors (0‑d) by reshaping them to (1, 1) before processing.  
2. During inference the regression head output is passed directly to `regress2class_prob` without an unsafe `squeeze()`.  

These changes resolve the `IndexError`, allow the model to generate class probabilities, and ensure the final CSV is written correctly.'
- What this solution (achieved 0.10436) has done: 'Implemented a streamlined inference that relies solely on the classifier head’s softmax probabilities (the most reliable signal observed) and removed the noisy regression / ordinal contributions. This change keeps the original model architecture untouched while improving prediction quality and moving the validation score closer to the target. The script now averages classifier probabilities over the original and horizontally‑flipped image, then selects the arg‑max class for the submission.'

# 9. Code solution

## === cell 0
try:
    import subprocess, sys, os

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "../input/weights/timm-0.3.1-py3-none-any.whl",
        ]
    )
except Exception:
    pass



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
    prediction = torch.zeros(out.size(0))
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
    if out.dim() == 0:
        out = out.view(1, 1)
    elif out.dim() == 1:
        out = out.unsqueeze(1)
    batch = out.size(0)
    pred_prob = torch.zeros((batch, 5), device=device)
    for i in range(batch):
        val = out[i].item()
        if val < 4.0:
            l1 = int(math.floor(val))
            l2 = int(math.ceil(val))
            pred_prob[i][l1] = 1 - (val - l1)
            pred_prob[i][l2] = 1 - (l2 - val)
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


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b5_ns", pretrained=False, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=False, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1792, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )
        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1792, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )
        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1792, 500),
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
        left = (w - new_w) / 2
        top = (h - new_h) / 2
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




## === cell 5
test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = np.squeeze(test_ids.values)

input_size = 380
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

weight_path = "../input/weights/B4_3stage_41epoch_CLAHE.pkl"
try:
    net.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    print("Custom weights loaded (partial).")
except FileNotFoundError:
    print(
        "Custom weight file not found – initializing backbone with pretrained weights."
    )
    net.backbone = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
    )
    net.backbone.global_pool = GeM(flatten=True)
except Exception as e:
    print(f"Unexpected error loading weights: {e}")

net = net.to(device)
net.eval()



## === cell 6
submission = []
net.eval()
with torch.no_grad():
    for i, idx in enumerate(test_ids):
        print(f"Processing {i+1}/{len(test_ids)}")
        image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
        img = Image.open(image_path).convert("RGB")

        img_flip = transforms.functional.hflip(img)

        probs = []
        for im in [img, img_flip]:
            img_tensor = transform(im).unsqueeze(0).to(device)

            c_out, _, _ = net(img_tensor)
            prob_c = F.softmax(c_out, dim=1)  # (1,5) probability distribution
            probs.append(prob_c)

        final_prob = torch.stack(probs).mean(dim=0)  # (1,5)
        pred_class = torch.argmax(final_prob, dim=1)

        submission.append([idx, int(pred_class.item())])

submission = np.array(submission)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, rows: {len(df)}")
