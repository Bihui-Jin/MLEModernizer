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
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        input/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
            test_images/
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
            train_images/
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> input/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> working/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.9108465179007714

# 6. Current score

0.00929

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.04304) has done: 'Implemented fixes to resolve shape mismatches in the model and added robust weight loading. The backbone feature dimension is now dynamically obtained, aligning classifier, regressor, and ordinal heads. Weight loading now gracefully falls back to a pretrained EfficientNet if custom weights are missing or incompatible. These changes allow the inference loop to run without errors and generate a proper non‑empty submission CSV.'
- What this solution (achieved -0.13232) has done: 'Implemented a minimal change in the inference loop to use the model’s built‑in classifier head instead of the regression‑based conversion. The classifier provides direct five‑class logits, so taking the argmax yields a more appropriate class prediction, which is expected to raise the quadratic weighted kappa toward the target score without altering the core architecture or training regime. All other code remains unchanged.'
- What this solution (achieved -0.08213) has done: 'Implemented a robust weight‑loading routine that searches for any available *.pkl checkpoint and loads it with `strict=False` to avoid mismatches. Added a fallback to the pretrained EfficientNet backbone when no checkpoint is found. Adjusted inference to use the model’s `final` regressor output and the existing `regress2class` conversion (which respects the defined thresholds) instead of the raw classifier logits. These minimal changes keep the original architecture intact while leveraging the trained final head when possible, aiming to move the quadratic weighted‑kappa score much closer to the target.'
- What this solution (achieved 0.00701) has done: 'I adjust the inference step to use the model’s built‑in classifier logits instead of the final combined regressor output. By calling `net(img, final=False)` we get the classifier tensor `c_out`; taking the arg‑max yields a direct class prediction, which aligns with the original training objective and should raise the quadratic weighted kappa toward the target. This change is minimal, keeps the core architecture unchanged, and ensures a valid CSV is still written.'
- What this solution (achieved 0.03743) has done: 'I adjust the inference step to use the model’s final combined regressor output (which aggregates classifier, regressor, and ordinal information) and then convert that scalar prediction to a class label with the existing `regress2class` function. This minimal change keeps the architecture and training unchanged while likely producing more accurate predictions, moving the quadratic weighted kappa score toward the target.'
- What this solution (achieved -0.11491) has done: 'I adjust the inference step to use the model’s built‑in classifier logits instead of the final combined regressor output. By calling `net(img, final=False)` we obtain the classifier tensor `c_out`; taking `torch.argmax` yields a direct 0‑4 class prediction that matches the training objective and should raise the quadratic weighted‑kappa score toward the target while keeping the core architecture unchanged. No other parts of the pipeline are altered.'
- What this solution (achieved 0.09258) has done: 'We improve the inference step by blending the classifier’s arg‑max prediction with the regressor’s rounded output and by using a simple test‑time augmentation (horizontal flip). This modest change keeps the model architecture untouched, adds only minimal post‑processing, and is expected to raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.00929) has done: 'We modify the inference step to rely solely on the classifier’s soft‑max probabilities (averaged over the original and horizontally‑flipped image) and take the class with highest probability. This removes the noisy blending with the regressor output, which was hurting performance, while keeping the model architecture and training untouched. The rest of the pipeline stays the same, ensuring a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os, random, math, time, glob
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops, ImageOps
import cv2
from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze()
    return prediction.long()


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


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
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=False, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)

        feat_dim = (
            self.backbone.num_features
            if hasattr(self.backbone, "num_features")
            else 1792
        )

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




## === cell 2
test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = np.squeeze(test_ids.values)

input_size = 384
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

weights_path = "../input/weights/B4_3stage_27epoch_320.pkl"
candidate_paths = [weights_path] + glob.glob("../**/*.pkl", recursive=True)

loaded = False
for path in candidate_paths:
    if os.path.isfile(path):
        try:
            state = torch.load(path, map_location=device)
            net.load_state_dict(state, strict=False)
            print(f"Loaded weights from {path}")
            loaded = True
            break
        except Exception as e:
            print(f"Failed to load weights from {path} ({e})")
if not loaded:
    net.backbone = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
    )
    net.backbone.global_pool = GeM(flatten=True)
    print("No suitable checkpoint found; using pretrained EfficientNet backbone.")

net = net.to(device)
net.eval()



## === cell 3
submission = []
with torch.no_grad():
    for i, idx in enumerate(test_ids):
        print(f"Processing {i+1}/{len(test_ids)}: {idx}")
        img_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
        img = Image.open(img_path).convert("RGB")

        orig_tensor = transform(img).unsqueeze(0).to(device)
        c_out_orig, _, _ = net(orig_tensor, final=False)
        prob_orig = F.softmax(c_out_orig, dim=1)

        flipped_img = ImageOps.mirror(img)
        flip_tensor = transform(flipped_img).unsqueeze(0).to(device)
        c_out_flip, _, _ = net(flip_tensor, final=False)
        prob_flip = F.softmax(c_out_flip, dim=1)

        prob_avg = (prob_orig + prob_flip) / 2.0
        pred_class = prob_avg.argmax(dim=1).item()

        submission.append([idx, int(pred_class)])

if not submission:
    raise RuntimeError("Inference produced no predictions; submission list is empty.")
submission = np.array(submission)



## === cell 4
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, rows: {len(df)}")
