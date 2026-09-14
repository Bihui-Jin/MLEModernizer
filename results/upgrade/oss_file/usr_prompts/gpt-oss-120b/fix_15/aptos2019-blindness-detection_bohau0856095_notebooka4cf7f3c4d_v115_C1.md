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

0.9258867812669724

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime errors by (1) making the device selection CPU‑fallback when no GPU is available, (2) handling the missing weight file gracefully with a try/except and switching to a simple baseline predictor that uses the most frequent label from the training data, and (3) ensuring the submission DataFrame is correctly built and saved. These changes keep the original model code unchanged but guarantee the script runs end‑to‑end and produces a non‑empty `submission.csv`.'
- What this solution (achieved 0.70092) has done: 'The changes fix the attribute error in the EfficientNet wrapper by safely getting activation layers, replace the fragile threshold‑based class conversion with a robust `argmax` on the classifier logits, and ensure the script runs end‑to‑end producing a non‑empty `submission.csv`. These minimal fixes keep the original model architecture while correcting runtime bugs and allowing a valid submission.'
- What this solution (achieved 0.68314) has done: 'I keep the original model and data pipeline unchanged but improve the fallback‐centroid predictor. Instead of Euclidean distance, I compare cosine similarity between the image feature vector and class centroids (both L2‑normalized). This small change often yields better class discrimination and should raise the Quadratic Weighted Kappa toward the target score while preserving all core logic.'
- What this solution (achieved 0.50082) has done: 'I enhance the fallback prediction by using two different image preprocessings (the existing `transform1` and `transform2`) to extract two feature vectors, compute cosine similarity with the class centroids for each, average these similarities and pick the class with the highest average similarity. This leverages complementary information from both augmentations and is a minimal change that should raise the quadratic weighted kappa toward the target score while keeping the core model logic unchanged.'
- What this solution (achieved 0.38418) has done: 'The update improves the fallback centroid predictor by merging the two transformed feature vectors into a single normalized representation before similarity comparison, which gives a richer image descriptor and typically raises the quadratic weighted kappa. The change is isolated to the prediction loop, preserving all other logic and model architecture.'
- What this solution (achieved 0.50082) has done: 'I slightly improve the fallback predictor by L2‑normalising the two feature vectors extracted with the two different transforms before combining them, rather than adding the raw features. This yields a more stable image descriptor and typically raises the quadratic weighted kappa, moving the score closer to the target while keeping the overall architecture unchanged.'
- What this solution (achieved 0.43766) has done: 'Implemented fixes to resolve runtime errors and ensure a valid submission:
- Added missing `import timm` and clarified imports.
- Initialized fallback‑related variables before conditional logic.
- Refactored weight loading to set a clear `weight_loaded` flag and proper `use_fallback` determination.
- Guaranteed `backbone`, `train_df`, and related tensors are defined when needed.
- Preserved core model architecture while correcting variable scope and ensuring the script runs end‑to‑end, producing a non‑empty `submission.csv`.'
- What this solution (achieved 0.55584) has done: 'I improve the fallback k‑NN prediction by normalising the combined feature vector and using a similarity‑weighted average of the nearest‑neighbour labels instead of a simple mode vote. This small change keeps the overall model architecture unchanged while giving a more nuanced prediction, which should raise the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageChops
import timm
from timm.layers import GeM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
    return prediction


class backboneNet_efficient(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 5)

    def forward(self, x):
        batch = x.shape[0]
        dummy = torch.zeros(batch, 1, device=x.device)
        logits = self.fc(dummy)
        return dummy, logits, dummy




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/2259329166.py in <cell line: 0>()
     10 from PIL import Image, ImageChops
     11 import timm
---> 12 from timm.layers import GeM
     13 
     14 # device selection

ImportError: cannot import name 'GeM' from 'timm.layers' (/usr/local/lib/python3.11/dist-packages/timm/layers/__init__.py)

## === cell 1
class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
            transforms.functional.adjust_hue,
        ]
        random.shuffle(distortions)
        for d in distortions:
            if random.random() < 0.5:
                adjust_factor = (
                    random.uniform(-16 / 255.0, 16 / 255.0)
                    if d.__name__ == "adjust_hue"
                    else random.uniform(0.7, 1.3)
                )
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
        return image.crop((left, top, left + new_w, top + new_h))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        return image.crop(bbox) if bbox else image




## === cell 2
test_ids_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_ids_df["id_code"].values.squeeze()

transform1 = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

transform2 = transforms.Compose(
    [
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

backbone = None
centroid_predictor = None
norm_centroids = None
train_features_norm = None
train_labels = None
train_df = None

net1 = backboneNet_efficient()
weight_loaded = False
try:
    weight_path = "../input/weights/B4_3stage_13epoch_320finetune.pkl"
    if os.path.exists(weight_path):
        net1.load_state_dict(torch.load(weight_path, map_location=device))
        weight_loaded = True
        print("Loaded pretrained weights.")
    else:
        raise FileNotFoundError
except Exception as e:
    print(f"Weight loading failed ({e}); will use fallback baseline.")
    weight_loaded = False

use_ffallback = not weight_loaded  # typo intentional to keep original name logic safe
use_fallback = not weight_loaded  # correct flag for later logic

net1 = net1.to(device)
net1.eval()

if use_fallback:
    train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
    backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
    backbone.global_pool = GeM(flatten=True)
    backbone = backbone.to(device).eval()

    class SimpleDataset(Dataset):
        def __init__(self, df, img_dir, transform):
            self.ids = df["id_code"].values
            self.labels = df["diagnosis"].values
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            id_code = self.ids[idx]
            img_path = os.path.join(self.img_dir, f"{id_code}.png")
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            label = int(self.labels[idx])
            return img, label

    train_dataset = SimpleDataset(
        train_df,
        "../input/aptos2019-blindness-detection/train_images",
        transform2,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=2
    )

    class_sums = {}
    class_counts = {}
    all_feat_list = []
    all_label_list = []

    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            feats = backbone(imgs)  # [B, D]

            for f, l in zip(feats, labels):
                l = int(l.item())
                if l not in class_sums:
                    class_sums[l] = f.clone()
                    class_counts[l] = 1
                else:
                    class_sums[l] += f
                    class_counts[l] += 1

            norm_feats = F.normalize(feats, p=2, dim=1)  # [B, D]
            all_feat_list.append(norm_feats.cpu())
            all_label_list.append(labels.cpu())

    centroids = {cls: class_sums[cls] / class_counts[cls] for cls in class_sums}
    centroid_predictor = centroids
    norm_centroids = {
        cls: (c / (c.norm(p=2) + 1e-6)).clone() for cls, c in centroids.items()
    }
    print("Centroid predictor built from training data.")

    train_features_norm = torch.cat(all_feat_list, dim=0)  # [N, D]
    train_labels = torch.cat(all_label_list, dim=0)  # [N]
    print(
        f"Stored {train_features_norm.size(0)} training embeddings for k-NN fallback."
    )




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2421630565.py in <cell line: 0>()
     32 
     33 # attempt to load primary model
---> 34 net1 = backboneNet_efficient()
     35 weight_loaded = False
     36 try:

NameError: name 'backboneNet_efficient' is not defined

## === cell 3
submission = []
k_neighbors = 20  # slightly more neighbours for a smoother vote
with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(f"Processing {i}/{len(test_ids)}")
        image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
        img = Image.open(image_path).convert("RGB")

        if use_fallback and centroid_predictor is not None:
            f1 = backbone(transform1(img).unsqueeze(0).to(device))  # [1, D]
            f2 = backbone(transform2(img).unsqueeze(0).to(device))  # [1, D]

            n1 = F.normalize(f1.squeeze(0), p=2, dim=0)
            n2 = F.normalize(f2.squeeze(0), p=2, dim=0)
            combined_norm = (n1 + n2) / 2.0
            combined_norm = F.normalize(combined_norm, p=2, dim=0)  # unit length

            if train_features_norm is not None:
                sims = torch.matmul(train_features_norm, combined_norm.cpu())
                topk = torch.topk(sims, k=min(k_neighbors, sims.size(0)))
                top_sims = topk.values
                neighbor_labels = train_labels[topk.indices]

                class_sim = torch.zeros(5)  # 5 possible diagnoses 0‑4
                for cls in range(5):
                    mask = neighbor_labels == cls
                    if mask.any():
                        class_sim[cls] = top_sims[mask].sum()
                pred_label = int(torch.argmax(class_sim).item())
            else:
                max_sim = -float("inf")
                pred_label = None
                for cls, cent_norm in norm_centroids.items():
                    sim = F.cosine_similarity(combined_norm, cent_norm, dim=0)
                    sim_val = sim.item()
                    if sim_val > max_sim:
                        max_sim = sim_val
                        pred_label = cls
        elif use_fallback:
            pred_label = int(train_df["diagnosis"].mode().iloc[0])
        else:
            img_tensor = transform2(img).unsqueeze(0).to(device)
            _, r_out, _ = net1(img_tensor)
            pred_label = int(torch.argmax(r_out, dim=1).cpu().item())
        submission.append([idx, int(pred_label)])

submission = np.array(submission)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2987148735.py in <cell line: 0>()
      8         img = Image.open(image_path).convert("RGB")
      9 
---> 10         if use_fallback and centroid_predictor is not None:
     11             f1 = backbone(transform1(img).unsqueeze(0).to(device))  # [1, D]
     12             f2 = backbone(transform2(img).unsqueeze(0).to(device))  # [1, D]

NameError: name 'use_fallback' is not defined

## === cell 4
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, rows: {len(df)}")

## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame should not be empty
