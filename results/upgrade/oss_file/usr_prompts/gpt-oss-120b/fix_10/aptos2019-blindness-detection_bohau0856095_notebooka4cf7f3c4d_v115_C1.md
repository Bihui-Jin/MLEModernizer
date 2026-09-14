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

# 9. Code solution

## === cell 0
import torch
import torchvision, csv, time, glob, copy, json, pickle
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models, datasets
from torch import nn as nn
from torch.optim import lr_scheduler
from timm.models import *
from torch.nn.parameter import Parameter


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
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


class backboneNet_efficient(nn.Module):
    def __init__(self):
        super(backboneNet_efficient, self).__init__()
        net = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)

        layers_to_train = ["blocks"]
        for name, parameter in net.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        self.num_features = 1792
        self.conv_stem = getattr(net, "conv_stem", nn.Identity())
        self.bn1 = getattr(net, "bn1", nn.Identity())
        self.act1 = getattr(net, "act1", getattr(net, "act", nn.Identity()))
        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.conv_head = net.conv_head
        self.bn2 = getattr(net, "bn2", nn.Identity())
        self.act2 = getattr(net, "act2", getattr(net, "act", nn.Identity()))
        self.global_pool = net.global_pool
        self.drop_rate = 0.4
        self.rg_cls = Linear(self.num_features, 1, bias=True)
        self.cls_cls = Linear(self.num_features, 5, bias=True)
        self.ord_cls = Linear(self.num_features, 4, bias=True)

    def forward(self, x):
        x1 = self.conv_stem(x)
        x2 = self.bn1(x1)
        x3 = self.act1(x2)
        x4 = self.block0(x1)
        x5 = self.block1(x4)
        x6 = self.block2(x5)
        x7 = self.block3(x6)
        x8 = self.block4(x7)
        x9 = self.block5(x8)
        x10 = self.block6(x9)
        x11 = self.conv_head(x10)
        x12 = self.bn2(x11)
        x13 = self.act2(x12)
        x14 = self.global_pool(x13)
        if self.drop_rate > 0.0:
            x14 = F.dropout(x14, p=self.drop_rate, training=self.training)
        x15 = self.rg_cls(x14)
        x16 = self.cls_cls(x14)
        x17 = self.ord_cls(x14)
        return x15, x16, x17




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
    return prediction




## === cell 2
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




## === cell 3
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

net1 = backboneNet_efficient()
use_fallback = False
try:
    weight_path = "../input/weights/B4_3stage_13epoch_320finetune.pkl"
    if os.path.exists(weight_path):
        net1.load_state_dict(torch.load(weight_path, map_location=device))
        print("Loaded pretrained weights.")
    else:
        raise FileNotFoundError
except Exception as e:
    print(f"Weight loading failed ({e}); will use fallback baseline.")
    use_fallback = True

net1 = net1.to(device)
net1.eval()

centroid_predictor = None
norm_centroids = None
train_features_norm = None
train_labels = None

if use_fallback:

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

    backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
    backbone.global_pool = GeM(flatten=True)
    backbone = backbone.to(device).eval()

    train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
    train_dataset = SimpleDataset(
        train_df, "../input/aptos2019-blindness-detection/train_images", transform2
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
            feats = backbone(imgs)  # shape [B, D]

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

    centroids = {}
    for cls in class_sums:
        centroids[cls] = class_sums[cls] / class_counts[cls]
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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2644159424.py in <cell line: 0>()
----> 1 test_ids_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
      2 test_ids = test_ids_df["id_code"].values.squeeze()
      3 
      4 transform1 = transforms.Compose(
      5     [

NameError: name 'pd' is not defined

## === cell 4
submission = []
k_neighbors = 5  # number of nearest neighbours to vote
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

            combined_norm = (n1 + n2) / 2.0  # [D]

            if train_features_norm is not None:
                sims = torch.matmul(train_features_norm, combined_norm.cpu())  # [N]
                topk = torch.topk(sims, k=min(k_neighbors, sims.size(0)))
                neighbor_labels = train_labels[topk.indices]
                pred_label = int(neighbor_labels.mode().values.item())
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




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4155047873.py in <cell line: 0>()
      2 k_neighbors = 5  # number of nearest neighbours to vote
      3 with torch.no_grad():
----> 4     for i, idx in enumerate(test_ids):
      5         if i % 50 == 0:
      6             print(f"Processing {i}/{len(test_ids)}")

NameError: name 'test_ids' is not defined

## === cell 5
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, rows: {len(df)}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1190200053.py in <cell line: 0>()
----> 1 df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
      2 output_path = "submission.csv"
      3 df.to_csv(output_path, index=False)
      4 print(f"Submission saved to {output_path}, rows: {len(df)}")

NameError: name 'pd' is not defined
