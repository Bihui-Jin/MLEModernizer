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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.8

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
sklearn-pandas==2.2.0
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.03227

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, re, math, csv, random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.vision.all import *
from torchvision.models import vgg16_bn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = default_device()



## === cell 1
input_path = Path("/kaggle/input/denoising-dirty-documents")

path_train = input_path / "train"
path_train_cleaned = input_path / "train_cleaned"
path_test = input_path / "test"

path_working = Path("/kaggle/working")
path_submission_dir = path_working / "submission"
path_submission_dir.mkdir(parents=True, exist_ok=True)

sample_sub_path = input_path / "sampleSubmission.csv"
print("Input path:", input_path)
print("Train images:", len(list(path_train.glob("*.png"))))
print("Train cleaned:", len(list(path_train_cleaned.glob("*.png"))))
print("Test images:", len(list(path_test.glob("*.png"))))
print("Sample submission exists:", sample_sub_path.exists())



## === cell 2
bs, size = 4, 128
arch = resnet34


def label_func(fn: Path) -> Path:
    return path_train_cleaned / fn.name


item_tfms = Resize(size, method="bilinear")
batch_tfms = [Normalize.from_stats(*imagenet_stats, cuda=False)]

dblock = DataBlock(
    blocks=(ImageBlock, ImageBlock),
    get_items=get_image_files,
    get_y=label_func,
    splitter=RandomSplitter(valid_pct=0.2, seed=SEED),
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
)

dls = dblock.dataloaders(path_train, bs=bs, num_workers=2)
print(dls)
print("Train/valid:", len(dls.train_ds), len(dls.valid_ds))



## === cell 3
try:
    dls.show_batch(max_n=2, figsize=(6, 6))
except Exception as e:
    print("show_batch skipped:", repr(e))




## === cell 4
def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)) / (c * h * w)


base_loss = F.l1_loss



## === cell 5
vgg_m = vgg16_bn(weights="DEFAULT").features.to(device).eval()
for p in vgg_m.parameters():
    p.requires_grad = False

children_list = list(vgg_m.children())
blocks = [i - 1 for i, o in enumerate(children_list) if isinstance(o, nn.MaxPool2d)]
print("VGG blocks:", blocks)




## === cell 6
class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = (
            ["pixel"]
            + [f"feat_{i}" for i in range(len(layer_ids))]
            + [f"gram_{i}" for i in range(len(layer_ids))]
        )

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)

        feat_losses = [base_loss(input, target)]
        feat_losses += [
            base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        feat_losses += [
            base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w**2 * 5e3
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        self.feat_losses = feat_losses
        self.metrics = dict(zip(self.metric_names, feat_losses))
        return sum(feat_losses)

    def __del__(self):
        try:
            self.hooks.remove()
        except Exception:
            pass


feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5, 15, 2])



## === cell 7
wd = 1e-3

learn = unet_learner(
    dls,
    arch,
    n_out=3,
    loss_func=feat_loss,
    wd=wd,
    pretrained=True,
    normalize=False,  # we already normalized via batch_tfms
).to_fp32()

learn.path = path_working
learn.model_dir = Path("models")
print("Learner ready. Model dir:", learn.path / learn.model_dir)

gc.collect()
torch.cuda.empty_cache()



## === cell 8
try:
    learn.lr_find()
    learn.recorder.plot()
except Exception as e:
    print("lr_find skipped:", repr(e))

print(f"Validation set size: {len(learn.dls.valid_ds)}")



## === cell 9
lr = 1e-3


def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(10, lrs, pct_start=pct_start)
    learn.save(save_name)
    try:
        learn.show_results(max_n=1, figsize=(6, 6))
    except Exception as e:
        print("show_results skipped:", repr(e))




## === cell 10
do_fit("1a", slice(lr * 10))



## === cell 11
learn.unfreeze()
do_fit("1b", slice(1e-5, lr))



## === cell 12
data2_size = size * 2
dblock2 = DataBlock(
    blocks=(ImageBlock, ImageBlock),
    get_items=get_image_files,
    get_y=label_func,
    splitter=RandomSplitter(valid_pct=0.2, seed=SEED),
    item_tfms=Resize(data2_size, method="bilinear"),
    batch_tfms=batch_tfms,
)
dls2 = dblock2.dataloaders(path_train, bs=12, num_workers=2)

learn.dls = dls2
learn.freeze()
gc.collect()
torch.cuda.empty_cache()

learn.load("1b")



## === cell 13
do_fit("2a", slice(lr))



## === cell 14
learn.unfreeze()
do_fit("2b", slice(1e-6, 1e-4), pct_start=0.3)



## === cell 15
from skimage.color import rgb2gray as _rgb2gray
from imageio import imwrite


def rgb2gray_tensor(img_chw: torch.Tensor) -> torch.Tensor:
    """
    img_chw: Tensor [3,H,W] in [0,1]
    returns: Tensor [1,H,W] in [0,1]
    """
    img_hwc = img_chw.permute(1, 2, 0).detach().cpu().numpy()
    g = _rgb2gray(img_hwc).astype(np.float32)
    return torch.from_numpy(g).unsqueeze(0)


def write_image(fname, img_tensor_1chw):
    img_u8 = (img_tensor_1chw.clamp(0, 1) * 255).to(dtype=torch.uint8)
    imwrite(path_submission_dir / fname, img_u8.squeeze(0).cpu().numpy())




## === cell 16
learn.model.eval()


def predict_fullres(fn: Path) -> torch.Tensor:
    """
    Returns cleaned grayscale prediction as Tensor [1,H,W] in [0,1]
    """
    img_pil = PILImage.create(fn)
    test_dl = learn.dls.test_dl([img_pil], rm_type_tfms=True, num_workers=0)
    pred, _ = learn.get_preds(dl=test_dl)
    pred = pred[0].detach().cpu()

    mean = torch.tensor(imagenet_stats[0]).view(3, 1, 1)
    std = torch.tensor(imagenet_stats[1]).view(3, 1, 1)
    pred_denorm = (pred * std) + mean

    g = rgb2gray_tensor(pred_denorm)
    return g.clamp(0, 1)




## === cell 17
sample = pd.read_csv(sample_sub_path)
print(sample.head(), sample.shape)

id_parts = sample["id"].str.split("_", expand=True)
sample_img_ids = id_parts[0].astype(int).values

unique_img_ids = pd.unique(sample_img_ids)
print(
    "Unique test images in sample:", len(unique_img_ids), "Example:", unique_img_ids[:5]
)

pred_cache = {}

for img_id in unique_img_ids:
    fn = path_test / f"{img_id}.png"
    if not fn.exists():
        raise FileNotFoundError(f"Missing test image: {fn}")
    print("Processing:", img_id)
    g = predict_fullres(fn)  # [1,H,W]
    write_image(f"{img_id}.png", g)
    pred_cache[img_id] = g.squeeze(0).numpy().reshape(-1)  # row-major

rows = id_parts[1].astype(int).values - 1
cols = id_parts[2].astype(int).values - 1

any_id = unique_img_ids[0]
H, W = (
    (pred_cache[any_id].shape[0] // 420, 420) if False else (None, None)
)  # unused safeguard

values = np.empty(len(sample), dtype=np.float32)
for i, (img_id, r, c) in enumerate(zip(sample_img_ids, rows, cols)):
    if i == 0:
        pass
    gflat = pred_cache[img_id]
    values[i] = 0.0  # temporary; overwritten below

widths = {}
for img_id in unique_img_ids:
    img_pil = PILImage.create(path_test / f"{img_id}.png")
    widths[img_id] = img_pil.size[0]  # (W,H)

for i, (img_id, r, c) in enumerate(zip(sample_img_ids, rows, cols)):
    W = widths[img_id]
    values[i] = pred_cache[img_id][r * W + c]

sub = pd.DataFrame({"id": sample["id"], "value": values})
out_path = path_working / "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())
print("Submission file size (MB):", out_path.stat().st_size / 1e6)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/764040658.py in <cell line: 0>()
     56 for i, (img_id, r, c) in enumerate(zip(sample_img_ids, rows, cols)):
     57     W = widths[img_id]
---> 58     values[i] = pred_cache[img_id][r * W + c]
     59 
     60 sub = pd.DataFrame({"id": sample["id"], "value": values})

IndexError: index 65536 is out of bounds for axis 0 with size 65536
