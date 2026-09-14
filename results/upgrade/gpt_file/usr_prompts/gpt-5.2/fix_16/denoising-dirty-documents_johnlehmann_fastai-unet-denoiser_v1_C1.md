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

0.32223

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28) has done: 'The failure comes from a 1-based/0-based mismatch in how the sample submission encodes row/col: columns in this competition are 1..W (so max col equals 256), but the code subtracts 1 and then still encounters `c=256`, indicating the parsing sometimes produces values outside the expected range. The safest minimal fix is to parse `id` robustly, keep row/col as 1-based integers, and index with `r-1, c-1` while also validating against the predicted image shape. Separately, your `predict_fullres` currently returns predictions at the resized resolution (because `test_dl` applies the same `Resize`), but the submission expects original 256×256 pixels; we upsample the model output back to the original image size before extracting pixels (score improvement and fixes out-of-bounds). Finally, we keep submission writing identical but ensure all predicted arrays are exactly 256×256 so indexing always matches the sample submission grid.'
- What this solution (achieved 0.26783) has done: 'Your score (0.28 RMSE) is far from the target (0.03227), so the most likely issue is a semantics mismatch rather than model quality. The biggest minimal fix is to make the model output and submission values live in the same pixel scale: currently you denormalize using ImageNet stats even though `normalize=False` in the learner and the model outputs are already in the normalized space induced by `batch_tfms`. I remove that incorrect denormalization and instead convert the model output back to grayscale directly (then clamp to [0,1]), while keeping the same model, training, and resizing/upsampling logic. I also force deterministic ordering when enumerating test images and keep the 256×256 alignment checks to prevent silent misalignment.'
- What this solution (achieved 0.34208) has done: 'Your current RMSE is far from the target, so the most likely issue is still a scale/space mismatch rather than model capacity. The minimal fix is to stop unnormalizing with ImageNet stats at inference: with `normalize=False` in the learner, fastai already handles normalization consistently, and your manual `_unnormalize_imagenet_like` is pushing predictions to the wrong intensity range. I also make the grayscale conversion robust by clamping the RGB prediction to a reasonable range before converting, and keep the 256×256 alignment exactly as required by the sample submission. These changes preserve the same model, loss, training schedule, and resizing/upsampling logic, but should move the score substantially toward the target by correcting output semantics.'
- What this solution (achieved 0.2799) has done: 'I fix the inference bug by handling `learn.predict` outputs correctly: in fastai it returns a `TensorImage` (already a tensor), not a PIL-like object with a `.px` attribute. I keep the same trained learner, transforms, and upsampling-to-original-resolution logic, but adjust `predict_fullres` to robustly accept either tensor or PIL and to ensure it’s CHW float in `[0,1]`. This unblocks end-to-end execution and guarantees every test prediction is exactly at the original image resolution so submission indexing matches the sample grid. Finally, I keep the submission writing identical (`id,value`) and ensure a `.csv` is produced in `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.27983) has done: 'Your score is still far from the target, so the most likely remaining issue is output semantics: the model was trained on ImageNet-normalized tensors (because of `batch_tfms=Normalize.from_stats(*imagenet_stats)`), but at inference you currently clamp the *normalized* network output to `[0,1]` and then gray-convert it, which puts pixels on the wrong scale. I minimally fix inference to (1) apply the *inverse* ImageNet normalization to the model output before grayscale conversion, and (2) make sure we feed `learn.predict` the same pre-processing pipeline by predicting from the test dataloader (avoids any transform mismatch). The model, training schedule, loss, and resizing/upsampling logic remain unchanged; only the prediction-to-pixel-value conversion is corrected to move RMSE down toward the target.'
- What this solution (achieved 0.3183) has done: 'I fix the submission indexing bug by parsing the `id` field robustly and clamping row/col indices to the valid `[1..256]` range, since the sample submission occasionally contains a col value of 257 and that currently crashes generation. This is a minimal change that unblocks end-to-end execution and ensures a valid `/kaggle/working/submission.csv` is always written. I also add a safety check to force every predicted image array to be exactly 256×256 before pixel extraction, keeping your model/training and inference semantics unchanged. These changes are score-neutral except that they prevent invalid/out-of-bounds pixels from breaking the run.'
- What this solution (achieved 0.31922) has done: 'Your current RMSE is still far from the target, so the most likely remaining issue is that the pixel intensity mapping at inference doesn’t match what the model actually learned. With `batch_tfms=Normalize.from_stats(*imagenet_stats)` and `normalize=False`, the model is trained to output **normalized** targets; for this setup, converting predictions back to `[0,1]` should use the exact inverse of the training normalization (as you do), but grayscale should match the true target’s channel convention more closely than a plain channel-mean. I make one minimal, metric-relevant change: convert RGB→gray using standard luminance weights (closer to typical grayscale conversion used by PIL/skimage and often to the dataset’s grayscale semantics), while keeping the same model, training, resizing/upsampling, and submission wiring. I also ensure the inverse-normalization tensors are created on the correct device/dtype to avoid subtle scale drift, without changing any learning logic.'
- What this solution (achieved 0.31323) has done: 'Your RMSE is still far from the target, so the most likely remaining issue is a train/inference mismatch rather than model capacity. The smallest fix that preserves your model/training core logic is to generate test predictions through the same fastai pipeline the model was trained with (including the same `Normalize.from_stats(*imagenet_stats)`), by using `learn.dls.test_dl(..., rm_type_tfms=None)` instead of the default behavior that can drop/alter type transforms. I also ensure the grayscale conversion matches the targets’ single-channel convention by extracting only one channel after inverse-normalization (instead of a luma mix), which is often closer to how these images are stored/loaded. Finally, I keep the same 256×256 upsampling and the same sampleSubmission-driven indexing so the submission schema stays identical while correcting prediction semantics.'
- What this solution (achieved 0.32223) has done: 'Your current RMSE (0.31323) is still far above the target (0.03227), so we should assume a remaining train/inference mismatch rather than insufficient model capacity. The smallest metric-relevant change is to ensure the submission values are derived from the model output in the same “space” the cleaned targets occupy: for this competition the targets are effectively single-channel grayscale intensities, so we should take the prediction in grayscale and apply inverse ImageNet normalization on that grayscale (not on RGB then pick a channel). I keep your model, training schedule, resizing/upsampling, and submission indexing identical, but change only the final prediction-to-pixel conversion to: (1) upsample normalized preds, (2) convert to grayscale in normalized space, (3) inverse-normalize using the corresponding grayscale mean/std derived from ImageNet stats, then clamp to [0,1]. This should move the score down toward the target without altering the core training logic.'

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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = default_device()
print("device:", device)



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
if torch.cuda.is_available():
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
if torch.cuda.is_available():
    torch.cuda.empty_cache()

learn.load("1b")



## === cell 13
do_fit("2a", slice(lr))



## === cell 14
learn.unfreeze()
do_fit("2b", slice(1e-6, 1e-4), pct_start=0.3)



## === cell 15
from imageio import imwrite

learn.model.eval()


def _to_chw_float_tensor(x) -> torch.Tensor:
    """
    Convert model output to float tensor [3,H,W] (still possibly normalized).
    """
    if isinstance(x, torch.Tensor):
        t = x
    else:
        try:
            t = x.px
        except Exception:
            t = torch.as_tensor(x)

    t = t.detach().float()
    if t.ndim == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3:
        pass
    else:
        raise ValueError(
            f"Unexpected prediction ndim: {t.ndim}, shape={tuple(t.shape)}"
        )

    if t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1).contiguous()

    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)

    if t.shape[0] != 3:
        raise ValueError(
            f"Expected 3 channels after conversion, got shape {tuple(t.shape)}"
        )

    return t


def _rgb_to_gray_normed(pred_normed_chw: torch.Tensor) -> torch.Tensor:
    """
    Change (metric-relevant, minimal): convert prediction to grayscale *in normalized space*
    and only then inverse-normalize with the corresponding gray mean/std. This matches the
    competition's single-channel ground-truth semantics more closely than inverse-norming
    RGB then picking/mixing channels.
    Returns: gray_normed [1,H,W]
    """
    w = pred_normed_chw.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    g = (pred_normed_chw * w).sum(dim=0, keepdim=True)
    return g


def _inv_imagenet_norm_gray(gray_normed_1hw: torch.Tensor) -> torch.Tensor:
    """
    Inverse-normalize a grayscale tensor that was produced as luma(R,G,B) in normalized space.
    If rgb_norm = (rgb01 - mean)/std, then:
      gray_norm = sum_i w_i * rgb_norm_i
               = sum_i w_i/std_i * rgb01_i  - sum_i w_i*mean_i/std_i
      => gray01 = (gray_norm - b) / a, where:
           a = sum_i w_i/std_i
           b = - sum_i w_i*mean_i/std_i
    """
    mean = gray_normed_1hw.new_tensor(imagenet_stats[0]).view(3, 1, 1)
    std = gray_normed_1hw.new_tensor(imagenet_stats[1]).view(3, 1, 1)
    w = gray_normed_1hw.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)

    a = (w / std).sum()  # scalar
    b = -(w * mean / std).sum()  # scalar

    gray01 = (gray_normed_1hw - b) / a
    return gray01.clamp(0, 1)


def write_image(fname, img_tensor_1chw):
    img_u8 = (img_tensor_1chw.clamp(0, 1) * 255).to(dtype=torch.uint8)
    imwrite(path_submission_dir / fname, img_u8.squeeze(0).cpu().numpy())


test_files = sorted(get_image_files(path_test), key=lambda p: int(p.stem))
test_dl = learn.dls.test_dl(test_files, with_labels=False, rm_type_tfms=None)

preds, _ = learn.get_preds(dl=test_dl)  # [N,3,h,w] in normalized space
print("Got preds:", preds.shape, "for test files:", len(test_files))


def predict_fullres_from_pred(pred_normed_chw: torch.Tensor) -> torch.Tensor:
    """
    Returns cleaned grayscale prediction as Tensor [1,256,256] in [0,1].
    """
    target_h, target_w = 256, 256
    pred_normed_chw = _to_chw_float_tensor(pred_normed_chw)

    pred_up_norm = F.interpolate(
        pred_normed_chw.unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    gray_norm = _rgb_to_gray_normed(pred_up_norm)  # [1,256,256] normalized
    gray_01 = _inv_imagenet_norm_gray(gray_norm)  # [1,256,256] in [0,1]

    if gray_01.shape != (1, target_h, target_w):
        gray_01 = F.interpolate(
            gray_01.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return gray_01


sample = pd.read_csv(sample_sub_path)
print(sample.head(), sample.shape)

parts = sample["id"].astype(str).str.split("_", expand=True)
if parts.shape[1] != 3:
    raise ValueError("Unexpected id format; expected 'image_row_col'")

sample_img_ids = parts[0].astype(np.int32).to_numpy()
rows_1b = parts[1].astype(np.int32).to_numpy()
cols_1b = parts[2].astype(np.int32).to_numpy()

rows_1b = np.clip(rows_1b, 1, 256)
cols_1b = np.clip(cols_1b, 1, 256)

unique_img_ids = np.unique(sample_img_ids)
print(
    "Unique test images in sample:", len(unique_img_ids), "Example:", unique_img_ids[:5]
)

id_to_idx = {int(p.stem): i for i, p in enumerate(test_files)}

pred_cache = {}
shapes = {}

for img_id in unique_img_ids:
    fn = path_test / f"{img_id}.png"
    if not fn.exists():
        raise FileNotFoundError(f"Missing test image: {fn}")
    if img_id not in id_to_idx:
        raise KeyError(f"Test image id {img_id} not found in test_files list")

    i = id_to_idx[img_id]
    g = predict_fullres_from_pred(preds[i])  # [1,256,256]
    write_image(f"{img_id}.png", g)

    g_np = g.squeeze(0).cpu().numpy()
    H, W = g_np.shape
    if (H, W) != (256, 256):
        raise ValueError(
            f"Predicted shape mismatch for image {img_id}: got {(H,W)} expected (256,256)"
        )
    pred_cache[img_id] = g_np
    shapes[img_id] = (H, W)

values = np.empty(len(sample), dtype=np.float32)

for img_id in unique_img_ids:
    mask = sample_img_ids == img_id
    if not np.any(mask):
        continue
    H, W = shapes[img_id]

    r0 = rows_1b[mask].astype(np.int64) - 1
    c0 = cols_1b[mask].astype(np.int64) - 1

    r0 = np.clip(r0, 0, H - 1)
    c0 = np.clip(c0, 0, W - 1)

    values[mask] = pred_cache[img_id][r0, c0]

sub = pd.DataFrame({"id": sample["id"], "value": values})
out_path = path_working / "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())
print("Submission file size (MB):", out_path.stat().st_size / 1e6)
