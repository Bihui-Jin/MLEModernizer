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
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

dill==0.4.0
fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.5748358171255361

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I fix the pipeline so it always produces a valid `submission.csv` even when the external pretrained FastAI `.pkl` is missing in this environment (the root cause of your crash). Concretely, I add a safe model-loading block that tries the provided path and, on failure, falls back to a deterministic “most frequent hotels” baseline derived from `train.csv`, which is score-positive vs. random and guarantees end-to-end execution. I also correct the cell numbering and make the test image paths robust to the actual Kaggle directory layout. The rest of your inference logic (top-5 formatting, submission schema) is preserved.'
- What this solution (achieved 0.00209) has done: 'I remove the hard failure when the external exported FastAI learner `.pkl` is missing and replace it with a deterministic, fast baseline that always runs within the timeout and still yields a valid MAP@5 submission: predicting the globally most frequent `hotel_id`s from `train.csv`. I keep your existing model-loading/inference path intact when the learner is available, so core logic remains the same in that case. I also fix the cell numbering to start at 1 (as required) and ensure the submission is always written as `submission.csv` with the exact required columns and row count. This should unblock end-to-end execution and improve score versus empty/invalid output.'
- What this solution (achieved 0.0014) has done: 'Your current score (0.00209) is far below the target (0.5748), so we should improve it with the smallest change that preserves your overall pipeline: keep “load exported fastai learner if available, otherwise fall back”. The main issue is that the fallback “global top-5 hotels” is too weak for MAP@5; a minimal but much stronger fallback is a simple chain-aware prior using the `train.csv` metadata, while still producing exactly the same submission format. Concretely, when the learner is missing, we predict the most frequent hotels within each `chain` (with a safe fallback to global top hotels), and we also make sure the test image paths are correct and robust. This keeps the model/training logic untouched and only improves the deterministic baseline used when the `.pkl` isn’t present.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0014) is far below the target (0.5748), so we need to legitimately improve predictions while keeping your overall pipeline intact. The main limiter is the fallback branch (when the `.pkl` learner isn’t found): it currently tries to infer `chain` from the test image folder, but test images are not organized by chain, so it effectively becomes a weak global-popularity baseline. I keep your “load learner if available, else fallback” structure, but make the fallback stronger by building an image-similarity retrieval baseline using a pretrained ResNet18 embedding (no training loop changes) and kNN over a small, fixed-size subset of train images; then predict top-5 hotels by neighbor vote. This stays within Kaggle constraints, runs end-to-end, and should move MAP@5 substantially toward the target versus the current near-random baseline.'
- What this solution (achieved 0.0) has done: 'I fix the runtime error in the fallback embedding branch by passing an instantiated ResNet18 model into `create_body` (fastai expects a model object, not the constructor function). I also make the pretrained-weight loading robust to the current torchvision API (some environments require specifying `weights=` instead of `pretrained=`), while keeping the same retrieval logic and output formatting. These changes are execution-critical and score-positive versus the current 0.0 because they allow the stronger kNN fallback to actually run and produce a valid `submission.csv`. The rest of your pipeline (try loading the exported learner, otherwise do embedding+kNN, then write submission) is preserved.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import pandas as pd
import numpy as np

import fastai
from fastai.vision.all import *
import dill

import torch
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
fastai.__version__, torch.__version__



## === cell 2
sub_path_candidates = [
    Path("../input/hotel-id-2021-fgvc8/sample_submission.csv"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv"),
    Path("/kaggle/data/hotel-id-2021-fgvc8/sample_submission.csv"),
    Path("../input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/sample_submission.csv"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/sample_submission.csv"),
    Path("/kaggle/data/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/sample_submission.csv"),
]
sub_path = next((p for p in sub_path_candidates if p.exists()), None)
if sub_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in expected locations: {sub_path_candidates}"
    )

submission = pd.read_csv(sub_path)
test = submission.copy()

test_img_dir_candidates = [
    sub_path.parent / "test_images",
    Path("../input/hotel-id-2021-fgvc8/test_images"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/test_images"),
    Path("/kaggle/data/hotel-id-2021-fgvc8/test_images"),
    Path("../input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/test_images"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/test_images"),
    Path("/kaggle/data/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/test_images"),
]
test_img_dir = next((p for p in test_img_dir_candidates if p.exists()), None)
if test_img_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images dir in expected locations: {test_img_dir_candidates}"
    )

test["image_path"] = (test_img_dir.as_posix() + "/" + test["image"].astype(str)).astype(
    str
)
test.head()



## === cell 3
learn = None
learner_path_candidates = [
    Path("../input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl"),
    Path(
        "/kaggle/input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl"
    ),
    Path(
        "../input/hotel-train-fastai-densnet161/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl"
    ),
    Path(
        "/kaggle/input/hotel-train-fastai-densnet161/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl"
    ),
]

for p in learner_path_candidates:
    if p.exists():
        learn = load_learner(
            fname=p, cpu=not torch.cuda.is_available(), pickle_module=dill
        )
        break

learn is not None, learner_path_candidates



## === cell 4
preds = None

if learn is not None:
    num_workers = min(4, os.cpu_count() or 2)
    pin = torch.cuda.is_available()
    bs = 128 if pin else 64

    test_dl = learn.dls.test_dl(
        test[["image_path"]].rename(columns={"image_path": "image"}),
        bs=bs,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    with learn.no_bar():
        probs, _ = learn.get_preds(dl=test_dl)

    preds_idx = probs.topk(5, dim=1).indices.cpu().numpy()
    vocab = np.asarray(list(map(str, learn.dls.vocab)), dtype=object)
    preds = [" ".join(vocab[row]) for row in preds_idx]
else:
    train_path_candidates = [
        Path("../input/hotel-id-2021-fgvc8/train.csv"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/train.csv"),
        Path("/kaggle/data/hotel-id-2021-fgvc8/train.csv"),
        Path("../input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/train.csv"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/train.csv"),
        Path("/kaggle/data/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/train.csv"),
    ]
    train_path = next((p for p in train_path_candidates if p.exists()), None)

    train_img_dir_candidates = [
        sub_path.parent / "train_images",
        Path("../input/hotel-id-2021-fgvc8/train_images"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/train_images"),
        Path("/kaggle/data/hotel-id-2021-fgvc8/train_images"),
        Path("../input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/train_images"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/train_images"),
        Path("/kaggle/data/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/train_images"),
    ]
    train_img_dir = next((p for p in train_img_dir_candidates if p.exists()), None)

    if train_path is None or train_img_dir is None:
        global_top5 = ["0", "1", "2", "3", "4"]
        preds = [" ".join(global_top5)] * len(test)
    else:
        train_df = pd.read_csv(train_path, usecols=["image", "hotel_id"])
        train_df["hotel_id"] = train_df["hotel_id"].astype(str)

        global_top5 = train_df["hotel_id"].value_counts().head(5).index.tolist()
        if len(global_top5) < 5:
            global_top5 = (global_top5 + ["0", "1", "2", "3", "4"])[:5]

        n_gallery = 25000
        if len(train_df) > n_gallery:
            train_df_g = train_df.sample(n=n_gallery, random_state=0).reset_index(
                drop=True
            )
        else:
            train_df_g = train_df.reset_index(drop=True)

        train_paths = (
            train_img_dir.as_posix() + "/" + train_df_g["image"].astype(str)
        ).tolist()
        train_hotels = train_df_g["hotel_id"].tolist()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from torchvision.models import resnet18 as tv_resnet18

            try:
                from torchvision.models import ResNet18_Weights

                base_model = tv_resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception:
                base_model = tv_resnet18(pretrained=True)
        except Exception:
            try:
                from fastai.vision.models import resnet18 as fa_resnet18

                base_model = fa_resnet18(pretrained=True)
            except Exception:
                base_model = resnet18(pretrained=True)

        body = create_body(base_model, pretrained=False, cut=-2)  # output feature map
        embed_model = nn.Sequential(body, AdaptiveAvgPool2d(1), Flatten()).to(device)
        embed_model.eval()

        tfm = Compose(
            [
                Resize(256, method=ResizeMethod.Squish),
                CenterCrop(224),
                IntToFloatTensor(),
                Normalize.from_stats(*imagenet_stats),
            ]
        )

        def _embed_batch(img_paths, bs=64):
            feats = []
            with torch.inference_mode():
                for i in range(0, len(img_paths), bs):
                    batch_paths = img_paths[i : i + bs]
                    ims = []
                    for p in batch_paths:
                        try:
                            im = PILImage.create(p)
                            ims.append(tfm(im))
                        except Exception:
                            ims.append(torch.zeros(3, 224, 224))
                    xb = torch.stack(ims).to(device)
                    fb = embed_model(xb)
                    fb = F.normalize(fb, p=2, dim=1)
                    feats.append(fb.detach().cpu())
            return torch.cat(feats, dim=0)

        gallery_emb = _embed_batch(train_paths, bs=96 if device.type == "cuda" else 32)
        gallery_hotels = np.asarray(train_hotels, dtype=object)

        test_paths = test["image_path"].tolist()
        test_emb = _embed_batch(test_paths, bs=128 if device.type == "cuda" else 32)

        k = 50
        preds = []
        gallery_emb_t = gallery_emb.T  # [D, N]
        for i in range(test_emb.shape[0]):
            sims = (test_emb[i : i + 1] @ gallery_emb_t).squeeze(0)  # [N]
            nn_idx = torch.topk(
                sims, k=min(k, sims.numel()), largest=True
            ).indices.numpy()
            nn_hotels = gallery_hotels[nn_idx]

            counts = {}
            order = []
            for h in nn_hotels:
                if h not in counts:
                    counts[h] = 0
                    order.append(h)
                counts[h] += 1
            ranked = sorted(order, key=lambda h: (-counts[h], order.index(h)))
            top5 = []
            seen = set()
            for h in ranked:
                if h in seen:
                    continue
                top5.append(str(h))
                seen.add(h)
                if len(top5) == 5:
                    break
            for h in global_top5:
                if len(top5) == 5:
                    break
                if h not in seen:
                    top5.append(str(h))
                    seen.add(h)
            preds.append(" ".join(top5))

preds[:3], len(preds), len(submission)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3703020155.py in <cell line: 0>()
     88 
     89         body = create_body(base_model, pretrained=False, cut=-2)  # output feature map
---> 90         embed_model = nn.Sequential(body, AdaptiveAvgPool2d(1), Flatten()).to(device)
     91         embed_model.eval()
     92 

NameError: name 'AdaptiveAvgPool2d' is not defined

## === cell 5
submission["hotel_id"] = preds
out_path = Path("submission.csv")
submission.to_csv(out_path, index=False)

assert list(submission.columns) == ["image", "hotel_id"]
assert len(submission) == len(pd.read_csv(sub_path))
out_path, submission.head()
