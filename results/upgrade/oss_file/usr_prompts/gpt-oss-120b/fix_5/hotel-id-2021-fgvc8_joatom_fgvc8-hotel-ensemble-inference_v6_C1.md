# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import pandas as pd
import numpy as np
import torch
import fastai
from fastai.vision.all import *
import dill
from pathlib import Path
from torchvision import models as tv_models
from torchvision import transforms
from PIL import Image



## === cell 1
print(f"fastai {fastai.__version__}", f"torch {torch.__version__}")



## === cell 2
possible_model_paths = [
    "../input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",
    "../input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",
    "../input/fgvc8hotel/export_res101_Fall_HQAdam.pkl",
    "../input/fgvc8hotel/export_res101_Fall_5it_4.pkl",
    "../input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl",
    "/kaggle/input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",
    "/kaggle/input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",
    "/kaggle/input/fgvc8hotel/export_res101_Fall_HQAdam.pkl",
    "/kaggle/input/fgvc8hotel/export_res101_Fall_5it_4.pkl",
    "/kaggle/input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl",
]
model_files = [Path(p) for p in possible_model_paths if Path(p).exists()]
print(f"Found {len(model_files)} model file(s).")



## === cell 3
submission = pd.read_csv("../input/hotel-id-2021-fgvc8/sample_submission.csv")
test = submission.copy()

base_test_path = Path("../input/hotel-id-2021-fgvc8/test_images")
image_path_map = {p.name: p for p in base_test_path.rglob("*.jpg")}

test["image_path"] = test["image"].map(image_path_map.get)
test["chain"] = test["image_path"].apply(
    lambda p: (
        int(p.parent.name) if (p is not None and p.parent.name.isdigit()) else np.nan
    )
)

test["image"] = test["image_path"]



## === cell 4
probs = None
learn = None  # will hold the last successfully loaded learner (if any)

for model_path in model_files:
    try:
        learn = load_learner(fname=model_path, cpu=False, pickle_module=dill)
        test_dl = learn.dls.test_dl(test)
        probs_temp, _ = learn.tta(dl=test_dl, n=5)
        probs = probs_temp if probs is None else probs + probs_temp
        print(f"Loaded predictions from {model_path.name}")
    except Exception as e:
        print(f"Skipping {model_path.name}: {e}")

if probs is None:
    print("No models loaded – using K‑NN embedding fallback.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_model = tv_models.resnet34(pretrained=True)
    emb_model.fc = torch.nn.Identity()
    emb_model = emb_model.to(device).eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_path = "../input/hotel-id-2021-fgvc8/train.csv"
    if not Path(train_path).exists():
        train_path = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
    train_df = pd.read_csv(train_path)

    base_train_path = Path("../input/hotel-id-2021-fgvc8/train_images")
    train_image_map = {p.name: p for p in base_train_path.rglob("*.jpg")}
    train_df["image_path"] = train_df["image"].map(train_image_map.get)

    sample_frac = 20000
    if len(train_df) > sample_frac:
        train_sample = train_df.sample(sample_frac, random_state=42).reset_index(
            drop=True
        )
    else:
        train_sample = train_df.copy()

    train_embs = []
    train_ids = []

    for _, row in train_sample.iterrows():
        p = row["image_path"]
        if p is None or not Path(p).exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = emb_model(tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1).squeeze(0).cpu()
            train_embs.append(emb)
            train_ids.append(row["hotel_id"])
        except Exception:
            continue

    if len(train_embs) == 0:
        print("Embedding failed – reverting to global frequency baseline.")
        global_top5 = train_df["hotel_id"].value_counts().index.tolist()[:5]
        preds = [" ".join(map(str, global_top5))] * len(test)
    else:
        train_embs = torch.stack(train_embs)  # (N, 512)
        preds = []
        for _, row in test.iterrows():
            p = row["image_path"]
            if p is None or not Path(p).exists():
                fallback_ids = (
                    train_ids[:5]
                    if len(train_ids) >= 5
                    else train_ids + [train_ids[0]] * (5 - len(train_ids))
                )
                preds.append(" ".join(map(str, fallback_ids)))
                continue
            try:
                img = Image.open(p).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = emb_model(tensor)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1).squeeze(
                    0
                )  # (512,)
                sims = torch.mm(emb.unsqueeze(0), train_embs.t()).squeeze(0)  # (N,)
                topk = torch.topk(sims, k=5)
                top_ids = [train_ids[i] for i in topk.indices.tolist()]
                while len(top_ids) < 5:
                    top_ids.append(train_ids[0])
                preds.append(" ".join(map(str, top_ids)))
            except Exception:
                fallback_ids = (
                    train_ids[:5]
                    if len(train_ids) >= 5
                    else train_ids + [train_ids[0]] * (5 - len(train_ids))
                )
                preds.append(" ".join(map(str, fallback_ids)))
else:
    if isinstance(probs, torch.Tensor) and probs.ndim > 1:
        topk_indices = probs.topk(5, dim=1).indices  # (N,5)
        vocab = learn.dls.vocab if learn is not None else []
        preds = [
            " ".join(map(str, [vocab[i] for i in idx.tolist()])) for idx in topk_indices
        ]
    else:
        preds = [""] * len(submission)



## === cell 5
submission["hotel_id"] = preds
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
submission.head()
