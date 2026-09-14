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
from pathlib import Path
import fastai
from fastai.vision.all import *
import dill

print(f"fastai {fastai.__version__}, torch {torch.__version__}")



## === cell 1
models = [
    "../input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",  # v7
    "../input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",  # v8
    "../input/fgvc8hotel/export_res101_Fall_5it_4.pkl",  # v11
]



## === cell 2
submission_path = "../input/hotel-id-2021-fgvc8/sample_submission.csv"
submission = pd.read_csv(submission_path)
test = submission.copy()
test["image"] = "hotel-id-2021-fgvc8/test_images/" + test["image"]



## === cell 3
train_path = "../input/hotel-id-2021-fgvc8/train.csv"
train_df = pd.read_csv(train_path)
vocab = sorted(train_df["hotel_id"].astype(str).unique())
num_classes = len(vocab)
print(f"Number of unique hotel_id classes: {num_classes}")



## === cell 4
probs = None
learn = None  # will hold the last successfully loaded learner (for its vocab if needed)

for model_path in models:
    if not Path(model_path).exists():
        print(f"Model file not found, skipping: {model_path}")
        continue
    try:
        learn = load_learner(fname=Path(model_path), cpu=False, pickle_module=dill)
        test_dl = learn.dls.test_dl(test)
        probs_temp, _ = learn.tta(dl=test_dl, n=6)
        if probs is None:
            probs = probs_temp
        else:
            probs += probs_temp
    except Exception as e:
        print(f"Failed to load/apply model {model_path}: {e}")

if probs is None:
    print("No pretrained models loaded – using a lightweight k‑NN baseline.")
    from torchvision import models as tv_models, transforms as tv_transforms
    from PIL import Image
    import torch.nn.functional as F

    backbone = tv_models.resnet50(pretrained=True)
    backbone.fc = torch.nn.Identity()
    backbone.eval()

    preprocess = tv_transforms.Compose(
        [
            tv_transforms.Resize(256),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    def load_image(path):
        return preprocess(Image.open(path).convert("RGB"))

    train_subset_n = 5000  # size of the subset (adjustable)
    train_subset = train_df.head(train_subset_n)
    train_images_dir = "../input/hotel-id-2021-fgvc8/train_images"

    train_emb_list = []
    train_hotel_ids = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(train_subset), batch_size):
            batch_paths = []
            batch_ids = []
            batch = train_subset.iloc[i : i + batch_size]
            for _, row in batch.iterrows():
                img_path = Path(train_images_dir) / str(row["chain"]) / row["image"]
                if not img_path.exists():
                    continue
                batch_paths.append(img_path)
                batch_ids.append(str(row["hotel_id"]))
            if not batch_paths:
                continue
            batch_tensor = torch.stack([load_image(p) for p in batch_paths])
            emb = backbone(batch_tensor)
            train_emb_list.append(emb)
            train_hotel_ids.extend(batch_ids)
    if not train_emb_list:
        raise RuntimeError("No training embeddings could be generated.")
    train_embeddings = torch.cat(train_emb_list, dim=0)  # (N, D)
    train_embeddings = F.normalize(train_embeddings, dim=1)  # cosine‑norm

    test_image_paths = test["image"].tolist()
    test_emb_list = []
    with torch.no_grad():
        for i in range(0, len(test_image_paths), batch_size):
            batch_paths = test_image_paths[i : i + batch_size]
            batch_tensor = torch.stack([load_image(p) for p in batch_paths])
            emb = backbone(batch_tensor)
            test_emb_list.append(emb)
    test_embeddings = torch.cat(test_emb_list, dim=0)  # (M, D)
    test_embeddings = F.normalize(test_embeddings, dim=1)

    sim_matrix = torch.mm(test_embeddings, train_embeddings.t())
    top5_vals, top5_idx = sim_matrix.topk(5, dim=1)
    top5_hotel_ids = []
    for row in top5_idx.cpu().numpy():
        ids = [train_hotel_ids[idx] for idx in row]
        top5_hotel_ids.append(" ".join(ids))

    preds = top5_hotel_ids
    _knn_preds_ready = True
else:
    _knn_preds_ready = False



## === cell 5
if _knn_preds_ready:
    pass
else:
    top5_idx = probs.topk(5, dim=1)[1]  # shape: (n_test, 5)
    vocab_use = learn.dls.vocab if learn is not None else vocab
    preds = [" ".join(vocab_use[idx] for idx in row) for row in top5_idx.cpu().numpy()]



## === cell 6
submission["hotel_id"] = preds
submission_path_out = "submission.csv"
submission.to_csv(submission_path_out, index=False)
print(f"Submission file written to {submission_path_out}")
print(submission.head())
