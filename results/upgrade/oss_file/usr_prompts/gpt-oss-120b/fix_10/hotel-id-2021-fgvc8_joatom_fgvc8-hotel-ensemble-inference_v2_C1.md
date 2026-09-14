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
import os
import pandas as pd
import numpy as np
import torch
import fastai
from fastai.vision.all import *
import dill
from pathlib import Path
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(os.cpu_count() or 1)
torch.backends.cudnn.benchmark = True  # speed‑up on GPU without altering results




## === cell 1
print("fastai:", fastai.__version__, "torch:", torch.__version__)




## === cell 2
models = [
    "../input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",
    "../input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",
    "../input/fgvc8hotel/export_res101_Fall_HQAdam.pkl",
]




## === cell 3
submission = pd.read_csv(
    "../input/hotel-id-2021-hotel-id-2021-fgvc8/sample_submission.csv"
)
test = submission.copy()
test["image"] = "../input/hotel-id-2021-fgvc8/test_images/" + test["image"]




## === cell 4
probs_agg = None
model_loaded = False

first_model_path = next((p for p in models if Path(p).exists()), None)
if first_model_path:
    temp_learner = load_learner(
        fname=Path(first_model_path), cpu=False, pickle_module=dill
    )
    test_dl = temp_learner.dls.test_dl(test, bs=256, num_workers=4)

for model_path in models:
    try:
        learn = load_learner(fname=Path(model_path), cpu=False, pickle_module=dill)
        model_loaded = True
        probs_temp, _ = learn.tta(dl=test_dl)  # (n_samples, n_classes) probabilities
        log_probs = torch.log(probs_temp)
        probs_agg = log_probs if probs_agg is None else probs_agg + log_probs
    except FileNotFoundError:
        continue

if model_loaded:
    preds_idx = probs_agg.topk(5, dim=1)[1]  # indices of top‑5 classes
    preds = [" ".join(map(str, learn.dls.vocab[pred.tolist()])) for pred in preds_idx]
else:
    train_csv_path = "../input/hotel-id-2021-fgvc8/train.csv"
    train_images_root = Path("../input/hotel-id-2021-fgvc8/train_images")
    test_images_root = Path("../input/hotel-id-2021-fgvc8/test_images")

    train_df = pd.read_csv(train_csv_path)
    train_df["hotel_id"] = train_df["hotel_id"].astype(str)

    embed_cache_path = Path("train_embeddings.pt")
    labels_cache_path = Path("train_labels.npy")

    if embed_cache_path.exists() and labels_cache_path.exists():
        train_embeddings = torch.load(embed_cache_path)
        train_labels = np.load(labels_cache_path, allow_pickle=True).tolist()
    else:
        all_train_files = list(train_images_root.rglob("*.jpg"))
        file_map = {p.name: p for p in all_train_files}
        train_df["path"] = train_df["image"].map(file_map)
        train_df = train_df.dropna(subset=["path"]).reset_index(drop=True)

        class ImgDataset(Dataset):
            def __init__(self, paths, labels, transform):
                self.paths = paths
                self.labels = labels
                self.transform = transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                img = PILImage.create(self.paths[idx])
                img = self.transform(img)
                return img, self.labels[idx]

        tfms = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1]),
            ]
        )

        train_dataset = ImgDataset(
            train_df["path"].tolist(),
            train_df["hotel_id"].tolist(),
            tfms,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=os.cpu_count() or 4,
            pin_memory=True,
            persistent_workers=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = torchvision.models.resnet34(pretrained=True)
        resnet.fc = torch.nn.Identity()
        resnet.eval()
        resnet.to(device)

        train_embeddings_list = []
        train_labels = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for imgs, labs in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                emb = resnet(imgs)  # (B, 512)
                train_embeddings_list.append(emb.cpu())
                train_labels.extend(labs)
        train_embeddings = torch.cat(train_embeddings_list, dim=0).to(device)

        torch.save(train_embeddings.cpu(), embed_cache_path)
        np.save(labels_cache_path, np.array(train_labels, dtype=object))

    train_norm = (train_embeddings.pow(2).sum(dim=1, keepdim=True)).t()  # (1, N_train)

    test_paths = test["image"].tolist()
    test_dataset = ImgDataset(test_paths, [0] * len(test_paths), tfms)  # dummy labels
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=os.cpu_count() or 4,
        pin_memory=True,
        persistent_workers=True,
    )

    device = train_embeddings.device
    preds = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for imgs, _ in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            test_emb = resnet(imgs)  # (B, 512)
            test_norm = test_emb.pow(2).sum(dim=1, keepdim=True)  # (B, 1)
            dists = (
                test_norm + train_norm - 2.0 * test_emb @ train_embeddings.t()
            )  # (B, N_train)
            topk = torch.topk(-dists, 5, dim=1)[1]  # indices of nearest neighbours
            for inds in topk:
                nearest_ids = [train_labels[i] for i in inds.tolist()]
                preds.append(" ".join(nearest_ids))




## === cell 5
submission["hotel_id"] = preds
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
print(submission.head())
