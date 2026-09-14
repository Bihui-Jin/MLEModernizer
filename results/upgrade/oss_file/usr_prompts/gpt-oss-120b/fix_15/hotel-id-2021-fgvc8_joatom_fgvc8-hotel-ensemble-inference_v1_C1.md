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
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from pathlib import Path
import pandas as pd
from fastai.vision.all import *
import os

try:
    import faiss

    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True




## === cell 1
base_path = Path("../input/hotel-id-2021-fgvc8")
train_csv = base_path / "train.csv"
test_csv = base_path / "sample_submission.csv"
train_img_path = base_path / "train_images"
test_img_path = base_path / "test_images"

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

num_workers = min(32, os.cpu_count() or 1)
torch.set_num_threads(num_workers)




## === cell 2
def get_train_img_path(row):
    return train_img_path / str(row["chain"]) / row["image"]


dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=get_train_img_path,
    get_y=ColReader("hotel_id"),
    splitter=IndexSplitter([]),  # all data used for embedding extraction
    item_tfms=Resize(224, method="crop"),
)

dls = dblock.dataloaders(
    train_df,
    bs=1024,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    batch_tfms=Normalize.from_stats(*imagenet_stats),
)




## === cell 3
learn = cnn_learner(dls, resnet34, pretrained=True, loss_func=CrossEntropyLoss())
learn.model = learn.model[0]  # keep only the body (remove the head)
learn.model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learn.model.to(device)
learn.model.half()




## === cell 4
emb_path = Path("train_embeddings.pt")
lbl_path = Path("train_labels.pt")

if emb_path.is_file() and lbl_path.is_file():
    train_embeddings = torch.load(emb_path, map_location=device)  # (N,512) half
    train_labels = torch.load(lbl_path, map_location="cpu")
else:
    N = len(train_df)
    train_embeddings = torch.empty((N, 512), dtype=torch.float16, device=device)
    train_labels = torch.empty(N, dtype=torch.int64, device="cpu")
    idx = 0
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for xb, yb in dls.train:
            xb = xb.to(device)
            feats = learn.model(xb)  # (bs,512,7,7)
            feats = (
                F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)
            )  # (bs,512)
            feats = F.normalize(feats, p=2, dim=1)
            bs = feats.shape[0]
            train_embeddings[idx : idx + bs] = feats.half()
            train_labels[idx : idx + bs] = yb.cpu()
            idx += bs
    torch.save(train_embeddings, emb_path)
    torch.save(train_labels, lbl_path)

if _FAISS_AVAILABLE:
    train_embeddings_f32 = train_embeddings.cpu().to(torch.float32).numpy()
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        faiss_index = faiss.GpuIndexFlatIP(res, 512)
        faiss_index.add(train_embeddings_f32)
    else:
        faiss_index = faiss.IndexFlatIP(512)
        faiss_index.add(train_embeddings_f32)
else:
    train_embeddings_T = train_embeddings.t().contiguous()  # (512, N) on device

vocab = dls.vocab  # list of hotel_id strings ordered by class index




## === cell 5
test_image_paths = [test_img_path / fname for fname in test_df["image"].values]

test_dl = dls.test_dl(
    test_image_paths,
    bs=1024,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

preds_list = [None] * len(test_df)
idx = 0

with torch.inference_mode(), torch.cuda.amp.autocast():
    for xb in test_dl:
        if isinstance(xb, (list, tuple)):
            xb = xb[0]
        xb = xb.to(device)
        feats = learn.model(xb)  # (bs,512,7,7)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)  # (bs,512)
        feats = F.normalize(feats, p=2, dim=1)

        if _FAISS_AVAILABLE:
            query_np = feats.to(torch.float32).cpu().numpy()
            distances, topk_idx = faiss_index.search(query_np, 5)  # (bs,5)
            topk_idx = torch.from_numpy(topk_idx).to(torch.long)
        else:
            feats = feats.half()
            sims = torch.mm(feats, train_embeddings_T)  # (bs, N_train)
            _, topk_idx = sims.topk(5, dim=1)  # (bs,5)

        topk_cpu = topk_idx.cpu().numpy()
        batch_preds = [" ".join(vocab[i] for i in row) for row in topk_cpu]
        batch_len = len(batch_preds)
        preds_list[idx : idx + batch_len] = batch_preds
        idx += batch_len




## === cell 6
submission = pd.read_csv(test_csv)  # columns: image, hotel_id (placeholder)
submission["hotel_id"] = preds_list
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(submission.head())
