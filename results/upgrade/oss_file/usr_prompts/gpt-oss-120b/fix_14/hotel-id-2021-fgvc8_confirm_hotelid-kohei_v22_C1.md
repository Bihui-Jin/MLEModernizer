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

No external packages required in the script and installed.

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
import random
import numpy as np

BASE_DIR = "/kaggle/input/hotel-id-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUBMIT_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_ROOT = os.path.join(BASE_DIR, "train_images")
TEST_IMG_ROOT = os.path.join(BASE_DIR, "test_images")
OUTPUT_SUBMIT = "submission.csv"
TOP_N = 5
MAX_TRAIN_PER_HOTEL = 20  # limit to keep embedding stage fast
RANDOM_SEED = 42

train_df = pd.read_csv(TRAIN_CSV, dtype=str)

try:
    import torch
    from torch import nn
    from torchvision import models, transforms
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Identity()  # output 2048‑dim feature vector
    backbone = backbone.to(device).eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def load_image(path: str):
        img = Image.open(path).convert("RGB")
        return preprocess(img).unsqueeze(0).to(device)

    def embed_image(path: str):
        with torch.no_grad():
            return backbone(load_image(path)).cpu().numpy().flatten()

    class ImagePathDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            p = self.paths[idx]
            try:
                img = Image.open(p).convert("RGB")
                return preprocess(img)
            except Exception:
                return torch.zeros(3, 224, 224)

    TORCH_AVAILABLE = True
    print("Torch import successful – using visual embeddings.")
except Exception as e:
    TORCH_AVAILABLE = False
    print("Torch not available or failed to load:", e)
    print("Falling back to original filename heuristic.")

if not TORCH_AVAILABLE:
    PREFIX_LEN = 2
    SUFFIX_LEN = 2
    global_top_ids = train_df["hotel_id"].value_counts().head(TOP_N).index.tolist()
    prefix_groups = (
        train_df.assign(prefix=train_df["image"].str[:PREFIX_LEN])
        .groupby("prefix")["hotel_id"]
        .apply(lambda x: x.value_counts().index.tolist()[:TOP_N])
        .to_dict()
    )
    suffix_groups = (
        train_df.assign(suffix=train_df["image"].str[-SUFFIX_LEN:])
        .groupby("suffix")["hotel_id"]
        .apply(lambda x: x.value_counts().index.tolist()[:TOP_N])
        .to_dict()
    )
    image_to_hotel = train_df.set_index("image")["hotel_id"].to_dict()




## === cell 1
if TORCH_AVAILABLE:
    random.seed(RANDOM_SEED)

    hotel_to_paths = {}
    for _, row in train_df.iterrows():
        hotel = row["hotel_id"]
        img_path = os.path.join(TRAIN_IMG_ROOT, row["chain"], row["image"])
        hotel_to_paths.setdefault(hotel, []).append(img_path)

    sampled_paths = []
    sampled_hotel_ids = []
    for hotel, paths in hotel_to_paths.items():
        sample = random.sample(paths, min(len(paths), MAX_TRAIN_PER_HOTEL))
        sampled_paths.extend(sample)
        sampled_hotel_ids.extend([hotel] * len(sample))

    BATCH_SIZE = 64
    dataset = ImagePathDataset(sampled_paths)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=False
    )

    hotel_means = {}
    cur_idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embeddings = backbone(batch).cpu().numpy()  # (batch, 2048)
            for emb in embeddings:
                hid = sampled_hotel_ids[cur_idx]
                hotel_means.setdefault(hid, []).append(emb)
                cur_idx += 1

    for hid, vecs in hotel_means.items():
        hotel_means[hid] = np.mean(vecs, axis=0)

    hotel_ids_list = list(hotel_means.keys())
    hotel_matrix = np.stack([hotel_means[hid] for hid in hotel_ids_list])
    print(f"Computed mean embeddings for {len(hotel_ids_list)} hotels.")




## === cell 2
def build_prediction(image_name: str) -> str:
    """
    Return a space‑separated list of TOP_N hotel IDs for the given image.
    Uses visual similarity if torch is available; otherwise falls back
    to the original filename‑based heuristic.
    """
    if TORCH_AVAILABLE:
        img_path = os.path.join(TEST_IMG_ROOT, image_name)
        try:
            query_vec = embed_image(img_path)
        except Exception:
            query_vec = np.zeros(2048, dtype=np.float32)

        dists = np.linalg.norm(hotel_matrix - query_vec, axis=1)
        nearest_idx = np.argsort(dists)[:TOP_N]
        preds = [hotel_ids_list[i] for i in nearest_idx]
        return " ".join(preds)

    preds = []

    if image_name in image_to_hotel:
        preds.append(image_to_hotel[image_name])

    if len(preds) < TOP_N:
        prefix = image_name[:PREFIX_LEN]
        for pid in prefix_groups.get(prefix, []):
            if pid not in preds:
                preds.append(pid)
            if len(preds) == TOP_N:
                break

    if len(preds) < TOP_N:
        suffix = image_name[-SUFFIX_LEN:]
        for sid in suffix_groups.get(suffix, []):
            if sid not in preds:
                preds.append(sid)
            if len(preds) == TOP_N:
                break

    if len(preds) < TOP_N:
        for gid in global_top_ids:
            if gid not in preds:
                preds.append(gid)
            if len(preds) == TOP_N:
                break

    return " ".join(preds)




## === cell 3
sample_sub_df = pd.read_csv(SAMPLE_SUBMIT_CSV, dtype=str)

expected_cols = {"image", "hotel_id"}
if not expected_cols.issubset(set(sample_sub_df.columns)):
    raise ValueError(f"Sample submission must contain columns {expected_cols}")

sample_sub_df["hotel_id"] = sample_sub_df["image"].apply(build_prediction)

sample_sub_df.to_csv(OUTPUT_SUBMIT, index=False)

print(f"Submission file written to {OUTPUT_SUBMIT}")
print(sample_sub_df.head())
