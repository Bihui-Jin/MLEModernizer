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

# 5. Target score

0.7445482866043597

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replaced the failing external module call with a self‑contained baseline that predicts the five most common hotel IDs for every test image. The script now safely loads the training data, computes global hotel‑id frequencies, builds a constant prediction string of the top‑5 IDs, applies it to all rows of the provided sample submission, and writes a correctly formatted `submission.csv`. This fixes the syntax and file‑not‑found errors while producing a valid submission ready for evaluation.'
- What this solution (achieved 0.00214) has done: 'I add a lightweight per‑image heuristic: use the first two characters of each image filename to look up the most common hotel ID seen with that prefix in the training data. For each test image the prediction list starts with this prefix‑specific ID (if any) and is then filled with the globally most frequent IDs to reach five IDs. This keeps the original constant‑prediction logic but tailors it slightly per image, which should raise MAP@5 toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.00142) has done: 'I extend the prefix‑based heuristic by using a longer prefix (3 characters) and by storing the top‑5 most frequent hotel IDs for each prefix instead of only the single most common one. The prediction now starts with the prefix‑specific list (up to five IDs) and is padded with the global most frequent IDs, which should raise MAP@5 modestly and move the score closer to the target while keeping the original logic intact.'
- What this solution (achieved 0.00142) has done: 'I add a direct lookup that assigns the exact hotel ID when a test image filename also appears in the training set, which gives a perfect first‑rank prediction for those cases. For all other images the existing prefix‑based heuristic remains unchanged, still padded with the global most‑frequent IDs. This small lookup can noticeably raise MAP@5 toward the target while preserving the original workflow and without altering any core modeling logic.'
- What this solution (achieved 0.00235) has done: 'I increase the specificity of the heuristic by using a longer filename prefix (6 characters) and also adding a suffix‑based lookup. The code now builds both prefix and suffix maps, tries an exact match first, then fills predictions with prefix‑specific IDs, suffix‑specific IDs, and finally the global most frequent IDs, keeping the total to five. This modest but targeted change should raise MAP@5 toward the target while preserving the original workflow.'
- What this solution (achieved 0.00237) has done: 'I keep the overall workflow unchanged but make the heuristic a bit more specific: lengthen the filename prefix, add a combined prefix‑suffix lookup, and try that map before falling back to the separate prefix and suffix maps. These tiny changes can capture more exact patterns in the filenames, giving a modest boost in MAP@5 while still writing a valid `submission.csv`.'
- What this solution (achieved 0.00201) has done: 'I reduce the filename prefix and suffix lengths (to 2 characters each) so the heuristic groups many more images together, giving a higher chance of predicting the correct hotel ID and therefore increasing MAP@5 toward the target. The rest of the workflow and prediction logic stays unchanged.'
- What this solution (achieved 0.00237) has done: 'I increase the filename prefix length back to 8 characters (and keep a 3‑character suffix) so the heuristic groups become far more specific, which should raise the MAP@5 score toward the target while leaving the overall workflow unchanged. No other logic is altered.'
- What this solution (achieved 0.00201) has done: 'I shorten the filename prefix and suffix lengths to 2 characters each so that the heuristic groups contain many more images and are far more likely to include the correct hotel ID. The combined‑prefix‑suffix map is removed because it becomes too specific at this short length and would never be hit; keeping only exact, prefix, suffix and global‑top look‑ups provides a smoother, higher‑coverage prediction that should raise MAP@5 toward the target while preserving the original workflow.'

# 9. Code solution

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

    BATCH_SIZE = 256
    NUM_WORKERS = min(8, os.cpu_count() or 1)

    dataset = ImagePathDataset(sampled_paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
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
_test_emb_cache = None  # dict: image_name -> vector
_test_emb_ready = False  # flag to avoid recomputation


def _build_test_embeddings():
    """Compute embeddings for every test image once, using a batch DataLoader."""
    global _test_emb_cache, _test_emb_ready
    if _test_emb_ready:
        return
    test_filenames = [
        f
        for f in os.listdir(TEST_IMG_ROOT)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    test_paths = [os.path.join(TEST_IMG_ROOT, f) for f in test_filenames]

    BATCH_SIZE = 256
    NUM_WORKERS = min(8, os.cpu_count() or 1)
    test_dataset = ImagePathDataset(test_paths)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
    )

    emb_dict = {}
    idx = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            embeddings = backbone(batch).cpu().numpy()
            for vec in embeddings:
                fname = test_filenames[idx]
                emb_dict[fname] = vec
                idx += 1

    _test_emb_cache = emb_dict
    _test_emb_ready = True


def build_prediction(image_name: str) -> str:
    """
    Return a space‑separated list of TOP_N hotel IDs for the given image.
    Uses visual similarity if torch is available; otherwise falls back
    to the original filename‑based heuristic.
    """
    if TORCH_AVAILABLE:
        if not _test_emb_ready:
            _build_test_embeddings()
        query_vec = _test_emb_cache.get(image_name, np.zeros(2048, dtype=np.float32))

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
