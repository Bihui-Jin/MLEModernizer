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
import subprocess, sys, os, glob

deps_path = "/kaggle/input/pekolib-deps"
if os.path.isdir(deps_path):
    for whl in glob.glob(os.path.join(deps_path, "*.whl")):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", whl])

pk_path = "/kaggle/input/pekolib"
if os.path.isdir(pk_path):
    for whl in glob.glob(os.path.join(pk_path, "*.whl")):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", whl])

abn_src = "/kaggle/input/pekolib-deps/inplace_abn-1.0.12/inplace_abn"
if os.path.isdir(abn_src):
    tmp_dir = "/tmp/inplace_abn"
    subprocess.run(["cp", "-r", abn_src, tmp_dir])
    subprocess.run(
        ["bash", "-c", f"cd {tmp_dir} && {sys.executable} -m pip install -q ."]
    )

print("Package installation completed.")



## === cell 1
import pandas as pd
from pathlib import Path
import hashlib
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np

train_path = next(Path("/kaggle/input").rglob("train.csv"))
sample_sub_path = next(Path("/kaggle/input").rglob("sample_submission.csv"))
train_df = pd.read_csv(train_path)

train_img_roots = list(Path("/kaggle/input").rglob("train_images"))
test_img_roots = list(Path("/kaggle/input").rglob("test_images"))
if not train_img_roots or not test_img_roots:
    raise FileNotFoundError("Could not find train_images or test_images directories.")

max_per_hotel = 2
hotel_to_paths = {}
for root in train_img_roots:
    for img_path in root.rglob("*.jpg"):
        img_name = img_path.name
        row = train_df[train_df["image"] == img_name]
        if row.empty:
            continue
        hotel_id = str(row.iloc[0]["hotel_id"])
        lst = hotel_to_paths.setdefault(hotel_id, [])
        if len(lst) < max_per_hotel:
            lst.append(img_path)

device = torch.device("cpu")
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # output 512‑dim embeddings
model.to(device)
model.eval()

transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def embed_image(p):
    try:
        img = Image.open(p).convert("RGB")
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model(t).squeeze(0)  # 512‑dim
        return vec.cpu().numpy()
    except Exception:
        return None


hotel_ids = []
centroids = []
for hid, paths in hotel_to_paths.items():
    vecs = []
    for p in paths:
        v = embed_image(p)
        if v is not None:
            vecs.append(v)
    if vecs:
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        hotel_ids.append(hid)
        centroids.append(centroid)

centroids = np.stack(centroids)  # shape (num_hotels, 512)

test_img_path = {}
for root in test_img_roots:
    for img_path in root.rglob("*.jpg"):
        test_img_path[img_path.name] = img_path


def get_top5_for_image(img_name):
    if img_name not in test_img_path:
        return " ".join(global_top5)
    vec = embed_image(test_img_path[img_name])
    if vec is None:
        return " ".join(global_top5)
    vec = vec / np.linalg.norm(vec)  # normalize
    sims = centroids @ vec  # cosine similarity
    top_idx = np.argpartition(-sims, 5)[:5]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    top_hotels = [hotel_ids[i] for i in top_idx]
    if len(top_hotels) < 5:
        for h in global_top5:
            if h not in top_hotels:
                top_hotels.append(h)
            if len(top_hotels) == 5:
                break
    return " ".join(top_hotels)


global_top5 = train_df["hotel_id"].value_counts().nlargest(5).index.astype(str).tolist()



## === cell 2
sample_sub = pd.read_csv(sample_sub_path)

submission = pd.DataFrame(
    {
        "image": sample_sub["image"],
        "hotel_id": sample_sub["image"].apply(get_top5_for_image),
    }
)

output_path = Path("/kaggle/working/submission.csv")
submission.to_csv(output_path, index=False)

print(f"Submission written to {output_path}")
print("Sample rows:")
print(submission.head())
