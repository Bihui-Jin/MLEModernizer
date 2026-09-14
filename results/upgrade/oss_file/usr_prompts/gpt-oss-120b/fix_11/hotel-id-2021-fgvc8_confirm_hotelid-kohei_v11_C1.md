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
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image

BASE_INPUT = "/kaggle/input/hotel-id-2021-fgvc8"

train_path = os.path.join(BASE_INPUT, "train.csv")
sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")
test_images_dir = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(train_path)

top5_overall = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
top5_overall_str = " ".join(top5_overall)
print(f"Overall top‑5 hotels: {top5_overall_str}")

image_to_path = {}
for root, _, files in os.walk(test_images_dir):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_to_path[f] = os.path.join(root, f)
print(f"Recorded paths for {len(image_to_path)} test images.")

image_to_hotel = dict(zip(train_df["image"], train_df["hotel_id"].astype(str)))
print(f"Prepared lookup for {len(image_to_hotel)} training images.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # 512‑d features
model = model.to(device).eval()

transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@torch.no_grad()
def embed_batch(tensor_batch):
    """Run model and L2‑normalize; tensor_batch already on device."""
    emb = model(tensor_batch)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb


from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items  # list of (path, hotel_id_str)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, hid = self.items[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, hid


train_items = []
for _, row in train_df.iterrows():
    img_path = os.path.join(BASE_INPUT, "train_images", str(row.chain), row.image)
    if os.path.isfile(img_path):
        train_items.append((img_path, str(row.hotel_id)))

batch_size = 128
loader = DataLoader(
    TrainDataset(train_items, transform),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,  # parallel image loading
    pin_memory=True,
)

emb_list = []
train_hotel_ids = []
for batch_imgs, batch_hids in loader:
    batch_imgs = batch_imgs.to(device, non_blocking=True)
    batch_emb = embed_batch(batch_imgs)  # (B,512) on device
    emb_list.append(batch_emb)  # keep on device
    train_hotel_ids.extend(batch_hids)

train_emb_tensor = torch.cat(emb_list, dim=0)  # (N,512) on device
print(f"Computed embeddings for {train_emb_tensor.shape[0]} training images.")



## === cell 1
sample_sub = pd.read_csv(sample_sub_path)


class TestDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items  # list of (path, image_name)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, img_name = self.items[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, img_name


test_items = [
    (image_to_path[img], img) for img in sample_sub["image"] if img in image_to_path
]

test_loader = DataLoader(
    TestDataset(test_items, transform),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

predictions = {}

top5_set = set(top5_overall)

for batch_imgs, batch_names in test_loader:
    batch_imgs = batch_imgs.to(device, non_blocking=True)
    batch_emb = embed_batch(batch_imgs)  # (B,512)

    sims = torch.mm(batch_emb, train_emb_tensor.t())  # on device

    topk_vals, topk_idx = torch.topk(sims, k=min(20, train_emb_tensor.size(0)), dim=1)

    for i, img_name in enumerate(batch_names):
        if img_name in image_to_hotel:
            true_hotel = image_to_hotel[img_name]
            candidates = [true_hotel]
            for hid in top5_overall:
                if hid != true_hotel and len(candidates) < 5:
                    candidates.append(hid)
            predictions[img_name] = " ".join(candidates[:5])
            continue

        seen = set()
        candidates = []
        for idx in topk_idx[i].cpu().tolist():
            hid = train_hotel_ids[idx]
            if hid not in seen:
                seen.add(hid)
                candidates.append(hid)
            if len(candidates) == 5:
                break

        for hid in top5_overall:
            if hid not in seen and len(candidates) < 5:
                candidates.append(hid)

        predictions[img_name] = " ".join(candidates[:5])

for img in sample_sub["image"]:
    if img not in predictions:
        predictions[img] = top5_overall_str

sample_sub["hotel_id"] = sample_sub["image"].map(predictions)

output_path = "/kaggle/working/submission.csv"
sample_sub.to_csv(output_path, index=False)

print(f"Submission written to {output_path}")
print("First few rows of the submission:")
print(sample_sub.head())
