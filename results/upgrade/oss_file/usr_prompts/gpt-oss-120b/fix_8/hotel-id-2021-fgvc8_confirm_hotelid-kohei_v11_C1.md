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

0.7607266144649304

# 6. Current score

0.0014

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'We replace the previous shell‑only steps with a small pure‑Python baseline: compute the five most frequent hotel IDs in the training data and assign that same list to every test image. This guarantees a correctly formatted `submission.csv`, allowing the notebook to finish and produce a concrete MAP@5 score that moves the result from “no score” toward the target.'
- What this solution (achieved 0.00209) has done: 'I keep the original logic of using the most frequent hotels but make the prediction chain‑aware: compute the top‑5 hotels for each chain from the training data, discover each test image’s chain by scanning the test_images folder, and assign the chain‑specific top‑5 list (falling back to the overall top‑5 when a chain is missing). This adds only a few lightweight steps and should raise the MAP@5 score toward the target while still producing a correctly formatted submission.csv.'
- What this solution (achieved 0.00209) has done: 'I add a direct image‑to‑hotel lookup so that when a test image happens to be present in the training set we can return its exact hotel ID (plus the most frequent others to keep five predictions). For all other images the existing chain‑aware top‑5 logic is kept. This tiny heuristic can raise the MAP@5 without altering the core modelling approach.'
- What this solution (achieved 0.00209) has done: 'I fix the chain‑folder mapping so each test image is correctly associated with its hotel chain (the previous walk used the deepest folder name, often “test_images”, causing most predictions to fall back to the overall top‑5). By extracting the first sub‑directory under `test_images` we obtain the true chain id and then look up the chain‑specific top‑5 hotels. This small change keeps the original logic while making the chain‑aware predictions apply to many more images, moving the MAP@5 score much closer to the target.'
- What this solution (achieved 0.00209) has done: 'I extend the chain‑specific top‑5 list with the overall most frequent hotels whenever it contains fewer than five IDs, and make sure the final prediction always has exactly five IDs (removing duplicates). This keeps the original heuristic but improves coverage, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.0014) has done: 'I make the chain‑extraction more robust by scanning the full path of each test image for the first numeric folder name (the chain id). When no chain is found I fall back to using the “unknown” chain 0 (if present) rather than the generic overall top‑5 list. This keeps the original heuristic intact while giving many more images a chain‑specific top‑5, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.0014) has done: 'I add a lightweight visual similarity step using a pretrained ResNet‑18 to find the nearest training image (by embedding) for each test picture. The nearest‑neighbor hotel ID becomes the first prediction, while the remaining slots are filled with the existing chain‑aware or overall top‑5 lists, keeping the original heuristic intact. This extra information should substantially raise MAP@5 toward the target without altering the core logic.'

# 9. Code solution

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

chain_top5_series = train_df.groupby("chain")["hotel_id"].apply(
    lambda s: s.value_counts().head(5).index.astype(str).tolist()
)
chain_top5 = chain_top5_series.to_dict()  # {chain_id: [hotel_id,...]}


def extract_chain_id_from_path(path_parts):
    for part in path_parts:
        try:
            return int(part)
        except ValueError:
            continue
    return None


image_to_chain = {}
image_to_path = {}
for root, _, files in os.walk(test_images_dir):
    rel_path = os.path.relpath(root, test_images_dir)
    parts = rel_path.split(os.sep)
    chain_id = extract_chain_id_from_path(parts)
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(root, f)
            image_to_path[f] = full_path
            if chain_id is not None:
                image_to_chain[f] = chain_id
print(f"Mapped {len(image_to_chain)} test images to chain ids.")
print(f"Recorded paths for {len(image_to_path)} test images.")

image_to_hotel = dict(zip(train_df["image"], train_df["hotel_id"].astype(str)))
print(f"Prepared lookup for {len(image_to_hotel)} training images.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # output 512‑d features
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
def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    emb = model(tensor)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # cosine‑ready
    return emb.squeeze(0)  # (512,)


rep_rows = train_df.drop_duplicates(subset="hotel_id")
rep_paths = [
    os.path.join(BASE_INPUT, "train_images", str(row.chain), row.image)
    for _, row in rep_rows.iterrows()
]

train_emb_list = []
train_hotel_ids = []
for path, hid in zip(rep_paths, rep_rows["hotel_id"].astype(str)):
    try:
        emb = get_embedding(path)
        train_emb_list.append(emb.cpu())
        train_hotel_ids.append(hid)
    except Exception as e:
        continue

train_emb_tensor = torch.stack(train_emb_list)  # (N,512)
print(f"Computed embeddings for {train_emb_tensor.shape[0]} hotel representatives.")



## === cell 1
sample_sub = pd.read_csv(sample_sub_path)


def get_top5_for_image(image_name: str) -> str:
    """Return a space‑separated list of exactly 5 hotel IDs for the given image."""

    if image_name in image_to_hotel:
        true_hotel = image_to_hotel[image_name]
        candidates = [true_hotel]
        for hid in top5_overall:
            if hid != true_hotel and len(candidates) < 5:
                candidates.append(hid)
        return " ".join(candidates)

    chain_id = image_to_chain.get(image_name, None)
    if chain_id is None and 0 in chain_top5:
        chain_id = 0

    if chain_id is not None and chain_id in chain_top5:
        chain_candidates = list(chain_top5[chain_id])
        for hid in top5_overall:
            if len(chain_candidates) >= 5:
                break
            if hid not in chain_candidates:
                chain_candidates.append(hid)
        return " ".join(chain_candidates[:5])

    img_path = image_to_path.get(image_name, None)
    if img_path is not None:
        try:
            test_emb = get_embedding(img_path)  # (512,)
            sims = torch.mm(train_emb_tensor, test_emb.unsqueeze(1)).squeeze(1)  # (N,)
            best_idx = torch.argmax(sims).item()
            nearest_hotel = train_hotel_ids[best_idx]
            candidates = [nearest_hotel]
            for hid in top5_overall:
                if hid != nearest_hotel and len(candidates) < 5:
                    candidates.append(hid)
            return " ".join(candidates)
        except Exception:
            pass  # fall back to overall top‑5 if anything goes wrong

    return top5_overall_str


sample_sub["hotel_id"] = sample_sub["image"].apply(get_top5_for_image)

output_path = "/kaggle/working/submission.csv"
sample_sub.to_csv(output_path, index=False)

print(f"Submission written to {output_path}")
print("First few rows of the submission:")
print(sample_sub.head())
