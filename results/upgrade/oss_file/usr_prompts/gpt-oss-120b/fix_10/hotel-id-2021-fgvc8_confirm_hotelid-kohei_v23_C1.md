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

0.7712701018775774

# 6. Current score

0.00235

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the external‑package commands with a self‑contained baseline that builds a valid `submission.csv`. The new code loads the training metadata, extracts the five most common `hotel_id`s, and writes those five IDs for every test image listed in the provided `sample_submission.csv`. This guarantees a correctly formatted CSV and yields a non‑zero MAP@5 score, moving the result toward the target without altering any core modeling logic.'
- What this solution (achieved 0.00209) has done: 'I keep the original data loading steps, compute the five most common `hotel_id`s globally and also the five most common per hotel `chain`. For each test image I infer its chain by checking the folder hierarchy under `test_images`; if the chain is found I use that chain‑specific list, otherwise I fall back to the global top‑5. This simple per‑chain heuristic should raise the MAP@5 score much closer to the target while preserving the overall structure of the script.'
- What this solution (achieved 0.00139) has done: 'I add a lightweight prefix‑based heuristic to the existing simple top‑5 logic: for each image I first look up the most frequent hotel IDs that share the same filename prefix in the training set, then fall back to the chain‑specific top‑5 and finally to the global top‑5, padding with global IDs when needed. This keeps the overall structure unchanged while giving more relevant predictions, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00139) has done: 'I add a more specific heuristic that first tries a (chain, prefix) based top‑5 list, then falls back to the chain‑specific top‑5, and finally to the global top‑5. This small change keeps the overall pipeline unchanged while giving the model a better chance of ranking the correct hotel ID higher, so the MAP@5 score should move closer to the target.'
- What this solution (achieved 0.00139) has done: 'The changes parallelize the expensive MD5 computation for all training images and replace the per‑image pandas lookup with a fast dictionary, drastically cutting I/O‑bound work while keeping every lookup and prediction identical. A dictionary of image‑to‑hotel IDs is built once, and a multiprocessing pool computes MD5 hashes concurrently, filling the same `train_md5_to_hotel` map used later. All other logic stays unchanged, preserving exact results.'
- What this solution (achieved 0.00139) has done: 'I add a direct filename‑to‑hotel lookup (using the training metadata) before the MD5 lookup and keep the existing chain‑based fall‑backs. This cheap addition can correctly predict many images that appear in both train and test sets, raising MAP@5 toward the target while preserving all other logic.'
- What this solution (achieved 0.00235) has done: 'I added the missing imports, defined `BASE_PATH` to point to the competition data folder, and ensured all required libraries (`os`, `pandas`, `hashlib`, `multiprocessing`) are available. The rest of the logic is unchanged, so the script now runs end‑to‑end, builds the necessary lookup tables, generates predictions, and writes a correctly formatted `submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import hashlib
import multiprocessing as mp

BASE_PATH = "/kaggle/input/hotel-id-2021-fgvc8"

train_path = os.path.join(BASE_PATH, "train.csv")
train_df = pd.read_csv(train_path)

global_top5 = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
print("Global top‑5 hotel_id values:", global_top5)

chain_top5 = {}
for chain_id, grp in train_df.groupby("chain"):
    top5 = grp["hotel_id"].value_counts().head(5).index.astype(str).tolist()
    chain_top5[int(chain_id)] = top5
print(f"Computed top‑5 hotels for {len(chain_top5)} chains.")

PREFIX_LEN = 6

prefix_top5 = {}
for prefix, grp in train_df.groupby(train_df["image"].str[:PREFIX_LEN]):
    top5 = grp["hotel_id"].value_counts().head(5).index.astype(str).tolist()
    prefix_top5[prefix] = top5
print(f"Computed prefix‑based top‑5 for {len(prefix_top5)} prefixes.")

chain_prefix_top5 = {}
for (chain_id, prefix), grp in train_df.groupby(
    ["chain", train_df["image"].str[:PREFIX_LEN]]
):
    top5 = grp["hotel_id"].value_counts().head(5).index.astype(str).tolist()
    chain_prefix_top5[(int(chain_id), prefix)] = top5
print(f"Computed chain‑prefix top‑5 for {len(chain_prefix_top5)} combinations.")


def md5_hash(file_path):
    """Return hex MD5 of a file, reading in chunks."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None
    return hash_md5.hexdigest()


train_images_root = os.path.join(BASE_PATH, "train_images")
train_md5_to_hotel = {}

image_to_hotel = dict(zip(train_df["image"], train_df["hotel_id"].astype(str)))

train_chain_dirs = [d for d in os.listdir(train_images_root) if d.isdigit()]

all_img_paths = []
for chain in train_chain_dirs:
    chain_path = os.path.join(train_images_root, chain)
    for img_name in os.listdir(chain_path):
        img_path = os.path.join(chain_path, img_name)
        all_img_paths.append((img_path, img_name))


def process_path(args):
    img_path, img_name = args
    h = md5_hash(img_path)
    if h:
        hotel_id = image_to_hotel.get(img_name)
        if hotel_id:
            return (h, hotel_id)
    return None


cpu = max(1, mp.cpu_count() - 1)
with mp.Pool(cpu) as pool:
    for result in pool.imap_unordered(process_path, all_img_paths, chunksize=100):
        if result:
            h, hotel_id = result
            train_md5_to_hotel[h] = hotel_id

print(f"Built MD5 map for {len(train_md5_to_hotel)} training images.")




## === cell 1
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_images_root = os.path.join(BASE_PATH, "test_images")
chain_dirs = [d for d in os.listdir(test_images_root) if d.isdigit()]

image_to_chain = {}
for chain in chain_dirs:
    chain_path = os.path.join(test_images_root, chain)
    for img_name in os.listdir(chain_path):
        image_to_chain[img_name] = int(chain)


def infer_chain(image_name):
    """Return the chain id (int) if known, else None."""
    return image_to_chain.get(image_name, None)


def get_image_md5(image_name, chain_id):
    """Compute MD5 of a test image given its name and optional chain."""
    possible_paths = []
    if chain_id is not None:
        possible_paths.append(os.path.join(test_images_root, str(chain_id), image_name))
    else:
        for ch in chain_dirs:
            possible_paths.append(os.path.join(test_images_root, ch, image_name))
    for p in possible_paths:
        if os.path.exists(p):
            return md5_hash(p)
    return None


predictions = []
for img in sample_sub["image"]:
    prefix = img[:PREFIX_LEN]  # use the same longer prefix as during training

    matched_hotel = image_to_hotel.get(img)

    chain_id = infer_chain(img)

    if not matched_hotel:
        md5 = get_image_md5(img, chain_id)
        if md5 and md5 in train_md5_to_hotel:
            matched_hotel = train_md5_to_hotel[md5]

    hotels = None
    if chain_id is not None:
        hotels = chain_prefix_top5.get((chain_id, prefix))

    if not hotels and chain_id is not None:
        hotels = chain_top5.get(chain_id)

    if not hotels:
        hotels = prefix_top5.get(prefix)

    if not hotels:
        hotels = global_top5

    hotels = [str(h) for h in hotels]

    final_list = []
    if matched_hotel:
        final_list.append(matched_hotel)

    for h in hotels:
        if h not in final_list:
            final_list.append(h)
        if len(final_list) == 5:
            break

    if len(final_list) < 5:
        for h in global_top5:
            if h not in final_list:
                final_list.append(h)
            if len(final_list) == 5:
                break

    predictions.append(" ".join(final_list[:5]))

sample_sub["hotel_id"] = predictions

output_path = "/kaggle/working/submission.csv"
sample_sub.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")




## === cell 2
print(pd.read_csv(output_path).head())
