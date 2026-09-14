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
import subprocess, os, sys, pandas as pd, hashlib

BASE_PATH = "/kaggle/input/hotel-id-2021-fgvc8"


def list_dir(path):
    try:
        subprocess.run(["ls", "-lha", path], check=True)
    except Exception as e:
        print(f"Unable to list {path}: {e}")


list_dir(BASE_PATH)



## === cell 1
train_path = os.path.join(BASE_PATH, "train.csv")
train_df = pd.read_csv(train_path)

global_top5 = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
print("Global top‑5 hotel_id values:", global_top5)

chain_top5 = {}
for chain_id, grp in train_df.groupby("chain"):
    top5 = grp["hotel_id"].value_counts().head(5).index.astype(str).tolist()
    chain_top5[int(chain_id)] = top5
print(f"Computed top‑5 hotels for {len(chain_top5)} chains.")

prefix_top5 = {}
for prefix, grp in train_df.groupby(train_df["image"].str[:4]):
    top5 = grp["hotel_id"].value_counts().head(5).index.astype(str).tolist()
    prefix_top5[prefix] = top5
print(f"Computed prefix‑based top‑5 for {len(prefix_top5)} prefixes.")

chain_prefix_top5 = {}
for (chain_id, prefix), grp in train_df.groupby(["chain", train_df["image"].str[:4]]):
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

train_chain_dirs = [d for d in os.listdir(train_images_root) if d.isdigit()]
for chain in train_chain_dirs:
    chain_path = os.path.join(train_images_root, chain)
    for img_name in os.listdir(chain_path):
        img_path = os.path.join(chain_path, img_name)
        h = md5_hash(img_path)
        if h:
            hotel_id = train_df.loc[train_df["image"] == img_name, "hotel_id"]
            if not hotel_id.empty:
                train_md5_to_hotel[h] = str(hotel_id.iloc[0])

print(f"Built MD5 map for {len(train_md5_to_hotel)} training images.")



## === cell 2
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
    prefix = img[:4]

    chain_id = infer_chain(img)

    md5 = get_image_md5(img, chain_id)
    matched_hotel = None
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



## === cell 3
print(pd.read_csv(output_path).head())
