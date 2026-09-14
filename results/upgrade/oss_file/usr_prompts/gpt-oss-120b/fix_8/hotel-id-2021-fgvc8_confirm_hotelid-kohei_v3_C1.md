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
import os, subprocess, sys, json, glob, pandas as pd


def run_shell(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()


print("Input directory listing:")
print(run_shell("ls -lha /kaggle/input/"))




## === cell 1
import importlib.util


def try_peko_solution():
    try:
        dep_path = "/kaggle/input/pekolib-deps"
        lib_path = "/kaggle/input/pekolib"
        if os.path.isdir(dep_path):
            run_shell(f"pip -q install {dep_path}/*.whl")
        if os.path.isdir(lib_path):
            run_shell(f"pip -q install {lib_path}/*.whl")
        inplace_dir = "/tmp/inplace_abn"
        if os.path.isdir("/kaggle/input/pekolib-deps/inplace_abn-1.0.12/inplace_abn"):
            run_shell(
                "cp -r /kaggle/input/pekolib-deps/inplace_abn-1.0.12/inplace_abn /tmp/inplace_abn"
            )
            run_shell("cd /tmp/inplace_abn && pip -q install .")
        run_shell("python -m peko.subs.hotelid.v3")
        if os.path.isfile("submission.csv"):
            print("Original Peko solution succeeded, submission.csv created.")
            return True
        else:
            print("Peko run completed but submission.csv not found.")
            return False
    except Exception as e:
        print(f"Peko solution failed with error: {e}")
        return False


_ = try_peko_solution()




## === cell 2
try:
    from PIL import Image
    import numpy as np
    import concurrent.futures
except Exception as e:
    print(f"Pillow not available ({e}), using frequency fallback.")
    train_path = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
    if not os.path.isfile(train_path):
        train_path = "/kaggle/input/train.csv"
    train_df = pd.read_csv(train_path)
    top5_hotels = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
    top5_str = " ".join(top5_hotels)

    test_images_dir = "/kaggle/input/hotel-id-2021-fgvc8/test_images"
    if not os.path.isdir(test_images_dir):
        test_images_dir = "/kaggle/input/test_images"
    test_image_paths = glob.glob(os.path.join(test_images_dir, "*.jpg"))
    test_images = [os.path.basename(p) for p in sorted(test_image_paths)]

    submission_df = pd.DataFrame(
        {"image": test_images, "hotel_id": [top5_str] * len(test_images)}
    )
    submission_df.to_csv("submission.csv", index=False)
    print(f"Fallback submission written with {len(submission_df)} rows.")
else:
    train_path = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
    if not os.path.isfile(train_path):
        train_path = "/kaggle/input/train.csv"
    train_df = pd.read_csv(train_path)
    img2hotel = dict(zip(train_df["image"], train_df["hotel_id"].astype(str)))

    train_images_dir = "/kaggle/input/hotel-id-2021-fgvc8/train_images"
    if not os.path.isdir(train_images_dir):
        train_images_dir = "/kaggle/input/train_images"

    train_image_paths = glob.glob(os.path.join(train_images_dir, "*", "*.jpg"))
    train_image_paths = [
        p for p in train_image_paths if os.path.basename(p) in img2hotel
    ]
    print(f"Found {len(train_image_paths)} training images for feature extraction.")

    train_feat_path = "train_features.npy"
    train_ids_path = "train_hotel_ids.npy"

    def mean_rgb(path):
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((32, 32), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            return arr.mean(axis=(0, 1))  # (3,)
        except Exception:
            return np.array([127.0, 127.0, 127.0], dtype=np.float32)

    if os.path.isfile(train_feat_path) and os.path.isfile(train_ids_path):
        train_features = np.load(train_feat_path).astype(np.float32)
        train_hotel_ids = np.load(train_ids_path).astype(object).tolist()
    else:
        cpu_cnt = min(
            max(1, os.cpu_count() - 1), 8
        )  # limit workers to avoid oversubscription
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cnt) as exe:
            train_features_list = list(exe.map(mean_rgb, train_image_paths))
        train_features = np.stack(train_features_list).astype(np.float32)  # (N,3)
        train_hotel_ids = [img2hotel[os.path.basename(p)] for p in train_image_paths]
        np.save(train_feat_path, train_features)
        np.save(train_ids_path, np.array(train_hotel_ids, dtype=object))

    train_norms = np.sum(train_features**2, axis=1, dtype=np.float32)  # (N,)

    test_images_dir = "/kaggle/input/hotel-id-2021-fgvc8/test_images"
    if not os.path.isdir(test_images_dir):
        test_images_dir = "/kaggle/input/test_images"
    test_image_paths = sorted(glob.glob(os.path.join(test_images_dir, "*.jpg")))
    test_images = [os.path.basename(p) for p in test_image_paths]

    test_feat_path = "test_features.npy"
    if os.path.isfile(test_feat_path):
        test_features = np.load(test_feat_path).astype(np.float32)
    else:
        cpu_cnt = min(max(1, os.cpu_count() - 1), 8)
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cnt) as exe:
            test_features_list = list(exe.map(mean_rgb, test_image_paths))
        test_features = np.stack(test_features_list).astype(np.float32)  # (M,3)
        np.save(test_feat_path, test_features)

    filler_hotels = train_df["hotel_id"].value_counts().index.astype(str).tolist()

    test_norms = np.sum(test_features**2, axis=1, dtype=np.float32)  # (M,)
    batch_size = 1024  # fits comfortably in RAM
    num_test = test_features.shape[0]
    predictions = []

    for start in range(0, num_test, batch_size):
        end = min(start + batch_size, num_test)
        batch_feats = test_features[start:end]  # (B,3)
        batch_norms = test_norms[start:end]  # (B,)

        dists = (
            train_norms[:, None]
            + batch_norms[None, :]
            - 2.0 * train_features.dot(batch_feats.T)
        )

        topk_idx = np.argpartition(dists, 5, axis=0)[:5, :]  # (5, B)

        for col in range(end - start):
            idxs = topk_idx[:, col]
            ordered = idxs[np.argsort(dists[idxs, col])]
            top5_hotels = [train_hotel_ids[i] for i in ordered]

            seen = set()
            ordered_unique = []
            for h in top5_hotels:
                if h not in seen:
                    ordered_unique.append(h)
                    seen.add(h)
                if len(ordered_unique) == 5:
                    break
            if len(ordered_unique) < 5:
                for f in filler_hotels:
                    if f not in seen:
                        ordered_unique.append(f)
                        seen.add(f)
                    if len(ordered_unique) == 5:
                        break
            predictions.append(" ".join(ordered_unique))

        if (end) % 1000 == 0 or end == num_test:
            print(f"Processed {end}/{num_test} test images")

    submission_df = pd.DataFrame({"image": test_images, "hotel_id": predictions})
    submission_df.to_csv("submission.csv", index=False)
    print(
        f"Generated submission with {len(submission_df)} rows using colour‑NN fallback."
    )
