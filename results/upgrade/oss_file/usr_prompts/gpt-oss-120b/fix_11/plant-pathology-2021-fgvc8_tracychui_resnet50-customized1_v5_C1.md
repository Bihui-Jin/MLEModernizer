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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.4458910433979671

# 6. Current score

0.28887

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'The script failed because it looked for the sample submission file in a hard‑coded “input” folder that doesn’t exist in this environment, which caused all downstream variables to be undefined. I added a small helper that searches recursively for the correct sample_submission.csv inside the *plant‑pathology‑2021‑fgvc8* directory, built the test image list from it, and then creates a baseline “healthy” prediction file. Unused image‑loading code was removed to keep the flow simple and error‑free, ensuring a valid submission.csv is written.'
- What this solution (achieved 0.38173) has done: 'I keep the existing data‑search helper, read the training labels to discover the most frequent disease classes, and use the top three of them as a simple multi‑label baseline for every test image. Predicting a few common classes improves recall across the board, which should raise the macro F1 toward the target while preserving the overall structure of the script.'
- What this solution (achieved 0.24507) has done: 'I add a lightweight image‑based heuristic that predicts “healthy” for test images whose average green channel is high and otherwise predicts the second most common disease label. This keeps the original top‑label logic but introduces per‑image variation, which should raise precision for the dominant “healthy” class and improve the macro F1, moving the score toward the target. The rest of the script – loading paths, counting labels, and writing the CSV – remains unchanged.'
- What this solution (achieved 0.24178) has done: 'I adjust the prediction logic so that images with a low green channel (likely diseased) receive the multi‑label baseline consisting of the most frequent disease classes instead of a single secondary label. This adds relevant labels for many images, improving recall and thus moving the macro F1 score upward toward the target while keeping the overall structure unchanged.'
- What this solution (achieved 0.29247) has done: 'I compute a data‑driven green‑channel threshold from a sample of training images (using the known “healthy” label) and apply a simple red‑vs‑green rule in `predict_label`. This keeps the original baseline‑plus‑healthy logic while making the threshold more appropriate, which should raise the macro F1 toward the target without altering the overall workflow.'
- What this solution (achieved 0.2892) has done: 'I add a simple red‑channel heuristic to complement the existing green‑channel rule: compute a red‑channel threshold from a small sample of training images and, when an image is below the green threshold, optionally add the “rust” label if its red mean exceeds this threshold. This small change should boost recall for a common disease class and move the macro F1 score closer to the target without altering the overall workflow.'
- What this solution (achieved 0.28887) has done: 'The changes speed up image processing by replacing costly NumPy array conversions with Pillow’s fast `ImageStat` means and parallelizing the test‑set predictions using a thread pool, while keeping all thresholds, heuristics, and label logic exactly the same.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageStat  # use ImageStat for fast channel statistics

print("Libraries loaded successfully.")



## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3




## === cell 2
def find_file(filename: str) -> str | None:
    """
    Recursively search for *filename* inside any directory that contains
    'plant-pathology-2021-fgvc8' and return the first match.
    """
    pattern = os.path.join("**", filename)
    candidates = glob.glob(pattern, recursive=True)
    for p in candidates:
        if "plant-pathology-2021-fgvc8" in p:
            return p
    return None


sample_path = find_file("sample_submission.csv")
if sample_path is None:
    raise FileNotFoundError(
        "sample_submission.csv not found in the dataset directories."
    )
sample_df = pd.read_csv(sample_path)
test_img_names = sample_df["image"].tolist()

test_img_dir = os.path.join(os.path.dirname(sample_path), "test_images")
test_img_paths = [os.path.join(test_img_dir, name) for name in test_img_names]

print(f"Found {len(test_img_names)} test images listed in sample submission.")
print(f"Sample submission path: {sample_path}")



## === cell 3
train_path = find_file("train.csv")
if train_path is None:
    raise FileNotFoundError("train.csv not found in the dataset directories.")
train_df = pd.read_csv(train_path)

from collections import Counter

label_counter = Counter()
for lbls in train_df["labels"]:
    for lbl in str(lbls).split():
        label_counter[lbl] += 1

TOP_N = 5
top_labels = [lbl for lbl, _ in label_counter.most_common(TOP_N)]
baseline_pred = " ".join(top_labels)

print(f"Top {TOP_N} frequent labels from training data: {top_labels}")
print(f"Using baseline prediction: '{baseline_pred}' for all test images (fallback).")

primary_label = "healthy" if "healthy" in top_labels else top_labels[0]
secondary_label = top_labels[1] if len(top_labels) > 1 else primary_label

train_img_dir = os.path.join(os.path.dirname(train_path), "train_images")
healthy_greens = []
diseased_greens = []
healthy_reds = []
diseased_reds = []
MAX_SAMPLES = 2000

for idx, (img_name, lbls) in enumerate(zip(train_df["image"], train_df["labels"])):
    if idx >= MAX_SAMPLES:
        break
    img_path = os.path.join(train_img_dir, img_name)
    try:
        img = Image.open(img_path).convert("RGB")
        stats = ImageStat.Stat(img)
        green_mean = stats.mean[1]  # channel 1 = G
        red_mean = stats.mean[0]  # channel 0 = R
        if "healthy" in str(lbls).split():
            healthy_greens.append(green_mean)
            healthy_reds.append(red_mean)
        else:
            diseased_greens.append(green_mean)
            diseased_reds.append(red_mean)
    except Exception:
        continue

if healthy_greens and diseased_greens:
    THRESHOLD = (np.mean(healthy_greens) + np.mean(diseased_greens)) / 2
else:
    THRESHOLD = 120  # fallback empirical value

if healthy_reds and diseased_reds:
    RED_THRESHOLD = (np.mean(healthy_reds) + np.mean(diseased_reds)) / 2
else:
    RED_THRESHOLD = 130  # fallback empirical value

print(f"Computed green‑channel threshold: {THRESHOLD:.2f}")
print(f"Computed red‑channel threshold (rust heuristic): {RED_THRESHOLD:.2f}")



## === cell 4
BASELINE_SET = set(baseline_pred.split())


def predict_label(image_path: str) -> str:
    """
    Predict labels for a single image.

    - If the average green channel is higher than THRESHOLD, predict the primary label (usually 'healthy').
    - Otherwise start from the multi‑label baseline.
      * If the average red channel exceeds RED_THRESHOLD, add the 'rust' label.
      * Always add the 'complex' label for diseased samples to capture multi‑disease cases.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        stats = ImageStat.Stat(img)
        green_mean = stats.mean[1]
        red_mean = stats.mean[0]
        if green_mean > THRESHOLD:
            return primary_label
        else:
            labels = set(BASELINE_SET)  # copy baseline
            if red_mean > RED_THRESHOLD:
                labels.add("rust")
            labels.add("complex")
            return " ".join(sorted(labels))
    except Exception:
        return baseline_pred


from concurrent.futures import ThreadPoolExecutor, as_completed


def parallel_predictions(paths):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(predict_label, paths))


test_predtags = parallel_predictions(test_img_paths)

df_submission = pd.DataFrame({"image": test_img_names, "labels": test_predtags})
print("Submission preview:")
print(df_submission.head())



## === cell 5
output_path = "submission.csv"
df_submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
