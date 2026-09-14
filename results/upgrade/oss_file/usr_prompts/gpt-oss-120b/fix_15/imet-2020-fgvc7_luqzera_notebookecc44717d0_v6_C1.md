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
Label artwork images with significant attributes.

## Metric
Micro averaged F1 score.

## Submission Format
```
id,attribute_ids
00011f01965f141f5d1eea6592fa9862,0 1 2
00014abc91ed3e4bf1663fde8136fe80,0 1 2
0002e2054e303badc1a33463f6fb7973,0 1 2
```

## Dataset
Multiple modalities can be expected and the camera sources are unknown. The photographs are often centered for objects, and in the case where the museum artifact is an entire room, the images are scenic in nature.

Each object is annotated by a single annotator without a verification step. You should consider these annotations noisy.

The filename of each image is its `id`.

- **train.csv** gives the `attribute_ids` for the train images in **/train**
- **/test** contains the test images. You must predict the `attribute_ids` for these images.
- **sample_submission.csv** contains a submission in the correct format
- **labels.csv** provides descriptions of the attributes

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
        input/
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
        working/
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
```

-> data/imet-2020-fgvc7/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/imet-2020-fgvc7/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/imet-2020-fgvc7/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> data/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> (stopped after 10 files for performance)

# 5. Target score

0.4693840214058197

# 6. Current score

0.17182

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00373) has done: 'The changes add a fast‑path for inference: we simplify image loading by using Pillow directly (removing the slow OpenCV‑to‑Pillow conversion), and we enable parallel data loading and pinned memory in the prediction DataLoader. A small guard is added so training only runs when `should_train` is set, keeping the original training logic untouched. These tweaks keep all model architecture and training behavior identical while dramatically reducing I/O overhead, allowing the script to finish well within the 600‑second limit.'
- What this solution (achieved 0.1669) has done: 'The changes add a small calibration to the frequency‑based baseline: the number K of predicted attributes per image is increased by a few units (capped by the total class count) to boost recall, which typically raises micro‑F1 and moves the score toward the target. The submission file is also written to the current working directory (`submission.csv`) so a valid CSV is always produced regardless of the script’s launch location. No model architecture or training logic is altered.'
- What this solution (achieved 0.1631) has done: 'The update slightly raises the calibrated number of predicted attributes per image (K) from `avg_labels + 2` to `avg_labels + 10`. This increases recall, which should improve the micro‑averaged F1 score and move the result closer to the target while keeping the original frequency‑based baseline intact.'
- What this solution (achieved 0.1706) has done: 'We add a lightweight validation split to tune the number K of top‑frequency attributes that are predicted for each image. By searching a small range of K values and picking the one that maximizes micro‑F1 on a held‑out part of the training data, we keep the original frequency‑based baseline while improving recall‑precision balance, moving the score toward the target without altering any model architecture.'
- What this solution (achieved 0.17182) has done: 'Implemented missing imports, reorganized cells for proper execution order, and ensured all variables are defined before use. Added a broader search range for K (up to avg_labels + 200) to improve micro‑F1 while keeping the original frequency‑based logic unchanged. The script now correctly writes a `submission.csv` file in the working directory.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

kaggle = True
should_train = False  # training disabled to avoid long runtime
TRAIN_EPOCHS = 5  # kept for compatibility if training is ever enabled
input_dir = "../input/"
base_dir = "../input/imet-2020-fgvc7/"
model_path_for_test = "best-model.pt"
train_root = base_dir + "train"
number_of_classes = 3474
num_workers = 4
batch_size = 32




## === cell 1
def _micro_f1(tp: int, fp: int, fn: int) -> float:
    """Return micro‑averaged F1 from global TP/FP/FN counts."""
    if tp + fp + fn == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def _evaluate_k(
    k: int,
    top_k_indices: np.ndarray,
    val_df: pd.DataFrame,
) -> float:
    """Compute micro‑F1 for a given K on the validation set."""
    tp = fp = fn = 0
    pred_set = set(top_k_indices.tolist())
    for attr_str in val_df["attribute_ids"]:
        if pd.isna(attr_str) or attr_str == "":
            true_set = set()
        else:
            true_set = set(map(int, attr_str.split()))
        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    return _micro_f1(tp, fp, fn)




## === cell 2
def test():
    """
    Frequency‑based baseline with calibrated K.
    K is selected by a quick validation split to maximise micro‑F1,
    then the same K is used for the final test predictions.
    """
    from sklearn.model_selection import train_test_split

    train_csv_path = os.path.join(base_dir, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, shuffle=True
    )

    attr_counts = np.zeros(number_of_classes, dtype=np.float32)
    label_counts = []
    for attr_str in train_split["attribute_ids"]:
        if pd.isna(attr_str) or attr_str == "":
            label_counts.append(0)
            continue
        ids = list(map(int, attr_str.split()))
        label_counts.append(len(ids))
        attr_counts[ids] += 1

    avg_labels = max(1, int(round(np.mean(label_counts))))

    max_k = min(number_of_classes, avg_labels + 200)
    candidate_K = list(range(max(1, avg_labels), max_k + 1, 1))

    best_k = avg_labels  # fallback if loop fails
    best_score = -1.0

    sorted_indices = np.argsort(-attr_counts)

    for k in candidate_K:
        top_k = sorted_indices[:k]
        score = _evaluate_k(k, top_k, val_split)
        if score > best_score:
            best_score = score
            best_k = k

    final_K = best_k
    top_k_indices = sorted_indices[:final_K]
    attr_str = " ".join(map(str, top_k_indices.tolist()))

    test_img_dir = os.path.join(base_dir, "test")
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    test_ids = [
        os.path.splitext(fname)[0]
        for fname in os.listdir(test_img_dir)
        if os.path.splitext(fname)[1].lower() in valid_ext
    ]

    rows = [{"id": img_id, "attribute_ids": attr_str} for img_id in test_ids]
    submission_df = pd.DataFrame(rows, columns=["id", "attribute_ids"])
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Best K found: {final_K} (validation micro‑F1 ≈ {best_score:.5f})")
    print(f"Submission written to {submission_path}")




## === cell 3
if __name__ == "__main__":
    test()
