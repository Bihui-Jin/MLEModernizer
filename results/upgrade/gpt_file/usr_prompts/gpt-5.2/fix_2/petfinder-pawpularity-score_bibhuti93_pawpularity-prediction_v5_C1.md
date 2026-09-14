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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

29.37556771172349

# 6. Current score

64.38149

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 64.38149) has done: 'I make the code run end-to-end by removing the broken external `imquality` dependency and its missing Kaggle input, and instead compute a lightweight image-quality proxy score directly with Pillow (no extra installs). I also fix the missing imports/NameErrors by consolidating imports and ensuring the submission file is created deterministically (single-process write, correct header, correct columns, `.csv` suffix). The core approach remains the same: generate a per-image “quality score” from the test photos and submit it as `Pawpularity`. This reliably produce `/kaggle/working/submission.csv` in the exact required format.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings
import json
import multiprocessing

import numpy as np
import pandas as pd
from PIL import Image, ImageStat, ImageFilter

warnings.filterwarnings("ignore")

TEST_IMG_GLOB = "/kaggle/input/petfinder-pawpularity-score/test/*.jpg"
SUB_TMP_PATH = "/kaggle/working/submission1.csv"
SUB_PATH = "/kaggle/working/submission.csv"




## === cell 1
def get_pet_category(image_file_name):
    raise NotImplementedError(
        "get_pet_category() requires external DL dependencies and is not used for submission generation."
    )




## === cell 2


def _quality_proxy_score(img: Image.Image) -> float:
    """
    Returns a score in [0, 100] based on sharpness/contrast/brightness heuristics.
    Deterministic and fast; no external ML libs required.
    """
    img = img.convert("RGB")
    gray = img.convert("L")

    gray_small = gray.resize((256, 256), resample=Image.Resampling.BILINEAR)

    edges = gray_small.filter(ImageFilter.FIND_EDGES)
    e = np.asarray(edges, dtype=np.float32)
    edge_var = float(e.var())

    g = np.asarray(gray_small, dtype=np.float32)
    contrast = float(g.std())

    brightness = float(g.mean())
    brightness_penalty = abs(brightness - 128.0)  # 0 best

    score = 0.18 * edge_var + 0.35 * contrast - 0.12 * brightness_penalty

    score = max(0.0, min(100.0, score))
    return float(round(score, 2))


def get_image_quality_score(full_path: str):
    img = Image.open(full_path)
    score = _quality_proxy_score(img)
    filename = os.path.basename(full_path)
    return {"Id": filename[:-4], "Pawpularity": score}




## === cell 3

file_list = sorted(glob.glob(TEST_IMG_GLOB))
if len(file_list) == 0:
    raise FileNotFoundError(f"No test images found with glob: {TEST_IMG_GLOB}")

results = []
try:
    with multiprocessing.Pool(processes=min(10, os.cpu_count() or 1)) as pool:
        for row in pool.imap_unordered(
            get_image_quality_score, file_list, chunksize=16
        ):
            results.append(row)
except Exception:
    results = [get_image_quality_score(p) for p in file_list]

df_sub = pd.DataFrame(results)

test_df = pd.read_csv(
    "/kaggle/input/petfinder-pawpularity-score/test.csv", usecols=["Id"]
)
df_sub = test_df.merge(df_sub, on="Id", how="left")

if df_sub["Pawpularity"].isna().any():
    df_sub["Pawpularity"] = df_sub["Pawpularity"].fillna(df_sub["Pawpularity"].median())

df_sub["Pawpularity"] = df_sub["Pawpularity"].clip(0, 100)



## === cell 4

if os.path.exists(SUB_TMP_PATH):
    try:
        os.remove(SUB_TMP_PATH)
    except OSError:
        pass

df_sub.to_csv(SUB_PATH, index=False)

print("Wrote submission to:", SUB_PATH)
print(df_sub.head())
print("Rows:", len(df_sub), "Cols:", df_sub.shape[1])
assert os.path.exists(SUB_PATH) and SUB_PATH.endswith(".csv")
assert list(df_sub.columns) == ["Id", "Pawpularity"]
assert len(df_sub) == len(
    pd.read_csv("/kaggle/input/petfinder-pawpularity-score/sample_submission.csv")
)
