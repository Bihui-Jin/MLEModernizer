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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.9112965725853808

# 6. Current score

0.59125

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58571) has done: 'I add robust loading that gracefully skips missing submission files, fall back to the provided sample submission, and then compute an average blend of any successfully loaded predictions (or a neutral 0.5 score when none are available). This ensures the script runs end‑to‑end and writes a proper `submission.csv` file without raising errors.'
- What this solution (achieved 0.5826) has done: 'I add a lightweight heuristic that uses each test image’s file size (larger files tend to contain more hidden data) and blend this prediction with any existing model submissions. This keeps the original averaging logic but adds useful signal when few or no model files are found, moving the score upward toward the target.'
- What this solution (achieved 0.5826) has done: 'I add a second size‑based heuristic that normalises each test image’s file size using the global minimum and maximum observed across all available training images (Cover and the three stego folders). This gives a more informative signal than normalising only within the test set, and by appending this extra heuristic dataframe to the list of submissions it receive equal weight in the final average, nudging the blended prediction toward a higher weighted‑AUC score.'
- What this solution (achieved 0.5826) has done: 'We fix the ambiguous DataFrame truth test by tracking model submissions via their object IDs, then use this set to assign weights correctly. This resolves the runtime error and enables proper blending of model predictions (and heuristics), allowing the script to produce a valid `submission.csv` and move the score toward the target.'
- What this solution (achieved 0.58403) has done: 'I keep the overall blending logic but improve the heuristic that uses file size.  Instead of feeding the raw min‑max normalized size, I apply a simple stretching transformation `(norm‑0.5)*2` (clipped to [0,1]) which makes the signal more discriminative.  I also rename the variable for clarity and import numpy.  The rest of the script stays unchanged, so it still writes a valid `submission.csv` while giving a stronger prediction signal that should raise the weighted‑AUC toward the target.'
- What this solution (achieved 0.59125) has done: 'I strengthen the size‑based heuristics and give the global‑size heuristic a higher weight when blending predictions. The raw normalized sizes are squared after stretching to increase contrast, and the global heuristic dataframe is tagged so the blending loop can apply a larger weight (1.5) to it. This small adjustment keeps the overall architecture unchanged while providing a more discriminative signal and nudging the weighted‑AUC closer to the target.'
- What this solution (achieved 0.59125) has done: 'I increase the influence of the size‑based heuristics while keeping the core blending logic unchanged. By raising the default heuristic weight from 0.2 to 0.5 (when model predictions are present) the file‑size signal contributes more, which should push the weighted‑AUC upward toward the target. No other parts of the pipeline are altered.'
- What this solution (achieved 0.59125) has done: 'I lower the weight given to the loaded model predictions (they currently dominate but score poorly) and raise the influence of the size‑based heuristics, especially the global‑size heuristic, so the blended output relies more on the discriminative signal that already improved the score. This small adjustment keeps the overall blending logic unchanged while steering the final predictions toward a higher weighted‑AUC, moving the score closer to the target.'
- What this solution (achieved 0.59125) has done: 'I reduce the influence of all loaded model predictions to zero and recompute the total blend weight purely from the heuristic signals (local size heuristic weight = 1, global size heuristic weight = 2). This keeps the core loading and heuristic generation unchanged while giving the more discriminative size‑based features full control of the final prediction, which should move the weighted‑AUC closer to the target.'
- What this solution (achieved 0.59125) has done: 'I re‑introduce the model predictions into the blended output by assigning them a modest positive weight, and I slightly reduce the dominant weight of the global size heuristic. This keeps the overall blending logic unchanged while allowing the more informative model scores to contribute, which should raise the weighted‑AUC toward the target.'
- What this solution (achieved 0.58403) has done: 'I increase the influence of the model predictions by giving them a higher blending weight (1.0 instead of 0.6) and make the size‑based heuristics more discriminative by using a cubic stretch instead of a square. These minimal changes keep the overall architecture unchanged while providing a stronger signal that should raise the weighted‑AUC toward the target.'
- What this solution (achieved 0.58403) has done: 'I lower the influence of the size‑based heuristic dataframes while keeping the full weight on the actual model predictions. By assigning a small weight (0.2) to any heuristic dataframe (both local‑size and global‑size) and keeping weight = 1.0 for real model submissions, the blended prediction relies much more on the model scores, which should move the weighted‑AUC closer to the target. The change is limited to the weight‑selection logic in cell 1, preserving the overall blending workflow.'
- What this solution (achieved 0.59125) has done: 'I keep the overall blending pipeline but give the size‑based heuristics more influence and reduce the weight of the loaded model predictions. This should push the blended scores toward the more discriminative size signal and raise the weighted‑AUC toward the target. I also soften the cubic stretch to a quadratic stretch, which provides a smoother yet still discriminative transformation of the normalized file‑size feature.'
- What this solution (achieved 0.59125) has done: 'I make the size‑based heuristics more discriminative by raising the stretch exponent from 2 to 4, and I give the heuristics a larger blending weight while dropping the noisy model predictions. This keeps the overall blending pipeline unchanged but should raise the weighted‑AUC toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import os
import pandas as pd

submission_paths = [
    "../input/model-1-fold0/submission_fold0_epoch35.csv",
    "../input/model-2-fold0/submission_fold0_epoch38.csv",
    "../input/model-3-fold0/submission_fold0_epoch39.csv",
    "../input/model-1-fold1/submission_fold1_epoch_30.csv",
    "../input/model-2-fold1/submission_fold1_epoch_34.csv",
    "../input/model-3-fold1/submission_fold1_epoch_38.csv",
    "../input/model-1-fold2/submission_fold2_epoch_34.csv",
    "../input/model-2-fold2/submission_fold2_epoch_37.csv",
    "../input/model-3-fold2/submission_fold2_epoch_39.csv",
    "../input/model-1-fold3/submission_fold3_epoch_35.csv",
    "../input/model-2-fold3/submission_fold3_epoch_36.csv",
    "../input/model-3-fold3/submission_fold3_epoch_39.csv",
    "../input/model-1-fold4/submission_fold4_epoch_33.csv",
    "../input/model-2-fold4/submission_fold4_epoch_35.csv",
    "../input/model-3-fold4/submission_fold4_epoch_36.csv",
    "../input/b3-fold3-m1/submission_fold3_b3_m1.csv",
    "../input/b3-fold3-m2/submission_fold3_b3_m2.csv",
    "../input/b3-fold3-m3/submission_fold3_b3_m3.csv",
    "../input/b3-fold0-m1/submission_fold0_b3_m1.csv",
    "../input/b3-fold0-m2/submission_fold0_b3_m2.csv",
    "../input/b3-fold0-m3/submission_fold0_b3_m3.csv",
    "../input/oc-m1/submission_openclose_m1.csv",
    "../input/oc-m2/submission_openclose_m2.csv",
    "../input/oc-m3/submission_openclose_m3.csv",
]

loaded_subs = []
model_subs = []  # keep track of real model predictions

for path in submission_paths:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if {"Id", "Label"}.issubset(df.columns):
                df = df.sort_values("Id").reset_index(drop=True)
                loaded_subs.append(df)
                model_subs.append(df)  # treat as a model submission
        except Exception:
            pass

if not loaded_subs:
    sample_path = "../input/sample_submission.csv"
    if not os.path.exists(sample_path):
        alternatives = ["sample_submission.csv", "/kaggle/input/sample_submission.csv"]
        for alt in alternatives:
            if os.path.exists(alt):
                sample_path = alt
                break
    sample_df = pd.read_csv(sample_path)
    sample_df = sample_df.sort_values("Id").reset_index(drop=True)
    loaded_subs.append(sample_df)
    model_subs.append(sample_df)

test_dir_candidates = [
    "./input/Test",
    "./kaggle/input/Test",
    "./working/Test",
    "./Test",
    "../input/Test",
    "../working/Test",
]
test_dir = None
for cand in test_dir_candidates:
    if os.path.isdir(cand):
        test_dir = cand
        break

if test_dir is not None:
    file_paths = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".jpg")
    ]
    sizes = [os.path.getsize(p) for p in file_paths]
    if sizes:
        min_sz, max_sz = min(sizes), max(sizes)
        if max_sz > min_sz:
            raw_norm = [(s - min_sz) / (max_sz - min_sz) for s in sizes]
        else:
            raw_norm = [0.5] * len(sizes)
        stretched = np.clip(((np.array(raw_norm) - 0.5) * 2) ** 4, 0, 1)
        heuristic_df = (
            pd.DataFrame(
                {
                    "Id": [os.path.basename(p) for p in file_paths],
                    "Label": stretched,
                }
            )
            .sort_values("Id")
            .reset_index(drop=True)
        )
        loaded_subs.append(heuristic_df)

    train_root_candidates = [
        "./input",
        "./kaggle/input",
        "./working",
        "./",
        "..",
    ]
    train_root = None
    for cand in train_root_candidates:
        if os.path.isdir(os.path.join(cand, "Cover")) and os.path.isdir(
            os.path.join(cand, "JMiPOD")
        ):
            train_root = cand
            break

    if train_root is not None:
        train_dirs = [
            os.path.join(train_root, "Cover"),
            os.path.join(train_root, "JMiPOD"),
            os.path.join(train_root, "JUNIWARD"),
            os.path.join(train_root, "UERD"),
        ]
        global_min, global_max = None, None
        for d in train_dirs:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if not f.lower().endswith(".jpg"):
                    continue
                sz = os.path.getsize(os.path.join(d, f))
                if global_min is None or sz < global_min:
                    global_min = sz
                if global_max is None or sz > global_max:
                    global_max = sz
        if (
            global_min is not None
            and global_max is not None
            and global_max > global_min
        ):
            raw_norm_global = [
                (os.path.getsize(p) - global_min) / (global_max - global_min)
                for p in file_paths
            ]
        else:
            raw_norm_global = [0.5] * len(file_paths)

        stretched_global = np.clip(((np.array(raw_norm_global) - 0.5) * 2) ** 4, 0, 1)

        heuristic_global_df = (
            pd.DataFrame(
                {
                    "Id": [os.path.basename(p) for p in file_paths],
                    "Label": stretched_global,
                }
            )
            .sort_values("Id")
            .reset_index(drop=True)
        )
        heuristic_global_df["source"] = "global"
        loaded_subs.append(heuristic_global_df)



## === cell 1
model_id_set = {id(df) for df in model_subs}

base_df = loaded_subs[0][["Id"]].copy()
label_sum = pd.Series(0.0, index=base_df.index)

weights_list = []  # track each dataframe's blending weight
for df in loaded_subs:
    if id(df) in model_id_set:
        weight = 0.0  # drop noisy model predictions
    else:
        if "source" in df.columns and (df["source"] == "global").any():
            weight = 2.0  # give strong influence to the global‑size heuristic
        else:
            weight = 1.0  # give moderate influence to the local‑size heuristic
    weights_list.append(weight)

    aligned = df.set_index("Id").reindex(base_df["Id"])["Label"]
    label_sum += aligned.values * weight

total_weight = sum(weights_list)

final_label = label_sum / total_weight if total_weight != 0 else 0.5
final_sub = pd.DataFrame({"Id": base_df["Id"], "Label": final_label})

output_path = "submission.csv"
final_sub.to_csv(output_path, index=False)
