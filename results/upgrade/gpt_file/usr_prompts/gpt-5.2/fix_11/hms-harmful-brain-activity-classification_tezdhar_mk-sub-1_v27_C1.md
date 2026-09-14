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
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

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
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Target score

0.3111068021557354

# 6. Current score

0.76513

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I (1) remove the failing offline `pip install` steps and instead run the provided repo using whatever is already available in the Kaggle image, (2) make the parquet→npy conversion and the multi-ckpt inference steps *non-fatal* so the notebook can still finish and always produce a valid `submission.csv`, and (3) add a robust fallback that writes a properly-normalized submission using `sample_submission.csv` priors when the external repo/weights paths are missing. This keeps your intended “use external repo + ensemble folds” core logic when those assets exist, but prevents runtime errors and guarantees a valid submission file in this environment. The fallback is score-worse than a real model, but it ensures you can submit; once the missing `/kaggle/input/hms-mk-codes` and `/kaggle/input/hms-mk-data` datasets are actually attached, the original inference/ensembling path run. Finally, I keep the original ensembling logic intact and only add alignment/normalization safety checks and graceful degradation.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower is better) indicates the fallback “global train prior” is being used or your ensemble predictions are miscalibrated; we make the fallback stronger without changing the modeling/inference core logic. Specifically, we (1) compute the prior at the correct granularity by first aggregating votes per `eeg_id` (train has many overlapping windows per recording), and (2) apply a small Dirichlet/Laplace smoothing to avoid overly-confident zeros and better match KL behavior. We also add a safety fallback to use `sample_submission` column means if train files are unavailable, but keep all paths and the ensemble mechanism intact. This should reduce KL vs the current constant-prior baseline and move you toward the target score.'
- What this solution (achieved 0.80606) has done: 'I fix the failing merge by removing the incorrect `validate="one_to_one"` assumption and instead explicitly de-duplicating `test.csv` and `sample_submission.csv` to one row per `eeg_id` before merging. This keeps the same fallback idea (patient-conditional smoothed priors) but makes it robust to duplicated `eeg_id` keys that can occur depending on the provided files. I also add a small safety alignment step so the final `sol` is guaranteed to have exactly the `sample_submission` `eeg_id` order and valid normalized probabilities, ensuring a valid `submission.csv` is always written. No model/inference core logic is changed.'
- What this solution (achieved 1.03226) has done: 'Your current score (0.80606, lower is better) is far above the target (0.3111), and because the external model assets aren’t being used, your score is dominated by the fallback prior. I keep your overall logic identical (try ensemble predictions, else fallback), but make the fallback materially closer to the evaluation target by (1) learning a patient prior with strength proportional to how many labeled EEGs that patient has (less overconfident for sparse patients), and (2) adding a small “unknown-test” mixture that reserves a little probability mass toward the global prior to reduce KL when a test patient’s distribution differs from their historical labels. These are minimal, metric-aligned calibration changes that don’t change any model architecture/training and still always produce a valid normalized submission.csv.'
- What this solution (achieved 0.9746) has done: 'We keep your “try external ensemble else fallback” flow unchanged, but strengthen the fallback in a metric-aligned way to reduce KL without any model/training changes. The main issue is that a single patient-wide prior can still be miscalibrated for a specific EEG; we add a minimal “conditioning” step using train-derived priors per `(patient_id, expert_consensus)` and per `expert_consensus`, and at inference we softly route each test EEG to the most likely consensus for that patient (mixture-of-priors). This remains a pure prior-based fallback (no new features, no new models), but typically moves KL much closer to the target than patient-only priors. We also keep all normalization/alignment safeguards so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.96065) has done: 'Your current KL (0.9746, lower is better) is far above the target (0.3111), so we should improve the fallback predictions without changing any model/training logic (since the external repo/ckpts aren’t available). I keep your “try external ensemble else fallback” flow identical, but strengthen the fallback by adding a minimal, metric-aligned personalization: build a **patient × consensus × class** prior, and for each test EEG infer a soft consensus mixture using the patient’s historical consensus frequencies; additionally, I add a tiny **patient_id + spectrogram_id** shrinkage term to better distinguish EEGs within the same patient without using raw signals. Finally, I keep your normalization/alignment safeguards and still always write a valid `submission.csv`.'
- What this solution (achieved 0.76513) has done: 'Your current KL (0.96065, lower is better) is far above the target (0.3111), and since the external ckpt inference isn’t running here, the score is dominated by the fallback. I keep your exact flow (“try ensemble else fallback”) but make the fallback closer to the label-generating process by predicting **Dirichlet posterior means** at the right granularity (aggregate train votes per `eeg_id`), using a **patient→(patient,consensus)→consensus→global** hierarchical shrinkage with strength based on evidence, and using the **known number of annotators in test (3–20)** to calibrate concentration. This preserves evaluation semantics (probabilities summing to 1) while improving calibration for KL with minimal code changes. The submission writing, column order, and normalization safeguards remain unchanged.'
- What this solution (achieved 0.76513) has done: 'Your current score is still dominated by the fallback (no external ckpt inference), so the smallest way to move KL down toward the 0.311 target is to make the fallback prior more informative without changing any model/training logic. I keep the same hierarchical Dirichlet-posterior approach, but add one more metadata-only conditioning layer: per-**spectrogram_id** priors learned from train (and a conservative patient×spectrogram shrinkage stays), then at inference we softly shrink test predictions toward the spectrogram prior when that spectrogram has enough evidence in train. This is a minimal, metric-aligned calibration change (still pure priors, still normalized probabilities) and should reduce KL vs the current patient/consensus-only hierarchy. I also keep all submission alignment/normalization safeguards unchanged so it always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import pandas as pd
import numpy as np



## === cell 1
MK_CODES_DIR = "/kaggle/input/hms-mk-codes"
if os.path.exists(MK_CODES_DIR) and MK_CODES_DIR not in sys.path:
    sys.path.append(MK_CODES_DIR)
print("MK_CODES_DIR exists:", os.path.exists(MK_CODES_DIR))



## === cell 2
import subprocess


def _run(cmd: str, check: bool = True):
    """
    Bugfix: allow non-fatal subprocess steps.
    Some Kaggle environments won't have the referenced wheel files / datasets attached.
    """
    print(cmd)
    r = subprocess.run(
        cmd,
        shell=True,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(r.stdout)
    if check and r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, cmd, output=r.stdout)
    return r


print(
    "Skipping offline pip installs because required wheel paths are not guaranteed in this environment."
)



## === cell 3
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

os.makedirs(OUT_PATH, exist_ok=True)

print("DATA_PATH exists:", os.path.exists(DATA_PATH))
print("OUT_PATH exists:", os.path.exists(OUT_PATH))



## === cell 4
convert_ok = False
if os.path.exists(MK_CODES_DIR):
    r = _run(
        f"cd {MK_CODES_DIR} && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}",
        check=False,
    )
    convert_ok = r.returncode == 0
else:
    print(f"External repo not found at {MK_CODES_DIR}; skipping conversion.")
print("convert_ok:", convert_ok)



## === cell 5
print("Listing /kaggle/input/hms-mk-data (if exists):")
if os.path.exists("/kaggle/input/hms-mk-data"):
    print("\n".join(os.listdir("/kaggle/input/hms-mk-data")[:50]))
else:
    print("Path not found: /kaggle/input/hms-mk-data")



## === cell 6
cmds = [
    ("fold0_levit_pseudo.ckpt", "conv1d_tfm2d_pseudo", "submission_fold0_v0.csv"),
    ("fold1_levit_pseudo.ckpt", "conv1d_tfm2d_pseudo", "submission_fold1_v0.csv"),
    ("fold2_levit_pseudo.ckpt", "conv1d_tfm2d_pseudo", "submission_fold2_v0.csv"),
    ("fold3_levit_pseudo.ckpt", "conv1d_tfm2d_pseudo", "submission_fold3_v0.csv"),
    ("fold4_levit_pseudo.ckpt", "conv1d_tfm2d_pseudo", "submission_fold4_v0.csv"),
    ("fold0_pseudo_resv2.ckpt", "conv1d_resv2", "submission_fold0_v2.csv"),
    ("fold1_pseudo_resv2.ckpt", "conv1d_resv2", "submission_fold1_v2.csv"),
    ("fold2_pseudo_resv2.ckpt", "conv1d_resv2", "submission_fold2_v2.csv"),
    ("fold3_pseudo_resv2.ckpt", "conv1d_resv2", "submission_fold3_v2.csv"),
    ("fold4_pseudo_resv2.ckpt", "conv1d_resv2", "submission_fold4_v2.csv"),
]

inference_outputs = []
can_infer = os.path.exists(MK_CODES_DIR) and os.path.exists("/kaggle/input/hms-mk-data")
print("can_infer:", can_infer)

if can_infer:
    for ckpt, experiment, out_name in cmds:
        ckpt_path = f"/kaggle/input/hms-mk-data/{ckpt}"
        if not os.path.exists(ckpt_path):
            print(f"Missing ckpt: {ckpt_path} -> skipping this run.")
            continue

        r = _run(
            "cd /kaggle/input/hms-mk-codes && "
            f"python -m test paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} "
            f"ckpt_path={ckpt_path} hydra=test +model.test_output_dir={OUT_PATH} "
            f"experiment={experiment} +model.net.pretrained=False",
            check=False,
        )

        src_csv = "/kaggle/working/submission.csv"
        dst_csv = f"/kaggle/working/{out_name}"
        if r.returncode == 0 and os.path.exists(src_csv):
            os.replace(src_csv, dst_csv)
            inference_outputs.append(dst_csv)
            print(f"Saved {dst_csv}")
        else:
            print(f"Inference failed for {ckpt} (returncode={r.returncode}).")
else:
    print("External inference assets not available; skipping inference loop.")

print("Produced inference files:", len(inference_outputs))



## === cell 7
if MK_CODES_DIR not in sys.path and os.path.exists(MK_CODES_DIR):
    sys.path.append(MK_CODES_DIR)

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception as e:
    print(f"Warning: could not import TARGET_COLS from src.settings due to: {repr(e)}")
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

TARGET_COLS = list(TARGET_COLS)
print("TARGET_COLS:", TARGET_COLS)


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2"),
    weights=(0.3, 0.7),
    base_submission_path=f"{DATA_PATH}/sample_submission.csv",
):
    """
    Minimal but robust ensembling:
    - Reads each fold/version submission
    - Aligns by eeg_id (prevents row-order mismatch)
    - Weighted sum then renormalize to sum=1 (required by competition)
    """
    if len(versions) != len(weights):
        raise ValueError(
            f"versions and weights must have same length, got {len(versions)} vs {len(weights)}"
        )

    base = pd.read_csv(base_submission_path)
    if "eeg_id" not in base.columns:
        raise ValueError("sample_submission must contain eeg_id")

    base_ids = base[["eeg_id"]].drop_duplicates(subset=["eeg_id"]).copy()

    acc = None
    total_w = 0.0
    used_any = False

    for fold in folds:
        for version, w in zip(versions, weights):
            path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
            if not os.path.exists(path):
                continue  # allow partial availability

            df = pd.read_csv(path)
            missing = ({"eeg_id"} | set(TARGET_COLS)) - set(df.columns)
            if missing:
                raise ValueError(f"{path} missing columns: {missing}")

            df = df[["eeg_id"] + TARGET_COLS].copy()
            df = df.drop_duplicates(subset=["eeg_id"], keep="first")

            df = base_ids.merge(df, on="eeg_id", how="left")

            if df[TARGET_COLS].isna().any().any():
                print(
                    f"Warning: {path} produced NaNs after merging; skipping this prediction file."
                )
                continue

            pred = df[TARGET_COLS].to_numpy(dtype=np.float64)
            if acc is None:
                acc = np.zeros_like(pred)
            acc += pred * float(w)
            total_w += float(w)
            used_any = True

    if not used_any or acc is None or total_w <= 0:
        return None  # no predictions available

    acc /= max(total_w, 1e-12)
    acc = np.clip(acc, 1e-15, 1.0)
    acc = acc / acc.sum(axis=1, keepdims=True)

    out = base_ids.copy()
    out[TARGET_COLS] = acc
    return out


def make_patient_train_prior_submission(
    sample_path=f"{DATA_PATH}/sample_submission.csv",
    train_path=f"{DATA_PATH}/train.csv",
    test_path=f"{DATA_PATH}/test.csv",
    alpha_global: float = 8.0,
    alpha_cons: float = 6.0,
    alpha_patient: float = 4.0,
    alpha_patient_cons: float = 4.0,
    alpha_patient_spec: float = 2.0,
    alpha_spec: float = 3.0,
    k_patient: float = 25.0,
    k_patient_cons: float = 15.0,
    k_patient_spec: float = 10.0,
    k_spec: float = 18.0,
    unknown_mix: float = 0.03,
):
    """
    Prior-based fallback only (no model/training changes).

    Score-directed changes (minimal, metric-aligned):
    - Treat train votes as Dirichlet-multinomial evidence aggregated per eeg_id (correct granularity).
    - Build hierarchical priors: global -> consensus -> patient -> patient×consensus (+weak patient×spectrogram).
    - CHANGE: also learn a spectrogram_id prior from train and shrink test predictions toward it when supported.
    - For each test eeg, produce a posterior mean by shrinking toward broader priors based on evidence size.
    - Calibrate concentration using known test annotator range (3..20) implicitly via alpha_* and k_*.
    """
    sample = pd.read_csv(sample_path)
    sample_ids = sample[["eeg_id"]].drop_duplicates(subset=["eeg_id"]).copy()

    test = pd.read_csv(test_path, usecols=["eeg_id", "patient_id", "spectrogram_id"])
    test = test.drop_duplicates(subset=["eeg_id"], keep="first").copy()

    if not os.path.exists(train_path):
        out = sample_ids.copy()
        col_means = sample[TARGET_COLS].mean(axis=0).to_numpy(dtype=np.float64)
        col_means = np.clip(col_means, 1e-15, None)
        col_means = col_means / col_means.sum()
        for i, c in enumerate(TARGET_COLS):
            out[c] = col_means[i]
        return out

    usecols = [
        "eeg_id",
        "patient_id",
        "spectrogram_id",
        "expert_consensus",
    ] + TARGET_COLS
    train = pd.read_csv(train_path, usecols=usecols)

    per_eeg = train.groupby(
        ["eeg_id", "patient_id", "spectrogram_id", "expert_consensus"], as_index=False
    )[TARGET_COLS].sum()

    def _dirichlet_mean(votes_vec: np.ndarray, alpha: float) -> np.ndarray:
        vv = votes_vec.astype(np.float64, copy=False) + float(alpha)
        vv = np.clip(vv, 1e-15, None)
        return vv / vv.sum()

    v_global = per_eeg[TARGET_COLS].sum(axis=0).to_numpy(dtype=np.float64)
    p_global = _dirichlet_mean(v_global, alpha_global)

    c_votes = per_eeg.groupby("expert_consensus", as_index=False)[TARGET_COLS].sum()
    c_votes["_cons"] = c_votes["expert_consensus"].astype(str)
    cons_list = sorted(c_votes["_cons"].unique().tolist())
    cons_to_prior = {}
    for _, row in c_votes.iterrows():
        cons = str(row["_cons"])
        v = row[TARGET_COLS].to_numpy(dtype=np.float64)
        cons_to_prior[cons] = _dirichlet_mean(v, alpha_cons)

    per_patient_votes = per_eeg.groupby("patient_id", as_index=False)[TARGET_COLS].sum()
    patient_to_prior = {}
    patient_to_strength = {}
    for _, row in per_patient_votes.iterrows():
        pid = int(row["patient_id"])
        v = row[TARGET_COLS].to_numpy(dtype=np.float64)
        patient_to_strength[pid] = float(np.sum(v))
        patient_to_prior[pid] = _dirichlet_mean(v, alpha_patient)

    pc_votes = per_eeg.groupby(["patient_id", "expert_consensus"], as_index=False)[
        TARGET_COLS
    ].sum()
    patient_cons_to_prior = {}
    patient_cons_to_strength = {}
    for _, row in pc_votes.iterrows():
        pid = int(row["patient_id"])
        cons = str(row["expert_consensus"])
        v = row[TARGET_COLS].to_numpy(dtype=np.float64)
        patient_cons_to_strength[(pid, cons)] = float(np.sum(v))
        patient_cons_to_prior[(pid, cons)] = _dirichlet_mean(v, alpha_patient_cons)

    ps_votes = per_eeg.groupby(["patient_id", "spectrogram_id"], as_index=False)[
        TARGET_COLS
    ].sum()
    patient_spec_to_prior = {}
    patient_spec_to_strength = {}
    for _, row in ps_votes.iterrows():
        pid = int(row["patient_id"])
        sid = int(row["spectrogram_id"])
        v = row[TARGET_COLS].to_numpy(dtype=np.float64)
        s = float(np.sum(v))
        if s >= 6.0:
            patient_spec_to_strength[(pid, sid)] = s
            patient_spec_to_prior[(pid, sid)] = _dirichlet_mean(v, alpha_patient_spec)

    spec_votes = per_eeg.groupby("spectrogram_id", as_index=False)[TARGET_COLS].sum()
    spec_to_prior = {}
    spec_to_strength = {}
    for _, row in spec_votes.iterrows():
        sid = int(row["spectrogram_id"])
        v = row[TARGET_COLS].to_numpy(dtype=np.float64)
        s = float(np.sum(v))
        if s >= 8.0:  # conservative threshold to avoid noisy spec priors
            spec_to_strength[sid] = s
            spec_to_prior[sid] = _dirichlet_mean(v, alpha_spec)

    counts = (
        per_eeg.groupby(["patient_id", "expert_consensus"], as_index=False)["eeg_id"]
        .nunique()
        .rename(columns={"eeg_id": "n"})
    )
    cons_index = {c: i for i, c in enumerate(cons_list)}
    C = len(cons_list)
    patient_cons_probs = {}
    if C > 0:
        for pid, grp in counts.groupby("patient_id", sort=False):
            vec = np.full((C,), 0.25, dtype=np.float64)  # mild smoothing
            for cons, n in zip(
                grp["expert_consensus"].astype(str).to_numpy(), grp["n"].to_numpy()
            ):
                j = cons_index.get(str(cons))
                if j is not None:
                    vec[j] += float(n)
            vec = vec / vec.sum()
            patient_cons_probs[int(pid)] = vec

    out = sample_ids.merge(
        test[["eeg_id", "patient_id", "spectrogram_id"]], on="eeg_id", how="left"
    )
    pred = np.zeros((len(out), len(TARGET_COLS)), dtype=np.float64)

    unk = float(unknown_mix)
    unk = min(max(unk, 0.0), 1.0)

    kP = max(float(k_patient), 1e-12)
    kPC = max(float(k_patient_cons), 1e-12)
    kPS = max(float(k_patient_spec), 1e-12)
    kS = max(float(k_spec), 1e-12)

    for i, (pid, sid) in enumerate(
        zip(out["patient_id"].to_numpy(), out["spectrogram_id"].to_numpy())
    ):
        if pd.isna(pid):
            base = p_global
            if not pd.isna(sid):
                sid_int = int(sid)
                p_s = spec_to_prior.get(sid_int)
                if p_s is not None:
                    sS = spec_to_strength.get(sid_int, 0.0)
                    wS = sS / (sS + kS)
                    base = (1.0 - wS) * base + wS * p_s
        else:
            pid_int = int(pid)

            vec = patient_cons_probs.get(pid_int)
            if vec is None or C == 0:
                p_cons_mix = p_global
            else:
                mix_pc = np.zeros((len(TARGET_COLS),), dtype=np.float64)
                for cons, w in zip(cons_list, vec):
                    mix_pc += float(w) * cons_to_prior.get(cons, p_global)
                mix_pc = np.clip(mix_pc, 1e-15, 1.0)
                p_cons_mix = mix_pc / mix_pc.sum()

            sP = patient_to_strength.get(pid_int, 0.0)
            wP = sP / (sP + kP)
            p_patient = patient_to_prior.get(pid_int, p_cons_mix)
            base_patient = (1.0 - wP) * p_cons_mix + wP * p_patient

            if vec is None or C == 0:
                base_pc = base_patient
            else:
                mix_ref = np.zeros((len(TARGET_COLS),), dtype=np.float64)
                tot_w = 0.0
                for cons, w in zip(cons_list, vec):
                    pc = patient_cons_to_prior.get((pid_int, cons))
                    if pc is None:
                        continue
                    s = patient_cons_to_strength.get((pid_int, cons), 0.0)
                    ww = float(w) * (s / (s + kPC))
                    if ww <= 0:
                        continue
                    mix_ref += ww * pc
                    tot_w += ww
                if tot_w > 0:
                    mix_ref = np.clip(mix_ref, 1e-15, 1.0)
                    mix_ref = mix_ref / mix_ref.sum()
                    base_pc = 0.6 * base_patient + 0.4 * mix_ref
                else:
                    base_pc = base_patient

            if pd.isna(sid):
                base = base_pc
            else:
                sid_int = int(sid)
                p_ps = patient_spec_to_prior.get((pid_int, sid_int))
                if p_ps is None:
                    base = base_pc
                else:
                    sPS = patient_spec_to_strength.get((pid_int, sid_int), 0.0)
                    wPS = sPS / (sPS + kPS)
                    base = (1.0 - wPS) * base_pc + wPS * p_ps

                p_s = spec_to_prior.get(sid_int)
                if p_s is not None:
                    sS = spec_to_strength.get(sid_int, 0.0)
                    wS = sS / (sS + kS)
                    base = (1.0 - wS) * base + wS * p_s

        pred[i] = (1.0 - unk) * base + unk * p_global

    pred = np.clip(pred, 1e-15, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    out = out[["eeg_id"]].copy()
    out[TARGET_COLS] = pred
    return out




## === cell 8
ens = merge_preds(folds=(0, 1, 2, 3, 4), versions=("v0", "v2"), weights=(0.3, 0.7))

if ens is None:
    print(
        "No fold predictions found in /kaggle/working; using patient-conditional train-prior fallback submission."
    )
    sol = make_patient_train_prior_submission(
        alpha_global=8.0,
        alpha_cons=6.0,
        alpha_patient=4.0,
        alpha_patient_cons=4.0,
        alpha_patient_spec=2.0,
        alpha_spec=3.0,  # CHANGE: enable spectrogram-level prior (metadata-only)
        k_patient=25.0,
        k_patient_cons=15.0,
        k_patient_spec=10.0,
        k_spec=18.0,  # CHANGE: conservative shrink strength for spec prior
        unknown_mix=0.03,
    )
else:
    sample = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
    sample_ids = sample[["eeg_id"]].drop_duplicates(subset=["eeg_id"]).copy()

    ens = ens.drop_duplicates(subset=["eeg_id"], keep="first")
    sol = sample_ids.merge(ens, on="eeg_id", how="left")

    if sol[TARGET_COLS].isna().any().any():
        print(
            "Warning: NaNs in ensembled predictions after merge; filling with sample_submission values."
        )
        fill = sample[["eeg_id"] + TARGET_COLS].drop_duplicates(
            subset=["eeg_id"], keep="first"
        )
        sol = sol.drop(columns=TARGET_COLS).merge(fill, on="eeg_id", how="left")

    sol = sol[["eeg_id"] + TARGET_COLS].copy()

    p = sol[TARGET_COLS].to_numpy(dtype=np.float64)
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    sol[TARGET_COLS] = p

sample = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
final_ids = sample[["eeg_id"]].copy()
sol = final_ids.merge(sol, on="eeg_id", how="left")

if sol[TARGET_COLS].isna().any().any():
    means = sample[TARGET_COLS].mean(axis=0).to_numpy(dtype=np.float64)
    means = np.clip(means, 1e-15, None)
    means = means / means.sum()
    for j, c in enumerate(TARGET_COLS):
        sol[c] = sol[c].fillna(means[j])

p = sol[TARGET_COLS].to_numpy(dtype=np.float64)
p = np.clip(p, 1e-15, 1.0)
p = p / p.sum(axis=1, keepdims=True)
sol[TARGET_COLS] = p

out_path = "/kaggle/working/submission.csv"
sol.to_csv(out_path, index=False)

assert list(sol.columns) == ["eeg_id"] + TARGET_COLS
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums))
assert np.max(np.abs(row_sums - 1.0)) < 1e-6

print(f"Wrote: {out_path} with shape {sol.shape}")
print(sol.head())



## === cell 9
print("Exists submission.csv:", os.path.exists("/kaggle/working/submission.csv"))
print(
    "File size (bytes):",
    (
        os.path.getsize("/kaggle/working/submission.csv")
        if os.path.exists("/kaggle/working/submission.csv")
        else None
    ),
)
