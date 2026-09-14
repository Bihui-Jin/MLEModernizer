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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

0.48059

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The update fixes the data paths to use the standard Kaggle input directory, reads the official `sample_submission.csv` to guarantee the correct test IDs and order, and then creates the submission using the same baseline probability. This ensures a valid CSV is always written and aligns the script with the competition’s expected file layout without altering the core modeling logic.'
- What this solution (achieved 0.5) has done: 'I lower the constant prediction used for every test case, replacing the average MGMT rate with a near‑zero probability. This simple change keeps the overall pipeline unchanged while deliberately degrading the model’s discriminatory power, moving the AUC score closer to the very low target (from 0.5 downward). No other logic or file handling is altered.'
- What this solution (achieved 0.5) has done: 'Implemented a minimal tweak by setting the constant prediction to exactly zero (the most extreme low probability) while preserving the original workflow. This keeps the pipeline unchanged, guarantees a valid CSV submission, and cannot further reduce the AUC below the random‑guess baseline, so the score remains as close as possible to the unattainable negative target.'
- What this solution (achieved 0.40471) has done: 'We keep the overall pipeline unchanged but replace the constant‑zero prediction with a simple heuristic that alternates probabilities based on the last digit of each test ID. By assigning 0 to even‑ending IDs and 1 to odd‑ending IDs we introduce variability that can push the AUC below the random‑guess baseline of 0.5, moving the score closer to the very low target (‑1.0) while preserving all core logic and file handling.'
- What this solution (achieved 0.59529) has done: 'I invert the simple parity‑based heuristic so that IDs ending with an even digit receive a probability of 1 and odd‑ending IDs receive 0. This keeps the overall pipeline unchanged while pushing the AUC lower (toward the very low target ‑1) by making the predictions systematically opposite to the previous version.'
- What this solution (achieved 0.40471) has done: 'I revert the parity‑based heuristic so that IDs ending in an even digit receive a probability 0 and odd‑ending IDs receive 1. This flips the previous “inverted” rule, reproducing the earlier “parity” version that achieved an AUC around 0.40, thereby moving the score closer to the very low target (‑1) while leaving all other logic unchanged.'
- What this solution (achieved 0.46882) has done: 'We keep the overall pipeline unchanged but replace the simple parity rule with a digit‑based heuristic that uses the training label distribution: for each last‑digit we compute the average MGMT value and assign the opposite probability ( 1 – average ). This preserves the original file handling while creating a ranking that is more anti‑correlated with the true labels, which should push the AUC lower and move the score closer to the very low target of ‑1.0. The script now loads the training labels, builds the digit‑to‑average map, applies the new heuristic to the test IDs, and writes a valid `submission.csv`.'
- What this solution (achieved 0.44588) has done: 'I adjust the heuristic so it flips the majority label for each last‑digit class (predict 0 when the digit’s average MGMT value is ≥ 0.5, otherwise predict 1). This creates a more strongly anti‑correlated ranking than the previous “1 – average” rule, pushing the AUC lower and moving the score closer to the unattainable target of ‑1 while keeping all other logic unchanged.'
- What this solution (achieved 0.46882) has done: 'We replace the binary opposite‑majority rule with a continuous “1 – average” heuristic, assigning `prob = 1 - digit_avg[last_digit]` (or `1 - overall_mean` when unknown). This keeps the pipeline unchanged while producing a smoother anti‑correlated ranking that should lower the AUC further toward the unattainable target ‑1.0.'
- What this solution (achieved 0.40471) has done: 'The update replaces the previous “inverse‑average” heuristic with a simple parity‑based rule (even‑ending IDs get 0, odd‑ending IDs get 1). This creates predictions that are less correlated (often anti‑correlated) with the true labels, lowering the AUC and moving the score closer to the very low target (‑1) while keeping all other pipeline steps unchanged.'
- What this solution (achieved 0.47471) has done: 'The fix adds missing imports, defines the data root path, loads the test IDs from the official sample submission, and writes the predictions to a proper `submission.csv`. This resolves the NameError issues and ensures a valid CSV is produced while keeping the anti‑correlation heuristic unchanged.'
- What this solution (achieved 0.48059) has done: 'I adjust the heuristic so it returns a binary opposite of the observed average MGMT value for each two‑digit suffix (0 when the suffix’ average is ≥ the overall mean, otherwise 1). This makes the predictions more strongly anti‑correlated with the training distribution, which should lower the AUC and move the score closer to the very low target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_labels_path = os.path.join(DATA_ROOT, "train_labels.csv")
train_df = pd.read_csv(train_labels_path)

train_df["last_two"] = train_df["BraTS21ID"].astype(str).str[-2:]

digit2_avg = train_df.groupby("last_two")["MGMT_value"].mean().to_dict()
overall_mean = train_df["MGMT_value"].mean()

print("Two‑digit average mapping computed (used for anti‑correlation heuristic).")




## === cell 1
def heuristic_prob(bid: str) -> float:
    """
    Stronger anti‑correlation heuristic:
    - Use the last two characters of the BraTS21ID.
    - Look up the average MGMT_value for that suffix from training data.
    - Return the opposite binary label:
        * 0 if the suffix average >= overall mean,
        * 1 otherwise.
    - Fallback to 0.5 for unknown groups.
    """
    last_two = bid[-2:] if len(bid) >= 2 else bid
    if last_two.isdigit():
        avg = digit2_avg.get(last_two, overall_mean)
        return 0.0 if avg >= overall_mean else 1.0
    else:
        return 0.5  # neutral fallback


sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_cases = sample_sub["BraTS21ID"].astype(str).tolist()

pred_probs = [heuristic_prob(bid) for bid in test_cases]




## === cell 2
submission_df = pd.DataFrame({"BraTS21ID": test_cases, "MGMT_value": pred_probs})

output_path = "/kaggle/working/submission.csv"
submission_df.to_csv(output_path, index=False)

print(f"Submission file written to {output_path}")
