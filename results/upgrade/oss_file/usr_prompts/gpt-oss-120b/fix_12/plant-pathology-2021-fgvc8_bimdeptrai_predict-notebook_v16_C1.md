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

0.791098799630657

# 6. Current score

0.34001

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I remove the unused and incompatible tensorflow_addons import, replace the missing model file with a simple placeholder prediction (all‑zeros), and correct the logic that builds the submission (use assignment instead of comparison and handle the case where no disease meets the threshold). These minimal fixes let the notebook run end‑to‑end and generate a valid `submission.csv` file.'
- What this solution (achieved 0.28656) has done: 'The fix removes the TensorFlow import that triggers a protobuf‑related error and replaces the dummy zero predictions with class‑frequency based priors derived from the training data, providing a more sensible baseline that should raise the F1 score toward the target. Only the import cell and the prediction‑initialisation cell are changed; all other logic and file handling remains untouched.'
- What this solution (achieved 0.28656) has done: 'I keep the existing data loading and frequency‑based prediction logic, but replace the threshold‑based label selection with a simple top‑k rule (k=2). This predicts the two most frequent classes for every test image (unless “healthy” is the highest‑scoring class, in which case we output “healthy”). Selecting a small fixed number of likely diseases should raise recall and improve the mean F1‑score, moving the current 0.28656 toward the target 0.7911 while preserving the core baseline approach.'
- What this solution (achieved 0.38173) has done: 'I keep the overall workflow unchanged but increase the number of predicted classes per image from 2 to 3 and remove the special‑case handling that forces a single “healthy” label. This lets the baseline frequency‑based model output a richer, more recall‑oriented label set, which is expected to raise the mean F1‑Score toward the target without altering the core logic.'
- What this solution (achieved 0.3327) has done: 'I adjust the baseline probability scores to give rarer classes a higher chance by weighting each class frequency with (1‑frequency). Then I increase the number of labels predicted per image from 3 to 5 so the model captures more possible diseases, which should raise recall and improve the mean F1‑Score, moving the current 0.38173 closer to the target 0.7911. The core workflow and file handling remain unchanged.'
- What this solution (achieved 0.38173) has done: 'I replace the adjusted‑frequency weighting with the raw class‑frequency scores (which better reflect the true label distribution) and add a simple healthy‑threshold rule: if the “healthy” probability is high we output only “healthy”, otherwise we output the top 3 predicted classes. This keeps the overall baseline workflow unchanged while improving the balance between precision and recall, moving the mean F1‑Score closer to the target.'
- What this solution (achieved 0.30565) has done: 'I adjust the baseline probabilities to give rarer diseases a relatively higher chance by applying a square‑root transform to the class‑frequency scores, and then select all classes whose score exceeds a modest cutoff (instead of a fixed k). This keeps the overall frequency‑based approach while improving recall, which should raise the mean F1 toward the target without altering the core workflow.'
- What this solution (achieved 0.3327) has done: 'I increase the baseline prediction by normalising the square‑root‑adjusted class frequencies into a probability vector and then, for each test image, output the top 5 classes (unless “healthy” is very likely, in which case we output “healthy” only). This keeps the original frequency‑based approach but adds a simple top‑k rule that should raise recall and improve the mean F1‑Score, moving the current 0.30565 closer to the target 0.7911.'
- What this solution (achieved 0.31456) has done: 'I slightly change how the class‑frequency vector is transformed (use a 0.6‑power instead of a square‑root) and reduce the number of top‑k predictions to 4 while lowering the “healthy” confidence threshold to 0.4. These minimal tweaks keep the overall frequency‑based baseline intact but should raise recall without adding many false positives, moving the mean F1 closer to the target score.'
- What this solution (achieved 0.29775) has done: 'I slightly increase the emphasis on rare disease classes by using a 0.5 power when adjusting class frequencies, raise the “healthy” confidence threshold to 0.6 so that we only predict “healthy” when it is very likely, and predict the top 5 diseases otherwise. These minimal tweaks keep the overall frequency‑based workflow intact while encouraging higher recall and a better mean F1‑Score, moving the current 0.31456 closer to the target 0.7911.'
- What this solution (achieved 0.34001) has done: 'I tighten the baseline by using the raw class‑frequency distribution (no power transformation), lower the “healthy” confidence threshold to 0.5, and predict only the top 3 classes per image. These minimal adjustments keep the original workflow but should raise recall while limiting false positives, moving the F1 score upward toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer




## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")




## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")




## === cell 3
h_target = 384
w_target = 384
batch_size = 32




## === cell 4
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(label_split)
labels_onehot = pd.DataFrame(mlb.transform(label_split), columns=mlb.classes_)

num_classes = labels_onehot.shape[1]

class_freq = labels_onehot.mean().values.astype(np.float32)  # (num_classes,)

prob_vector = class_freq / class_freq.sum()

preds = np.tile(prob_vector, (len(submissions), 1))




## === cell 5
healthy_threshold = 0.5  # slightly more permissive than before
healthy_idx = np.where(labels_onehot.columns == "healthy")[0]
if healthy_idx.size == 0:
    healthy_idx = None
else:
    healthy_idx = healthy_idx[0]

k = 3  # predict fewer labels to improve precision
for i in range(len(submissions)):
    if healthy_idx is not None and preds[i, healthy_idx] >= healthy_threshold:
        top_labels = ["healthy"]
    else:
        top_k_idx = np.argpartition(preds[i], -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(-preds[i][top_k_idx])]
        top_labels = [labels_onehot.columns[idx] for idx in top_k_idx]
        if "healthy" in top_labels and (
            healthy_idx is None or preds[i, healthy_idx] < healthy_threshold
        ):
            top_labels = [lbl for lbl in top_labels if lbl != "healthy"]
    submissions.at[i, "labels"] = " ".join(top_labels)




## === cell 6
submissions.to_csv("submission.csv", index=False)
