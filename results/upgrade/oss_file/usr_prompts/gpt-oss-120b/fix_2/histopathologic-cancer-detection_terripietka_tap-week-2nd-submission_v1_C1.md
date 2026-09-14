# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.6788606691978132

# 6. Current score

0.45758

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.45758) has done: 'We avoid the TensorFlow import that crashes, replace the model loading with a simple baseline that uses the average intensity of the central 32×32 pixel region as a probability score, and adjust the prediction‑generation code accordingly. This eliminates all runtime errors, ensures a valid submission.csv with the correct columns, and provides a reasonable AUC baseline that moves the score toward the target without altering the core competition logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    tf_available = True
except Exception as e:
    tf_available = False
    print("TensorFlow import failed (", e, "). Proceeding with a fallback model.")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_images = "/kaggle/input/histopathologic-cancer-detection/test/"

test_df = pd.read_csv(
    "/kaggle/input/histopathologic-cancer-detection/sample_submission.csv"
)
test_df["id_tif"] = test_df["id"] + ".tif"

print("Test Set Size:", test_df.shape)
test_df.head()



## === cell 2
sample_images = test_df.sample(16, random_state=42)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
fig.tight_layout(pad=1.0)

for i, ax in enumerate(axes.flat):
    img_id = sample_images.iloc[i]["id_tif"]
    label = sample_images.iloc[i]["label"]
    img_path = os.path.join(test_images, img_id)
    img = mpimg.imread(img_path)

    ax.imshow(img, cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")

plt.show()




## === cell 3
def central_region_mean(image: np.ndarray) -> float:
    """Return the mean pixel value of the central 32×32 region."""
    h, w = image.shape[:2]
    top = (h - 32) // 2
    left = (w - 32) // 2
    crop = image[top : top + 32, left : left + 32]
    if crop.ndim == 3:
        crop = crop.mean(axis=2)
    return crop.mean()


probabilities = []
for idx, row in test_df.iterrows():
    img_path = os.path.join(test_images, row["id_tif"])
    img = mpimg.imread(img_path)

    if img.dtype == np.uint8 or img.max() > 1.0:
        img = img.astype(np.float32) / 255.0

    mean_val = central_region_mean(img)
    prob = np.clip(mean_val, 0.0, 1.0)
    probabilities.append(prob)

probabilities = np.array(probabilities)



## === cell 4
submission = pd.DataFrame({"id": test_df["id"], "label": probabilities})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
print(submission.head())



## === cell 5
plt.figure(figsize=(6, 4))
plt.hist(submission["label"], bins=20, color="steelblue", edgecolor="black")
plt.title("Distribution of Predicted Probabilities")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.show()
