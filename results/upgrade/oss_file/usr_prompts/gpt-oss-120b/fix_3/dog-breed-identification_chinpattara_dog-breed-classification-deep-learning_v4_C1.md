# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
        input/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
            test/
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
            train/
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> working/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> working/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

# 5. Code solution

## === cell 0
import os, pathlib
import numpy as np
import pandas as pd
import PIL.Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss




## === cell 1
base_path = "/kaggle/input/dog-breed-identification"
train_img_dir = os.path.join(base_path, "train")
test_img_dir = os.path.join(base_path, "test")
labels_path = os.path.join(base_path, "labels.csv")
sample_sub_path = os.path.join(base_path, "sample_submission.csv")




## === cell 2
label_df = pd.read_csv(labels_path)




## === cell 3
le = LabelEncoder()
label_df["label_enc"] = le.fit_transform(label_df["breed"])
num_classes = len(le.classes_)




## === cell 4
label_df["img_path"] = label_df["id"].apply(
    lambda x: os.path.join(train_img_dir, f"{x}.jpg")
)




## === cell 5
def preprocess_image(path, size=(128, 128)):
    """Load an image, resize, normalize, and flatten to a 1‑D vector."""
    img = PIL.Image.open(path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()




## === cell 6
X = np.stack(label_df["img_path"].apply(preprocess_image))
y = label_df["label_enc"].values




## === cell 7
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)




## === cell 8
clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500,  # allow more iterations for convergence
    C=10.0,  # weaker regularization for richer model
    n_jobs=-1,
    verbose=0,
)
clf.fit(X_train, y_train)




## === cell 9
valid_pred = clf.predict_proba(X_valid)
print("Validation LogLoss:", log_loss(y_valid, valid_pred))




## === cell 10
test_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(".jpg")])
test_ids = [os.path.splitext(f)[0] for f in test_files]
test_paths = [os.path.join(test_img_dir, f) for f in test_files]

X_test = np.stack([preprocess_image(p) for p in test_paths])




## === cell 11
test_pred = clf.predict_proba(X_test)  # shape (num_test, num_classes)




## === cell 12
sample_sub = pd.read_csv(sample_sub_path)
breed_cols = sample_sub.columns.tolist()[1:]  # exclude 'id'

submission = pd.DataFrame(test_pred, columns=le.classes_)
submission = submission[breed_cols]  # ensure correct column order
submission.insert(0, "id", test_ids)




## === cell 13
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, shape: {submission.shape}")
