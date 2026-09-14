# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Use binary leaf images and extracted features to identify the species of plant.

## Metric
Multi-class log loss. 

The submitted probabilities for a given device are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum), but they need to be in the range of [0, 1]. In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the image id, all candidate species names, and a probability for each species. The order of the rows does not matter. The file must have a header and should look like the following:

id,Acer_Capillipes,Acer_Circinatum,Acer_Mono,...
2,0.1,0.5,0,0.2,...
5,0,0.3,0,0.4,...
6,0,0,0,0.7,...
etc.

## Dataset
The dataset consists of images of leaf specimens which have been converted to binary black leaves against white backgrounds. 

Three sets of features are also provided per image: a shape contiguous descriptor, an interior texture histogram, and a ﬁne-scale margin histogram. 

For each feature, a 64-attribute vector is given per leaf sample.

### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- **images/** - the image files (each image is named with its corresponding id)

### Data fields
- **id** - an anonymous id unique to an image
- **margin_1, margin_2, margin_3, ..., margin_64** - each of the 64 attribute vectors for the margin feature
- **shape_1, shape_2, shape_3, ..., shape_64** - each of the 64 attribute vectors for the shape feature
- **texture_1, texture_2, texture_3, ..., texture_64** - each of the 64 attribute vectors for the texture feature

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        input/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        working/
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
```

-> data/leaf-classification/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/leaf-classification/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/leaf-classification/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> data/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

train_path = "../input/train.csv"
test_path = "../input/test.csv"
sample_sub_path = "../input/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

ids = train_df.pop("id")  # keep ids if needed later
y = train_df.pop("species")  # target column
y_enc = LabelEncoder().fit_transform(y)  # integer encoding for modelling
label_encoder = LabelEncoder()
label_encoder.fit(y)  # for inverse mapping later

std_scaler = StandardScaler()
X = std_scaler.fit_transform(train_df)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=12345)
train_idx, val_idx = next(sss.split(X, y_enc))
x_train, x_val = X[train_idx], X[val_idx]
y_train, y_val = y_enc[train_idx], y_enc[val_idx]

c_options = [0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

models = []  # store trained models
val_preds = []  # store their validation predictions
val_losses = []  # store validation log‑losses
best_logloss = np.inf
best_model = None
best_name = ""

for C in c_options:
    logreg = LogisticRegression(
        C=C,
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=5,
        random_state=12345,
        class_weight="balanced",
    )
    logreg.fit(x_train, y_train)
    val_pred = logreg.predict_proba(x_val)
    val_loss = -np.mean(
        np.log(np.take_along_axis(val_pred, y_val[:, None], axis=1) + 1e-15)
    )
    models.append(logreg)
    val_preds.append(val_pred)
    val_losses.append(val_loss)
    if val_loss < best_logloss:
        best_logloss = val_loss
        best_model = logreg
        best_name = f"LogisticRegression (C={C})"

lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
lda_val_pred = lda.predict_proba(x_val)
lda_val_loss = -np.mean(
    np.log(np.take_along_axis(lda_val_pred, y_val[:, None], axis=1) + 1e-15)
)
models.append(lda)
val_preds.append(lda_val_pred)
val_losses.append(lda_val_loss)
if lda_val_loss < best_logloss:
    best_logloss = lda_val_loss
    best_model = lda
    best_name = "LinearDiscriminantAnalysis"

gbc = GradientBoostingClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=3,
    random_state=12345,
)
gbc.fit(x_train, y_train)
gbc_val_pred = gbc.predict_proba(x_val)
gbc_val_loss = -np.mean(
    np.log(np.take_along_axis(gbc_val_pred, y_val[:, None], axis=1) + 1e-15)
)
models.append(gbc)
val_preds.append(gbc_val_pred)
val_losses.append(gbc_val_loss)
if gbc_val_loss < best_logloss:
    best_logloss = gbc_val_loss
    best_model = gbc
    best_name = "GradientBoostingClassifier"

gbc2 = GradientBoostingClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=3,
    random_state=12345,
)
gbc2.fit(x_train, y_train)
gbc2_val_pred = gbc2.predict_proba(x_val)
gbc2_val_loss = -np.mean(
    np.log(np.take_along_axis(gbc2_val_pred, y_val[:, None], axis=1) + 1e-15)
)
models.append(gbc2)
val_preds.append(gbc2_val_pred)
val_losses.append(gbc2_val_loss)
if gbc2_val_loss < best_logloss:
    best_logloss = gbc2_val_loss
    best_model = gbc2
    best_name = "GradientBoostingClassifier (n=1200, lr=0.03)"

gbc3 = GradientBoostingClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=3,
    random_state=12345,
)
gbc3.fit(x_train, y_train)
gbc3_val_pred = gbc3.predict_proba(x_val)
gbc3_val_loss = -np.mean(
    np.log(np.take_along_axis(gbc3_val_pred, y_val[:, None], axis=1) + 1e-15)
)
models.append(gbc3)
val_preds.append(gbc3_val_pred)
val_losses.append(gbc3_val_loss)
if gbc3_val_loss < best_logloss:
    best_logloss = gbc3_val_loss
    best_model = gbc3
    best_name = "GradientBoostingClassifier (n=2000, lr=0.02)"

hgb = HistGradientBoostingClassifier(
    max_iter=1000,
    learning_rate=0.03,
    max_depth=3,
    random_state=12345,
)
hgb.fit(x_train, y_train)
hgb_val_pred = hgb.predict_proba(x_val)
hgb_val_loss = -np.mean(
    np.log(np.take_along_axis(hgb_val_pred, y_val[:, None], axis=1) + 1e-15)
)
models.append(hgb)
val_preds.append(hgb_val_pred)
val_losses.append(hgb_val_loss)
if hgb_val_loss < best_logloss:
    best_logloss = hgb_val_loss
    best_model = hgb
    best_name = "HistGradientBoostingClassifier"

hgb2 = HistGradientBoostingClassifier(
    max_iter=1500,
    learning_rate=0.03,
    max_depth=3,
    random_state=12345,
)
hgb2.fit(x_train, y_train)
hgb2_val_pred = hgb2.predict_proba(x_val)
hgb2_val_loss = -np.mean(
    np.log(np.take_along_axis(hgb2_val_pred, y_val[:, None], axis=1) + 1e-15)
)
models.append(hgb2)
val_preds.append(hgb2_val_pred)
val_losses.append(hgb2_val_loss)
if hgb2_val_loss < best_logloss:
    best_logloss = hgb2_val_loss
    best_model = hgb2
    best_name = "HistGradientBoostingClassifier (max_iter=1500)"

loss_arr = np.array(val_losses)
weights = np.exp(-loss_arr)  # higher weight for lower loss
weights /= weights.sum()  # normalize

ensemble_val_pred = np.tensordot(weights, np.stack(val_preds, axis=0), axes=([0], [0]))
ensemble_val_loss = -np.mean(
    np.log(np.take_along_axis(ensemble_val_pred, y_val[:, None], axis=1) + 1e-15)
)

use_ensemble = False
if ensemble_val_loss < best_logloss:
    best_logloss = ensemble_val_loss
    best_name = "Ensemble (softmax‑weighted soft‑voting of all models)"
    use_ensemble = True
    best_model = None  # not needed when ensemble is used


def temperature_scaled(probs, T):
    """Apply temperature scaling to a probability matrix."""
    scaled = np.power(probs, 1.0 / T)
    scaled_sum = scaled.sum(axis=1, keepdims=True)
    scaled_sum[scaled_sum == 0] = 1.0
    return scaled / scaled_sum


if use_ensemble:
    base_val_pred = ensemble_val_pred
else:
    base_val_pred = best_model.predict_proba(x_val)

temp_grid = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]
best_T = 1.0
best_T_loss = best_logloss
for T in temp_grid:
    scaled_pred = temperature_scaled(base_val_pred, T)
    loss = -np.mean(
        np.log(np.take_along_axis(scaled_pred, y_val[:, None], axis=1) + 1e-15)
    )
    if loss < best_T_loss:
        best_T_loss = loss
        best_T = T

if best_T != 1.0:
    best_logloss = best_T_loss
    best_name += f" + TempScale(T={best_T:.2f})"

print(f"Chosen model: {best_name}")
print("val_logloss:", best_logloss)




## === cell 1
test_ids = test_df.pop("id")
test_scaled = std_scaler.transform(test_df)

if use_ensemble:
    test_preds = np.tensordot(
        weights,
        np.stack([m.predict_proba(test_scaled) for m in models], axis=0),
        axes=([0], [0]),
    )
else:
    test_preds = best_model.predict_proba(test_scaled)  # shape (n_test, n_classes)

if best_T != 1.0:
    test_preds = temperature_scaled(test_preds, best_T)

sample_sub = pd.read_csv(sample_sub_path, nrows=0)  # only header
species_cols = [c for c in sample_sub.columns if c != "id"]

class_to_species = {
    i: label_encoder.inverse_transform([i])[0]
    for i in range(len(label_encoder.classes_))
}
species_to_col_idx = {class_to_species[i]: i for i in range(len(class_to_species))}

ordered_preds = np.zeros((test_preds.shape[0], len(species_cols)))
for idx, col in enumerate(species_cols):
    if col in species_to_col_idx:
        ordered_preds[:, idx] = test_preds[:, species_to_col_idx[col]]
    else:
        ordered_preds[:, idx] = 1e-15  # tiny probability for missing classes

ordered_preds = np.clip(ordered_preds, 1e-15, 1 - 1e-15)

submission_df = pd.DataFrame(ordered_preds, index=test_ids, columns=species_cols)

submission_path = "submission_nn_kernel.csv"
submission_df.to_csv(submission_path, index_label="id")
