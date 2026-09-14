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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-9.2867

# 6. Current score

-8.23389

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -11.48608) has done: 'I make the pipeline produce a valid submission deterministically and improve the expected metric by (1) removing the incompatible protobuf auto-downgrade/restart logic that prevents runs from yielding a file in this environment, (2) fixing a leakage/bug in scaling where test-time `WeeksZ` is computed with different min/max than training (hurts generalization), and (3) setting `Confidence` to a tuned constant closer to the metric-optimal range rather than a fixed 100. These are minimal changes that keep your model/feature set/training loop intact, but align preprocessing and prediction post-processing better with the Laplace log-likelihood metric. The code still train the same small MLP on the same features and write `submission.csv` with the correct columns and row order.'
- What this solution (achieved -7.91526) has done: 'I remove the protobuf “compatibility” shim that is now crashing at import time, since it’s not needed in this Kaggle environment and prevents any run from completing. Then I fix the TensorFlow error during training by ensuring `Sex` and `SmokingStatus` are always true strings (not ints/NaNs) before entering the `tf.data` pipeline, which avoids the unsupported `Cast int64 to string` graph op. Finally, I keep your exact model/training logic intact but make the dataset builder cast categorical tensors to `tf.string` defensively, so both train and inference run end-to-end and reliably write a valid `submission.csv`.'
- What this solution (achieved -8.22586) has done: 'We remove the protobuf-triggered import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which fixes the `MessageFactory.GetPrototype` error in this environment. Then we keep your exact preprocessing/model/training loop intact, but add a small safety fallback for the data directory so it runs whether Kaggle mounts as `/kaggle/input/...` or `../input/...`. Finally, we keep the same constant-confidence strategy but tune it slightly downward (still clipped to 70 by the metric) to move your score closer to the target band without changing the core model or feature logic.'
- What this solution (achieved -8.18466) has done: 'I fix the crash happening before any training by removing the protobuf environment override that is incompatible with TensorFlow 2.18 in this Kaggle image (it triggers the `MessageFactory.GetPrototype` error). I keep your preprocessing, model, training loop, and prediction pipeline identical, only making this import/runtime fix so the notebook runs end-to-end and writes `submission.csv`. Since your current score is already within ±10% of the target (and better than target), I not change any score-affecting logic like confidence calibration, epochs, features, or model structure.'
- What this solution (achieved -8.27773) has done: 'I fix the import-time crash by setting the protobuf implementation to the safe pure-Python backend before TensorFlow is imported (instead of popping the env var), which avoids the `MessageFactory.GetPrototype` error in this Kaggle image. I keep your data processing, model, training loop, and prediction logic unchanged so the score behavior stays essentially the same (already within ±10% of the target band and better than target). I also keep the same submission writing path/name and column order to ensure Kaggle accepts the output. No score-tuning changes (like Confidence) be made.'
- What this solution (achieved -8.26044) has done: 'I fix the import-time crash by removing the protobuf environment override that is incompatible with this TensorFlow/Kaggle image (it’s the direct cause of the `MessageFactory.GetPrototype` error). I keep your preprocessing, model definition, training loop, and prediction pipeline unchanged to preserve evaluation semantics and keep the score behavior essentially the same (your current score is already within ±10% of the target band). I also keep the submission writing exactly as `submission.csv` with the required columns and row order so Kaggle accepts it.'
- What this solution (achieved -8.23374) has done: 'I fix the import-time crash caused by the protobuf/TensorFlow incompatibility by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`. This is the minimal change needed to make the notebook run end-to-end again in the provided Kaggle environment and reliably write `submission.csv`. Because your current score (-8.26044) is already within the ±10% band around the target (-9.2867) and is better than target (higher is better), I not change any model/training/prediction logic or confidence calibration to avoid unnecessary score movement. Everything else (features, model, training loop, submission format) is kept identical.'
- What this solution (achieved -8.21289) has done: 'I remove the protobuf environment override that is currently triggering the TensorFlow/protobuf `MessageFactory.GetPrototype` crash at import time in this Kaggle image. Then I keep your preprocessing, model, training loop, and prediction pipeline unchanged so the score behavior stays essentially the same (you’re already within ±10% of the target band). Finally, I ensure the script still writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved -8.21541) has done: 'I fix the TensorFlow/protobuf import crash by avoiding the environment-variable removal that triggers the `MessageFactory.GetPrototype` error in this Kaggle image, and instead forcing the safe pure-Python protobuf backend before importing TensorFlow. I keep your preprocessing, model definition, training loop, and inference logic identical so score behavior stays essentially the same (and you’re already within the ±10% target band). I also add a small, non-score-affecting safety check that the submission has the required columns/row count before writing. The script then run end-to-end and reliably produce `submission.csv`.'
- What this solution (achieved -8.1799) has done: 'I fix the import-time protobuf/TensorFlow crash by removing the forced pure-Python protobuf backend, which is incompatible with this Kaggle TF 2.18 image and causes `MessageFactory.GetPrototype` to fail. I keep your preprocessing, model, training loop, and prediction logic unchanged so score behavior remains essentially the same (you’re already within the ±10% target band and better than target). I also keep the existing data-dir fallback and submission validation/writing so the notebook reliably produces a valid `submission.csv`.'
- What this solution (achieved -8.19988) has done: 'I fix the import-time TensorFlow/protobuf crash by removing the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` environment manipulation that triggers `MessageFactory.GetPrototype` in this Kaggle image, since it prevents any run from completing. Because your current score (-8.1799) is already within the ±10% tolerance band around the target (-9.2867) and is better than target (higher is better), I not change any training, preprocessing, or confidence calibration logic that would intentionally move the score. The rest of the pipeline is kept identical so it still trains the same model and writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved -8.23389) has done: 'I fix the import-time crash (`MessageFactory` has no `GetPrototype`) by avoiding TensorFlow import until after setting a safe protobuf implementation, which is the minimal change needed to make the notebook run in this Kaggle TF 2.18 environment. I keep your preprocessing, model, training loop, and inference logic identical so predictions remain effectively unchanged. Since your current score is already within the ±10% tolerance band around the target, I not tune confidence or any score-affecting parameters—only make the runtime stability fix and ensure the submission is written.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pathlib

DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(DATA_DIR):
    alt = "/kaggle/input/osic-pulmonary-fibrosis-progression"
    if os.path.exists(alt):
        DATA_DIR = alt

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
train_df.head()



## === cell 2
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
test_df.head()



## === cell 3
train_df = train_df.copy()
test_df = test_df.copy()
train_df["Source"] = "train"
test_df["Source"] = "test"
dataframe = pd.concat([train_df, test_df], axis=0, ignore_index=True)



## === cell 4
"""
Add patient level Baseline information, only the information that the test dataset will also have
1. Number of visits
2. Visit Number (0,1,2,3,4)
4. Variation in Percent
5. Change in smoking status
6. Range of Percent
"""



## === cell 5
CAT_FEATURES = ["Sex", "SmokingStatus"]
for c in CAT_FEATURES:
    dataframe[c] = dataframe[c].astype(str).fillna("Unknown")




## === cell 6
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target").astype(np.float32)

    features = {}
    for k, v in dict(dataframe).items():
        if k in CAT_FEATURES:
            features[k] = tf.constant(
                v.astype(str).fillna("Unknown").to_numpy(), dtype=tf.string
            )
        else:
            features[k] = tf.constant(
                pd.to_numeric(v, errors="coerce").fillna(0.0).to_numpy(),
                dtype=tf.float32,
            )

    ds = tf.data.Dataset.from_tensor_slices(
        (features, tf.constant(labels.to_numpy(), dtype=tf.float32))
    )
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(dataframe), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 7
def own_ZScaler(df, columns):
    for col in columns:
        new_col_name = col + "Z"
        col_min = df[col].min()
        col_max = df[col].max()
        denom = col_max - col_min
        if denom == 0:
            df[new_col_name] = 0.0
        else:
            df[new_col_name] = (df[col] - col_min) / denom


def fit_minmax(df, col):
    col_min = df[col].min()
    col_max = df[col].max()
    denom = col_max - col_min
    if denom == 0:
        denom = 1.0
    return float(col_min), float(col_max), float(denom)


def apply_minmax(df, col, col_min, denom):
    df[col + "Z"] = (df[col] - col_min) / denom




## === cell 8
numeric_columns = ["Weeks", "Age", "Percent"]



## === cell 9
train_only = dataframe.loc[dataframe.Source == "train"].copy()
weeks_min, weeks_max, weeks_denom = fit_minmax(train_only, "Weeks")
age_min, age_max, age_denom = fit_minmax(train_only, "Age")
pct_min, pct_max, pct_denom = fit_minmax(train_only, "Percent")

apply_minmax(dataframe, "Weeks", weeks_min, weeks_denom)
apply_minmax(dataframe, "Age", age_min, age_denom)
apply_minmax(dataframe, "Percent", pct_min, pct_denom)



## === cell 10
dataframe.head()



## === cell 11
dataframe["target"] = dataframe["FVC"].astype(float)



## === cell 12
from tensorflow import feature_column

feature_columns = []
for header in ["WeeksZ", "AgeZ", "PercentZ"]:
    feature_columns.append(feature_column.numeric_column(header))



## === cell 13
indicator_column_names = ["Sex", "SmokingStatus"]
for col_name in indicator_column_names:
    categorical_column = feature_column.categorical_column_with_vocabulary_list(
        col_name,
        list(pd.Series(dataframe[col_name].astype(str).fillna("Unknown")).unique()),
    )
    indicator_column = feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)



## === cell 14
"""## Try instead embedding columns
# embedding columns
embedded_column_names = ['Sex','SmokingStatus']
for col_name in embedded_column_names:
  m = len(dataframe[col_name].unique())
  categorical_column = feature_column.categorical_column_with_vocabulary_list(
      col_name, dataframe[col_name].unique())
  embedded_column = feature_column.embedding_column(categorical_column, dimension = min(50,m//2))
  feature_columns.append(embedded_column)"""



## === cell 15
"""sex_smoker_feature = feature_column.crossed_column(['Sex', 'SmokingStatus'], hash_bucket_size=100)
feature_columns.append(feature_column.indicator_column(sex_smoker_feature))"""



## === cell 16
feature_columns



## === cell 17
NUM_FEATURES = ["WeeksZ", "AgeZ", "PercentZ"]

for c in CAT_FEATURES:
    dataframe[c] = dataframe[c].astype(str).fillna("Unknown")
for c in NUM_FEATURES:
    dataframe[c] = (
        pd.to_numeric(dataframe[c], errors="coerce").fillna(0.0).astype(np.float32)
    )

inputs = {}
for f in NUM_FEATURES:
    inputs[f] = tf.keras.Input(shape=(1,), name=f, dtype=tf.float32)
for f in CAT_FEATURES:
    inputs[f] = tf.keras.Input(shape=(1,), name=f, dtype=tf.string)

num_concat = layers.Concatenate(name="num_concat")([inputs[f] for f in NUM_FEATURES])
normalizer = layers.Normalization(axis=-1, name="num_norm")

sex_lookup = layers.StringLookup(output_mode="one_hot", name="sex_lookup")
smoke_lookup = layers.StringLookup(output_mode="one_hot", name="smoke_lookup")

sex_oh = sex_lookup(inputs["Sex"])
smoke_oh = smoke_lookup(inputs["SmokingStatus"])

x = layers.Concatenate(name="all_features")([normalizer(num_concat), sex_oh, smoke_oh])

preprocess_model = tf.keras.Model(inputs=inputs, outputs=x, name="preprocess")



## === cell 18
train_df = dataframe.loc[dataframe.Source == "train"].copy()
test_df = dataframe.loc[dataframe.Source == "test"].copy()



## === cell 19
train, val = train_test_split(train_df, test_size=0.2, random_state=SEED)
print(len(train), "train examples")
print(len(val), "validation examples")



## === cell 20
if len(test_df) < 10:
    EPOCHS = 1000
else:
    EPOCHS = 1000
batch_size = 128



## === cell 21
for df_ in (train, val, test_df):
    for c in CAT_FEATURES:
        df_[c] = df_[c].astype(str).fillna("Unknown")
    for c in NUM_FEATURES:
        df_[c] = pd.to_numeric(df_[c], errors="coerce").fillna(0.0).astype(np.float32)
    df_["target"] = (
        pd.to_numeric(df_["target"], errors="coerce").fillna(0.0).astype(np.float32)
    )

train_num = np.stack(
    [train[c].astype(np.float32).to_numpy() for c in NUM_FEATURES], axis=1
)
normalizer.adapt(train_num)

sex_lookup.adapt(train["Sex"].astype(str).to_numpy())
smoke_lookup.adapt(train["SmokingStatus"].astype(str).to_numpy())

train_ds = df_to_dataset(
    train[NUM_FEATURES + CAT_FEATURES + ["target"]], batch_size=batch_size
)
val_ds = df_to_dataset(
    val[NUM_FEATURES + CAT_FEATURES + ["target"]], shuffle=False, batch_size=batch_size
)



## === cell 22
model = tf.keras.Sequential(
    [
        preprocess_model,
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="linear"),
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0)



## === cell 23
test_df.head()



## === cell 24
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sample_sub[["Patient", "Weeks"]] = sample_sub["Patient_Week"].str.split(
    "_", expand=True
)
sample_sub["Weeks"] = sample_sub["Weeks"].astype(int)

base = test_df[["Patient", "AgeZ", "PercentZ", "Sex", "SmokingStatus"]].copy()
for c in ["Sex", "SmokingStatus"]:
    base[c] = base[c].astype(str).fillna("Unknown")
for c in ["AgeZ", "PercentZ"]:
    base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0).astype(np.float32)

pred_df = sample_sub.merge(base, on="Patient", how="left")
pred_df["AgeZ"] = pred_df["AgeZ"].fillna(0.0).astype(np.float32)
pred_df["PercentZ"] = pred_df["PercentZ"].fillna(0.0).astype(np.float32)
pred_df["Sex"] = pred_df["Sex"].fillna("Unknown").astype(str)
pred_df["SmokingStatus"] = pred_df["SmokingStatus"].fillna("Unknown").astype(str)

pred_df["Weeks"] = pred_df["Weeks"].astype(int)

apply_minmax(pred_df, "Weeks", weeks_min, weeks_denom)
pred_df["WeeksZ"] = (
    pd.to_numeric(pred_df["WeeksZ"], errors="coerce").fillna(0.0).astype(np.float32)
)

pred_df["target"] = 0.0



## === cell 25
for c in NUM_FEATURES:
    pred_df[c] = (
        pd.to_numeric(pred_df[c], errors="coerce").fillna(0.0).astype(np.float32)
    )
for c in CAT_FEATURES:
    pred_df[c] = pred_df[c].astype(str).fillna("Unknown")

test_ds = df_to_dataset(
    pred_df[NUM_FEATURES + CAT_FEATURES + ["target"]],
    shuffle=False,
    batch_size=batch_size,
)



## === cell 26
preds = model.predict(test_ds, batch_size=100, verbose=0).reshape(-1)
preds[:10], preds.shape



## === cell 27
sub = sample_sub[["Patient_Week"]].copy()
sub["FVC"] = preds.astype(float)

sub["Confidence"] = 200.0
sub = sub[["Patient_Week", "FVC", "Confidence"]]

required_cols = ["Patient_Week", "FVC", "Confidence"]
missing = [c for c in required_cols if c not in sub.columns]
if missing:
    raise ValueError(f"Submission missing columns: {missing}")
if len(sub) != len(sample_sub):
    raise ValueError(f"Submission rowcount mismatch: {len(sub)} vs {len(sample_sub)}")



## === cell 28
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 29
sub.tail()
