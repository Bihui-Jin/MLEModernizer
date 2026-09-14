# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Given readings from several seismic sensors around a volcano, estimate how long it will be until the next eruption.

## Metric
Mean absolute error (MAE) between the predicted loss and the actual loss.

## Submission Format
For every id in the test set, you should predict the time until the next eruption. The file should contain a header and have the following format:

```
segment_id,time_to_eruption
1,1
2,2
3,3
etc.
```

## Data
### Dataset Description

#### Files
**train.csv** Metadata for the train files.

- `segment_id`: ID code for the data segment. Matches the name of the associated data file.
- `time_to_eruption`: The target value, the time until the next eruption.

**[train|test]/*.csv**: the data files. Each file contains ten minutes of logs from ten different sensors arrayed around a volcano. The readings have been normalized within each segment, in part to ensure that the readings fall within the range of int16 values. If you are using the Pandas library you may find that you still need to load the data as float32 due to the presence of some nulls.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
        input/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
        working/
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
```

-> data/predict-volcanic-eruptions-ingv-oe/sample_submission.csv has 444 rows and 2 columns.
The columns are: segment_id, time_to_eruption

-> data/predict-volcanic-eruptions-ingv-oe/test/1003520023.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1004346803.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1007996426.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1009749143.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1016956864.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1024522044.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1028325789.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> (stopped after 10 files for performance)

# 5. Target score

5793085.557252489

# 6. Current score

10751320.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 11518715.0) has done: 'I replace the missing‑file reads and the faulty PyTorch model with a simple baseline that predicts the overall mean eruption time. This removes the path errors, undefined variables, and heavy dependencies while still producing a valid `submission.csv`. The baseline give a MAE far lower than the huge target, satisfying the “lower‑is‑better” requirement without further score manipulation.'
- What this solution (achieved 11495728.0) has done: 'I fill missing values in the extracted feature matrices before fitting and predicting, preventing the NaN‑related errors while keeping the original model unchanged. This small preprocessing step ensures the LinearRegression model can be trained and used to generate a valid `submission.csv`, moving the pipeline from failure to a working state without altering core logic.'
- What this solution (achieved 10649104.0) has done: 'I extend the feature set by adding per‑sensor statistics (mean, std, min, max) instead of only the mean, and switch to a lightly regularised linear model (Ridge) which often reduces over‑fitting on these aggregate features. These changes keep the overall pipeline and model type unchanged while providing richer information to improve the MAE and move the score closer to the target.'
- What this solution (achieved 10649031.0) has done: 'I add a StandardScaler to normalise the aggregated sensor features before fitting the Ridge model, and apply the same scaling to the validation and test data. Normalising the features usually helps linear models converge to better coefficients, reducing MAE and moving the score closer to the target while keeping the original pipeline and model untouched.'
- What this solution (achieved 10701155.0) has done: 'We refit the Ridge model on the entire training set after the validation step (using a scaler fitted on all features) so the model can leverage more data, and then blend its predictions with the overall mean eruption time. This small change keeps the same feature extraction and model type while improving generalisation and moving the MAE closer to the target.'
- What this solution (achieved 10636902.0) has done: 'I lower the Ridge regularisation (alpha = 0.1) so the model can fit the data more closely and increase the contribution of the model prediction in the final blend (90 % model, 10 % global mean). These small adjustments keep the original pipeline intact while expectedly reducing the MAE toward the target.'
- What this solution (achieved 10641537.0) has done: 'I lower the Ridge regularisation (α = 0.05) to let the model fit the data more closely and replace the global‑mean blend with a small 5 % contribution of the overall median eruption time (95 % model). These minimal tweaks keep the original feature extraction and modelling pipeline unchanged while expectedly reducing the MAE toward the target.'
- What this solution (achieved 10718286.0) has done: 'I enrich the per‑segment feature extraction by adding median, 25th/75th percentiles and the inter‑quartile range for each sensor, then train the same Ridge model without regularisation (α = 0) and use the pure model prediction for the submission (dropping the small median blend). These small, targeted changes keep the overall pipeline unchanged while providing more informative features and a less constrained linear model, which should lower the MAE toward the target value.'
- What this solution (achieved 10809214.0) has done: 'I keep the existing feature extraction and model training but blend the model’s predictions with the overall median eruption time. Since the pure Ridge model yields a MAE far above the target, a modest contribution from the global median (which is a strong baseline) should bring the error down toward the target value without altering the core pipeline.'
- What this solution (achieved 10655144.0) has done: 'I add a modest regularisation to the Ridge model (alpha = 1.0) to prevent extreme coefficient values, and simplify the blending logic so that we fall back to the global median whenever the median baseline outperforms the model on the validation split. This keeps the original feature pipeline unchanged while making the predictions more stable and moving the MAE closer to the target.'
- What this solution (achieved 10667702.0) has done: 'We add a tiny grid‑search to pick the blend weight that gives the lowest validation MAE (instead of the existing pure‑model or pure‑median choice). This small change keeps the same feature extraction and Ridge model but should lower the overall MAE, moving the score closer to the target.'
- What this solution (achieved 10716295.0) has done: 'I replace the regularised Ridge model with an un‑regularised LinearRegression (which often fits the aggregated sensor statistics more closely) and refine the blend‑weight search to a finer 0.01 step grid. These small hyper‑parameter tweaks keep the overall pipeline unchanged while likely lowering the validation MAE, moving the score nearer to the target.'
- What this solution (achieved 10751320.0) has done: 'We enrich the segment feature set with a few aggregate statistics (overall mean, std, min, max of the sensor‑wise means) and search the blend weight with a finer 0.001 step. These tiny extensions keep the original pipeline and linear model intact while providing the model with more informative signals, which should lower the MAE toward the target without altering the core logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PATH_DATA = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/"



## === cell 1
train_df = pd.read_csv(os.path.join(PATH_DATA, "train.csv"))
submission = pd.read_csv(os.path.join(PATH_DATA, "sample_submission.csv"))


def segment_features(segment_id: str, folder: str) -> pd.Series:
    """Return aggregated statistics for each sensor in a segment,
    plus a few overall aggregates to give the model more signal."""
    file_path = os.path.join(folder, f"{segment_id}.csv")
    df = pd.read_csv(file_path, dtype=np.float32)

    feats = {}
    sensor_means = []
    for col in df.columns:
        col_data = df[col]
        mean_val = col_data.mean()
        std_val = col_data.std()
        min_val = col_data.min()
        max_val = col_data.max()
        median_val = col_data.median()
        q25 = col_data.quantile(0.25)
        q75 = col_data.quantile(0.75)
        iqr = q75 - q25

        feats[f"{col}_mean"] = mean_val
        feats[f"{col}_std"] = std_val
        feats[f"{col}_min"] = min_val
        feats[f"{col}_max"] = max_val
        feats[f"{col}_median"] = median_val
        feats[f"{col}_q25"] = q25
        feats[f"{col}_q75"] = q75
        feats[f"{col}_iqr"] = iqr

        sensor_means.append(mean_val)

    sensor_means_arr = np.array(sensor_means)
    feats["overall_mean_of_means"] = sensor_means_arr.mean()
    feats["overall_std_of_means"] = sensor_means_arr.std()
    feats["overall_min_of_means"] = sensor_means_arr.min()
    feats["overall_max_of_means"] = sensor_means_arr.max()

    return pd.Series(feats)


train_features_list = []
for seg_id in train_df["segment_id"]:
    feats = segment_features(seg_id, os.path.join(PATH_DATA, "train"))
    train_features_list.append(feats)

X_full = pd.DataFrame(train_features_list)
X_full = X_full.fillna(X_full.mean())
y_full = train_df["time_to_eruption"].values



## === cell 2
X_train, X_valid, y_train, y_valid = train_test_split(
    X_full, y_full, test_size=0.2, random_state=0
)

scaler_split = StandardScaler()
X_train_scaled = scaler_split.fit_transform(X_train)
X_valid_scaled = scaler_split.transform(X_valid)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

valid_pred = model.predict(X_valid_scaled)
val_mae = mean_absolute_error(y_valid, valid_pred)
print(f"Validation MAE (model only): {val_mae:.2f}")

global_median = np.median(y_full)
median_pred_valid = np.full_like(y_valid, global_median, dtype=float)
median_mae = mean_absolute_error(y_valid, median_pred_valid)
print(f"Validation MAE (global median): {median_mae:.2f}")

weights = np.arange(0.0, 1.001, 0.001)
best_w = 0.0
best_mae = float("inf")
for w in weights:
    blended = w * valid_pred + (1 - w) * global_median
    mae = mean_absolute_error(y_valid, blended)
    if mae < best_mae:
        best_mae = mae
        best_w = w
blend_weight = best_w
print(
    f"Chosen blend weight (model contribution): {blend_weight:.3f} => Validation MAE: {best_mae:.2f}"
)

scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_full)

model_full = LinearRegression()
model_full.fit(X_full_scaled, y_full)



## === cell 3
test_features = []
for seg_id in submission["segment_id"]:
    feats = segment_features(seg_id, os.path.join(PATH_DATA, "test"))
    test_features.append(feats)

X_test = pd.DataFrame(test_features)
X_test = X_test.fillna(X_test.mean())
X_test_scaled = scaler_full.transform(X_test)

model_pred = model_full.predict(X_test_scaled)

final_pred = blend_weight * model_pred + (1 - blend_weight) * global_median

submission["time_to_eruption"] = final_pred

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
