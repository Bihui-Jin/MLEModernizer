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

No external packages required in the script and installed.

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

-6.921331988056262

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -10.69004) has done: 'Implemented fixes to resolve import errors, replace failing TensorFlow models with a lightweight sklearn‑based regressor, and ensure required prediction columns are generated for blending. Added safe imports, a fallback `SimpleRegressor` that creates the expected CV columns, and updated the training/prediction steps accordingly. The pipeline now runs end‑to‑end and writes a valid `submission.csv` file.'
- What this solution (achieved -10.52357) has done: 'Implemented safe optional imports for cv2 and pydicom to avoid crashes when those libraries are missing, and added fallback handling in ImageDataPreprocessor. Enhanced SimpleRegressor by calibrating the confidence value to the residual standard deviation (clipped at 70) instead of a fixed 100, and increased the GradientBoostingRegressor complexity (more trees) for better predictive power. These changes fix the runtime error, keep the core pipeline intact, and provide a more appropriate confidence estimate, thereby moving the validation score toward the target.'
- What this solution (achieved -10.52357) has done: 'I protect the script from the TensorFlow import error by safely handling the import and removing its unused components, while keeping the rest of the pipeline unchanged. This change prevents the protobuf‑related crash, allowing the code to run end‑to‑end and generate a valid `submission.csv` file, moving the score closer to the target.'

# 9. Code solution

## === cell 0
DATA_ROOT = "./data/osic-pulmonary-fibrosis-progression"

df_train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
df_test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
df_submission = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

print(
    f'Training Set Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(f"Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f'Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f"Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f"Sample Submission Shape = {df_submission.shape}")
print(
    f"Sample Submission Memory Usage = {df_submission.memory_usage().sum() / 1024 ** 2:.2f} MB"
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3342472775.py in <cell line: 0>()
      3 
      4 # Load CSV files using the unified DATA_ROOT
----> 5 df_train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
      6 df_test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
      7 df_submission = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

NameError: name 'pd' is not defined

## === cell 1
tabular_data_preprocessor = TabularDataPreprocessor(
    train=df_train,
    test=df_test,
    submission=df_submission,
    n_folds=2,
    shuffle=True,
    ohe=True,
    scale=True,
)

df_train, df_test = tabular_data_preprocessor.create_tabular_features()

print(
    f'Training Set (Tabular Features) Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(
    f"Training Set (Tabular Features) Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB"
)
print(
    f'Set (Tabular Features) Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}'
)
print(
    f"Set (Tabular Features) Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB"
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3289844902.py in <cell line: 0>()
----> 1 tabular_data_preprocessor = TabularDataPreprocessor(
      2     train=df_train,
      3     test=df_test,
      4     submission=df_submission,
      5     n_folds=2,

NameError: name 'TabularDataPreprocessor' is not defined

## === cell 2
seed_everything(SEED)

X_train = df_train.drop(columns=["FVC", "Weeks"])
y_train = df_train["FVC"].copy(deep=True)

model_parameters = {
    "model": "Stack",
    "cv": [1],
    "predictors": [
        "Age",
        "Male",
        "Female",
        "Never smoked",
        "Ex-smoker",
        "Currently smokes",
        "FVC_Baseline",
        "Percent",
        "Weeks_Passed",
    ],
    "mlp_parameters": {"lr": 0.00025, "epochs": 800, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.2, 0.5, 0.8],
        "lr": 0.00025,
        "epochs": 800,
        "batch_size": 2**5,
    },
}


class SimpleRegressor:
    def __init__(self, predictors):
        self.predictors = predictors
        self.model = GradientBoostingRegressor(
            n_estimators=800,  # increased from 500
            learning_rate=0.05,
            max_depth=4,  # increased from 3
            random_state=SEED,
        )
        self.sigma = 100.0  # default, will be overwritten after training

    def train(self, X, y):
        self.model.fit(X[self.predictors], y)

        oof_pred = self.model.predict(X[self.predictors])
        residual_std = max(np.std(y - oof_pred), 70.0)
        self.sigma = residual_std

        df_train["CV1_MLP_FVC_Predictions"] = oof_pred
        df_train["CV1_MLP_Confidence_Predictions"] = self.sigma

        for q in [0.2, 0.5, 0.8]:
            df_train[f"CV1_QR_{q}_Predictions"] = oof_pred

    def predict(self, X):
        preds = self.model.predict(X[self.predictors])
        df_test["CV1_MLP_FVC_Predictions"] = preds
        df_test["CV1_MLP_Confidence_Predictions"] = self.sigma
        for q in [0.2, 0.5, 0.8]:
            df_test[f"CV1_QR_{q}_Predictions"] = preds


simple_reg = SimpleRegressor(predictors=model_parameters["predictors"])
simple_reg.train(X_train, y_train)
simple_reg.predict(df_test)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/885160577.py in <cell line: 0>()
----> 1 seed_everything(SEED)
      2 
      3 X_train = df_train.drop(columns=["FVC", "Weeks"])
      4 y_train = df_train["FVC"].copy(deep=True)
      5 

NameError: name 'seed_everything' is not defined

## === cell 3
class ImageDataPreprocessor:

    def __init__(
        self,
        train,
        test,
        resize_shape,
        window_width,
        window_center,
        y_min,
        y_max,
        scale,
    ):

        self.train = train.copy(deep=True)
        self.test = test.copy(deep=True)

        self.resize_shape = resize_shape
        self.window_width = window_width
        self.window_center = window_center
        self.y_min = y_min
        self.y_max = y_max
        self.scale = scale

    def crop(self, s):
        if np.all(s == 0):
            return s
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_cropped = s[~np.all(s == 0, axis=1)]
            s_cropped = s_cropped[:, ~np.all(s_cropped == 0, axis=0)]
        else:
            s_cropped = s
        return s_cropped

    def resize(self, s):
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_resized = cv2.resize(
                s, self.resize_shape, interpolation=cv2.INTER_NEAREST
            )
        else:
            s_resized = s
        return s_resized

    def window(self, s, slope, intercept, window_width, window_center, y_min, y_max):
        x = s * slope + intercept
        y = np.zeros_like(x)
        y[x <= (window_center - 0.5 - (window_width - 1) / 2)] = y_min
        y[x > (window_center - 0.5 + (window_width - 1) / 2)] = y_max
        mask = (x > (window_center - 0.5 - (window_width - 1) / 2)) & (
            x <= (window_center - 0.5 + (window_width - 1) / 2)
        )
        y[mask] = ((x[mask] - (window_center - 0.5)) / (window_width - 1) + 0.5) * (
            y_max - y_min
        ) + y_min
        return y

    def load_scan(self, dataset, patient_name):
        base_path = os.path.join(DATA_ROOT, dataset, patient_name)
        patient_directory = [
            pydicom.dcmread(os.path.join(base_path, s)) for s in os.listdir(base_path)
        ]
        try:
            patient_directory.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            slice_positions = np.round(
                [s.ImagePositionPatient[2] for s in patient_directory], 4
            )
            non_duplicate_idx = np.unique(
                [
                    np.where(slice_position == slice_positions)[0][0]
                    for slice_position in slice_positions
                ]
            )
        except AttributeError:
            patient_directory.sort(key=lambda s: int(s.InstanceNumber))
            instance_numbers = np.array(
                [int(s.InstanceNumber) for s in patient_directory]
            )
            non_duplicate_idx = np.unique(
                [
                    np.where(instance_number == instance_numbers)[0][0]
                    for instance_number in instance_numbers
                ]
            )
        patient_directory = list(np.array(patient_directory)[non_duplicate_idx])

        metadata = {}
        pixel_spacings = np.zeros((len(patient_directory), 2))
        slice_positions = np.zeros((len(patient_directory)))

        for i, s in enumerate(patient_directory):
            try:
                pixel_spacings[i, :] = np.array(s.PixelSpacing)
            except AttributeError:
                pixel_spacings[i, :] = np.nan
            try:
                slice_positions[i] = s.ImagePositionPatient[2]
            except AttributeError:
                continue

        metadata["PixelSpacing"] = list(np.round(pixel_spacings.mean(axis=0), 3))

        if patient_name == "ID00128637202219474716089":
            metadata["SliceSpacing"] = 5.0
        elif patient_name == "ID00132637202222178761324":
            metadata["SliceSpacing"] = 0.7
        else:
            metadata["SliceSpacing"] = list(
                mode(np.abs(np.diff(np.round(slice_positions, 3))))
            )[0][0]

        scan = np.zeros(
            (len(patient_directory), self.resize_shape[0], self.resize_shape[1]),
            dtype=np.int16,
        )
        for i, s in enumerate(patient_directory):
            s_processed = self.crop(s.pixel_array)
            s_processed = self.resize(s_processed)
            s_processed = self.window(
                s_processed,
                s.RescaleSlope,
                s.RescaleIntercept,
                self.window_width,
                self.window_center,
                self.y_min,
                self.y_max,
            )
            if np.all(s_processed == 0):
                continue
            else:
                scan[i] = np.int16(s_processed)

        del patient_directory
        scan = scan[~np.all(scan == 0, axis=(-1, -2))]
        return scan, metadata

    def create_image_features(self):
        if not (cv2_available and pydicom_available):
            print("cv2 or pydicom not available – skipping image feature creation.")
            return self.train.copy(deep=True), self.test.copy(deep=True)

        print(f'Creating Image Features for Training Set\n{"-" * 40}')
        for i, patient_name in enumerate(self.train["Patient"].unique()):
            scan, metadata = self.load_scan("train", patient_name)
            scan_size = scan.nbytes >> 20
            print(
                f'[{i + 1}/{len(self.train["Patient"].unique())}] {patient_name} Shape: {scan.shape} - Size: {scan_size} MB'
            )
            volume = (
                (metadata["SliceSpacing"] * scan.shape[0])
                * (metadata["PixelSpacing"][0] * scan.shape[1])
                * (metadata["PixelSpacing"][1] * scan.shape[2])
            )
            self.train.loc[self.train["Patient"] == patient_name, "VoxelVolume"] = (
                volume / (scan.shape[0] * scan.shape[1] * scan.shape[2])
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Skew"] = skew(
                scan.flatten()
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Kurtosis"] = (
                kurtosis(scan.flatten())
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Mean"] = (
                scan.flatten().mean()
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Std"] = (
                scan.flatten().std()
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Var"] = (
                scan.flatten().var()
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Min_Volume"] = (
                scan[scan == self.y_min].shape[0]
                * self.train.loc[self.train["Patient"] == patient_name, "VoxelVolume"]
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Max_Volume"] = (
                scan[scan == self.y_max].shape[0]
                * self.train.loc[self.train["Patient"] == patient_name, "VoxelVolume"]
            )
            slice_skews = [skew(s.flatten()) for s in scan]
            self.train.loc[self.train["Patient"] == patient_name, "Std_Slice_Skew"] = (
                np.std(slice_skews)
            )
            self.train.loc[self.train["Patient"] == patient_name, "Var_Slice_Skew"] = (
                np.var(slice_skews)
            )
            del scan, metadata, volume, slice_skews
            gc.collect()

        print(f'\nCreating Image Features for Test Set\n{"-" * 36}')
        for i, patient_name in enumerate(self.test["Patient"].unique()):
            scan, metadata = self.load_scan("test", patient_name)
            scan_size = scan.nbytes >> 20
            print(
                f'[{i + 1}/{len(self.test["Patient"].unique())}] {patient_name} Shape: {scan.shape} - Size: {scan_size} MB'
            )
            volume = (
                (metadata["SliceSpacing"] * scan.shape[0])
                * (metadata["PixelSpacing"][0] * scan.shape[1])
                * (metadata["PixelSpacing"][1] * scan.shape[2])
            )
            self.test.loc[self.test["Patient"] == patient_name, "VoxelVolume"] = (
                volume / (scan.shape[0] * scan.shape[1] * scan.shape[2])
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Skew"] = skew(
                scan.flatten()
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Kurtosis"] = (
                kurtosis(scan.flatten())
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Mean"] = (
                scan.flatten().mean()
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Std"] = (
                scan.flatten().std()
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Var"] = (
                scan.flatten().var()
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Min_Volume"] = (
                scan[scan == self.y_min].shape[0]
                * self.test.loc[self.test["Patient"] == patient_name, "VoxelVolume"]
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Max_Volume"] = (
                scan[scan == self.y_max].shape[0]
                * self.test.loc[self.test["Patient"] == patient_name, "VoxelVolume"]
            )
            slice_skews = [skew(s.flatten()) for s in scan]
            self.test.loc[self.test["Patient"] == patient_name, "Std_Slice_Skew"] = (
                np.std(slice_skews)
            )
            self.test.loc[self.test["Patient"] == patient_name, "Var_Slice_Skew"] = (
                np.var(slice_skews)
            )
            del scan, metadata, volume, slice_skews
            gc.collect()

        if self.scale:
            scale_features = [
                "Scan_Skew",
                "Scan_Kurtosis",
                "Scan_Mean",
                "Scan_Std",
                "Scan_Var",
                "Scan_Min_Volume",
                "Scan_Max_Volume",
            ]
            scaler = StandardScaler()
            scaler.fit(self.train.loc[:, scale_features])
            self.train.loc[:, scale_features] = scaler.transform(
                self.train.loc[:, scale_features]
            )
            self.test.loc[:, scale_features] = scaler.transform(
                self.test.loc[:, scale_features]
            )

        return self.train.copy(deep=True), self.test.drop(columns=["VoxelVolume"]).copy(
            deep=True
        )
