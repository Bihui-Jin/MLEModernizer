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
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
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

-9.3368

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch as pt
from torch.utils.data import Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import pydicom
from sklearn.model_selection import KFold
import seaborn as sns
from glob import glob
import scipy.ndimage
from skimage import morphology, measure
from skimage.filters import threshold_otsu, median
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import clear_border
from scipy.stats import describe

trainImagesPath = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"
dtype = pt.float
use_cuda = pt.cuda.is_available()
device = pt.device("cuda:0" if use_cuda else "cpu")

inputs = [
    "PercentIn",
    "min_Percent",
    "AgeIn",
    "WeekIn",
    "min_FVC",
    "SmokingStatus_Currently smokes",
    "SmokingStatus_Ex-smoker",
    "SmokingStatus_Never smoked",
    "Sex_Male",
    "Sex_Female",
]

test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
subms = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
subms["Patient"] = subms.Patient_Week.apply(lambda x: x.split("_")[0])
subms["Weeks"] = subms.Patient_Week.apply(lambda x: int(x.split("_")[-1]))
train["Split"] = "train"
test["Split"] = "test"
test["PatientDir"] = test.Patient.apply(
    lambda x: "../input/osic-pulmonary-fibrosis-progression/test/" + x
)
train["PatientDir"] = train.Patient.apply(
    lambda x: "../input/osic-pulmonary-fibrosis-progression/train/" + x
)
subms["PatientDir"] = subms.Patient.apply(
    lambda x: "../input/osic-pulmonary-fibrosis-progression/train/" + x
)
subms = subms[["Patient", "Weeks", "Patient_Week"]]
subms = subms.merge(test.drop("Weeks", axis=1), on="Patient")
subms["Split"] = "subm"
data = pd.concat([test, subms, train], ignore_index=True)

data["first_week"] = data.Weeks
data["first_week"] = data.groupby("Patient")["first_week"].transform("min")
data["first_week"] = data.Weeks - data.first_week
data["min_FVC"] = data.groupby("Patient")["FVC"].transform("min")
data["min_Percent"] = data.groupby("Patient")["Percent"].transform("min")
data["min_Percent"] = (data["min_Percent"] - data["min_Percent"].min()) / (
    data["min_Percent"].max() - data["min_Percent"].min()
)
data["WeekIn"] = (data.first_week - data.first_week.min()) / (
    data.first_week.max() - data.first_week.min()
)
data = pd.concat(
    [data, pd.get_dummies(data.SmokingStatus, prefix="SmokingStatus")], axis=1
)
data = pd.concat([data, pd.get_dummies(data.Sex, prefix="Sex")], axis=1)

train = data.loc[data.Split == "train"].copy()
subms = data.loc[data.Split == "subm"].copy()
test = data.loc[data.Split == "test"].copy()





## === cell 1
class OSICDataSet(Dataset):
    def __init__(self, data, mode="train"):
        self.data = data.copy()
        self.data["Smoke"] = self.data.SmokingStatus.replace(
            {"Ex-smoker": 0.5, "Never smoked": 0, "Currently smokes": 1}
        )
        self.data["Gender"] = self.data.Sex.replace({"Male": 1, "Female": 0})
        self.data["min_FVC"] = (self.data.min_FVC - data.min_FVC.min()) / (
            data.min_FVC.max() - data.min_FVC.min()
        )
        self.data["WeekIn"] = (self.data.first_week - data.first_week.min()) / (
            data.first_week.max() - data.first_week.min()
        )
        self.data["AgeIn"] = (self.data.Age - data.Age.min()) / (
            data.Age.max() - data.Age.min()
        )
        self.data["PercentIn"] = (self.data.Percent - data.Percent.min()) / (
            data.Percent.max() - data.Percent.min()
        )
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        otherData = pt.from_numpy(self.data.loc[idx, inputs].values.astype(np.float32))
        if self.mode == "train":
            targets = pt.from_numpy(
                self.data.loc[idx, ["FVC"]].values.astype(np.float32)
            )
            return otherData, targets
        return otherData


class Model(pt.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.start = pt.nn.Sequential(
            pt.nn.Linear(len(inputs), 128),
            pt.nn.ReLU(),
            pt.nn.Linear(128, 256),
            pt.nn.ReLU(),
            pt.nn.Linear(256, 512),
            pt.nn.ReLU(),
            pt.nn.Linear(512, 256),
            pt.nn.ReLU(),
        )
        self.left = pt.nn.Sequential(pt.nn.Linear(256, 128))
        self.sigmoid = pt.nn.Sigmoid()
        self.right = pt.nn.Sequential(pt.nn.Linear(256, 128))
        self.last = pt.nn.Sequential(pt.nn.Linear(128, 3))
        self.lastRelu = pt.nn.Sequential(pt.nn.Linear(128, 3), pt.nn.ReLU())

    def forward(self, x):
        h = self.start(x)
        l = self.left(h)
        r = self.right(h)
        h = l * self.sigmoid(r)
        p1 = self.last(h)
        p2 = self.lastRelu(h)
        out = p1 + pt.cumsum(p2, 1)
        return out


model = Model().to(device)

C1, C2 = pt.FloatTensor([70]), pt.FloatTensor([1000])


def score(y_pred, y_true):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = pt.max(sigma, C1.expand_as(sigma))
    delta = pt.abs(y_true - fvc_pred)
    delta = pt.min(delta, C2)
    sq2 = pt.sqrt(pt.FloatTensor([2]))
    metric = (delta / sigma_clip) * sq2 + pt.log(sigma_clip * sq2)
    return pt.mean(metric)


def pinballLoss(pred, label, quant):
    err = label - pred
    m = pt.mean(pt.max(quant * err, (quant - 1) * err))
    return m


def loss(pred, label):
    quantiles = pt.FloatTensor([0.2, 0.5, 0.8])
    return 0.8 * pinballLoss(pred, label, quantiles) + 0.2 * score(pred, label)




## === cell 2
trainDataSet = OSICDataSet(train)
submDataSet = OSICDataSet(subms, "submission")

optimizer = optim.Adam(model.parameters(), lr=0.075, weight_decay=0.1, eps=0.001)
losses = []
valLosses = []

for i in range(1):
    for idx in range(len(trainDataSet)):
        model.train()
        optimizer.zero_grad()
        x, y = trainDataSet[idx]
        x = x.to(device)
        y = y.to(device)
        pred = model(x.unsqueeze(0))
        los = loss(pred, y)
        los.backward()
        optimizer.step()
        if idx % 200 == 0:
            losses.append(los.item())
            model.eval()
            with torch.no_grad():
                val_pred = model(x.unsqueeze(0))
                val_losses = loss(val_pred, y)
                valLosses.append(val_losses.item())




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

KeyError: 0

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_54/1037378790.py in <cell line: 0>()
     15         model.train()
     16         optimizer.zero_grad()
---> 17         x, y = trainDataSet[idx]
     18         x = x.to(device)
     19         y = y.to(device)

/tmp/ipykernel_54/868205203.py in __getitem__(self, idx)
     25 
     26     def __getitem__(self, idx):
---> 27         otherData = pt.from_numpy(self.data.loc[idx, inputs].values.astype(np.float32))
     28         if self.mode == "train":
     29             targets = pt.from_numpy(

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1366         with suppress(IndexingError):
   1367             tup = self._expand_ellipsis(tup)
-> 1368             return self._getitem_lowerdim(tup)
   1369 
   1370         # no multi-index, so validate all of the indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_lowerdim(self, tup)
   1063                 # We don't need to check for tuples here because those are
   1064                 #  caught by the _is_nested_tuple_indexer check above.
-> 1065                 section = self._getitem_axis(key, axis=i)
   1066 
   1067                 # We should never have a scalar section here, because

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1429         # fall thru to straight lookup
   1430         self._validate_key(key, axis)
-> 1431         return self._get_label(key, axis=axis)
   1432 
   1433     def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_label(self, label, axis)
   1379     def _get_label(self, label, axis: AxisInt):
   1380         # GH#5567 this will fail if the label is not present in the axis.
-> 1381         return self.obj.xs(label, axis=axis)
   1382 
   1383     def _handle_lowerdim_multi_index_axis0(self, tup: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in xs(self, key, axis, level, drop_level)
   4299                     new_index = index[loc]
   4300         else:
-> 4301             loc = index.get_loc(key)
   4302 
   4303             if isinstance(loc, np.ndarray):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 0

## === cell 3
sns.set(style="white", palette="muted", color_codes=True)

if losses:
    plt.plot(losses, label="Train loss")
if valLosses:
    plt.plot(valLosses, label="Val loss")
if losses or valLosses:
    plt.legend()
    plt.show()

if len(trainDataSet) > 0:
    asd = pt.from_numpy(trainDataSet.data[inputs].values.astype(np.float32)).to(device)
    pred = model(asd).detach().cpu()
    idxs = np.random.randint(0, len(trainDataSet), min(100, len(trainDataSet)))
    plt.plot(
        trainDataSet.data.loc[idxs, ["FVC"]].values.astype(np.float32),
        label="ground truth",
    )
    plt.plot(pred[idxs, 0], label="q25")
    plt.plot(pred[idxs, 1], label="q50")
    plt.plot(pred[idxs, 2], label="q75")
    plt.legend(loc="best")
    plt.show()

c = pred[:, 2] - pred[:, 0]
f = pred[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.histplot(c, color="g", kde=False, ax=axes[0]).set_title(
    "Predicted Confidence on train set"
)
sns.histplot(f, color="g", kde=False, ax=axes[1]).set_title(
    "Predicted FVC on train set"
)
plt.show()

submDataSet = OSICDataSet(subms, "submission")
inp = pt.from_numpy(submDataSet.data[inputs].values.astype(np.float32)).to(device)
out = model(inp).detach().cpu()
confidence = out[:, 2] - out[:, 0]
fvc = out[:, 1]

submission = pd.DataFrame(
    {
        "Patient_Week": submDataSet.data.Patient_Week,
        "FVC": fvc,
        "Confidence": confidence,
    }
)

otest = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = f"{otest.Patient[i]}_{otest.Weeks[i]}"
    submission.loc[submission["Patient_Week"] == key, "FVC"] = otest.FVC[i]
    submission.loc[submission["Patient_Week"] == key, "Confidence"] = 0.1

submission.to_csv("submission.csv", index=False)

null_columns = submission.columns[submission.isnull().any()]
print(submission.head())
print(f"Submission rows: {len(submission)}")
print(f"Columns with null values: {list(null_columns)}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_54/3898531605.py in <cell line: 0>()
     15     idxs = np.random.randint(0, len(trainDataSet), min(100, len(trainDataSet)))
     16     plt.plot(
---> 17         trainDataSet.data.loc[idxs, ["FVC"]].values.astype(np.float32),
     18         label="ground truth",
     19     )

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1373         # ugly hack for GH #836
   1374         if self._multi_take_opportunity(tup):
-> 1375             return self._multi_take(tup)
   1376 
   1377         return self._getitem_tuple_same_dim(tup)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _multi_take(self, tup)
   1324         """
   1325         # GH 836
-> 1326         d = {
   1327             axis: self._get_listlike_indexer(key, axis)
   1328             for (key, axis) in zip(tup, self.obj._AXIS_ORDERS)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in <dictcomp>(.0)
   1325         # GH 836
   1326         d = {
-> 1327             axis: self._get_listlike_indexer(key, axis)
   1328             for (key, axis) in zip(tup, self.obj._AXIS_ORDERS)
   1329         }

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index([ 506, 1024,  584,  555,  288,  356,  747,   58,  238,  207,  903,  374,\n        648,   13,  171,  753,  920, 1068,  110,  429,  904, 1305, 1198,  641,\n       1023,  398,  319, 1257,  621,  584, 1294,  903,  725,  486,   54,   57,\n       1311,  733,  466,  514, 1277,  153,  574, 1107, 1250,  404, 1195,  543,\n        139,  853,  631,  657,  644, 1276,  940,  667,  596,  515,  384, 1140,\n        158, 1329,  266, 1244,   95,  496, 1045,  495,  819,  403,  406, 1138,\n       1014,  653,  864,  491, 1199,  751,   77,  806, 1131,   22, 1042,  475,\n        110, 1335,  108,  941,  208,  692, 1083,  246,  616,  239,  610,  753,\n        253,  237,  765,  922],\n      dtype='int64')] are in the [index]"
