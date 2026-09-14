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
Identify technosignature signals in cadence snippets taken from a digital spectrometer.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
00034abb3629,0.5
0004be0baf70,0.5
0005be4d0752,0.5
etc.

```

## Dataset
The data is from a digital spectrometer, which takes incoming raw data from the telescope (amounting to hundreds of TB per day) and performs a Fourier Transform to generate a spectrogram. These spectrograms, also referred to as filterbank files, or dynamic spectra, consist of measurements of signal intensity as a function of frequency and time.

Below is an example of an FM radio signal. This is not from the GBT, but from a small antenna attached to a software defined radio dongle (a $20 piece of kit that you can plug into your laptop to pick up signals). The data we get from the GBT are very similar, but split into larger numbers of frequency channels, covering a much broader instantaneous frequency range, and with much better sensitivity.

![frequency-time-plot](https://prod-files-secure.s3.us-west-2.amazonaws.com/667f1cbf-826f-4641-a321-96054292638d/b59a57f3-11a7-4493-8268-55c3fa632f7e/Untitled.png)

The screenshot above shows frequency on the horizontal axis (running from around 88.2 to 89.8 MHz) and time on the vertical axis. The bright orange feature at 88.5 MHz is the FM signal from KQED, a radio station in the San Francisco Bay Area. The solid yellow blocks on either side (one highlighted by the pointer in the screenshot) are the KQED “HD radio” signal (the same data as the FM signal, but encoded digitally). Additional FM stations are visible at different frequencies, including another obvious FM signal (without the corresponding digital sidebands) at 89.5 MHz.

The spectrometer generates similar spectrograms to the one shown above, but typically spanning several GHz of the radio spectrum (rather than the approx. 2 MHz shown above). The data are stored either as filterbank format or HDF5 format files, but essentially are arrays of intensity as a function of frequency and time, accompanied by headers containing metadata such as the direction the telescope was pointed in, the frequency scale, and so on. We generate over 1 PB of spectrograms per year; individual filterbank files can be tens of GB in size. We have discarded the majority of the metadata and are simply presenting numpy arrays consisting of small regions of the spectrograms that we refer to as “snippets”.

The spectrometer is searching for candidate signatures of extraterrestrial technology - so-called technosignatures. The main obstacle to doing so is that our own human technology (not just radio stations, but wifi routers, cellphones, and even electronics that are not deliberately designed to transmit radio signals) also gives off radio signals. We refer to these human-generated signals as “radio frequency interference”, or RFI.

One method we use to isolate candidate technosignatures from RFI is to look for signals that appear to be coming from particular positions on the sky. Typically we do this by alternating observations of our primary target star with observations of three nearby stars: 5 minutes on star “A”, then 5 minutes on star “B”, then back to star “A” for 5 minutes, then “C”, then back to “A”, then finishing with 5 minutes on star “D”. One set of six observations (ABACAD) is referred to as a “cadence”. Since we're just giving you a small range of frequencies for each cadence, we refer to the datasets you'll be analyzing as “cadence snippets”.

An example of an extraterrestrial signal:

![voyager-signal](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.39.42.png)

As the plot title suggests, this is the Voyager 1 spacecraft. Even though it's 20 billion kilometers from Earth, it's picked up clearly by the GBT. The first, third, and fifth panels are the “A” target (the spacecraft, in this case). The yellow diagonal line is the radio signal coming from Voyager. It's detected when we point at the spacecraft, and it disappears when we point away. It's a diagonal line in this plot because the relative motion of the Earth and the spacecraft imparts a Doppler drift, causing the frequency to change over time. As it happens, that's another possible way to reject RFI, which has a higher tendency to remain at a fixed frequency over time.

While it would be nice to train our algorithms entirely on observations of interplanetary spacecraft, there are not many examples of them, and we also want to be able to find a wider range of signal types. So we've turned to simulating technosignature candidates.

We've taken tens of thousands of cadence snippets, which we're calling the haystack, and we've hidden needles among them. Some of these needles look similar to the Voyager 1 signal above and should be easy to detect, even with classical detection algorithms. Others are hidden in noisy regions of the spectrum and will be harder, even though they might be relatively obvious on visual inspection:

![needle-signal](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.34.06.png)

After we perform the signal injections, we normalize each snippet, so you probably can't identify most of the needles just by looking for excess energy in the corresponding array. You'll likely need a more subtle algorithm that looks for patterns that appear only in the on-target observations.

Not all of the “needle” signals look like diagonal lines, and they may not be present for the entirety of all three “A” observations, but what they do have in common is that they are only present in some or all of the “A” observations (panels 1, 3, and 5 in the cadence snippets). Your challenge is to train an algorithm to find as many needles as you can, while minimizing the number of false positives from the haystack.

- **train/** - a training set of cadence snippet files stored in `numpy` `float16` format (v1.20.1), one file per cadence snippet `id`, with corresponding labels found in the `train_labels.csv` file. Each file has dimension `(6, 273, 256)`, with the 1st dimension representing the 6 positions of the cadence, and the 2nd and 3rd dimensions representing the 2D spectrogram.
- **test/** - the test set cadence snippet files; you must predict whether or not the cadence contains a "needle", which is the `target` for this competition
- **sample_submission.csv** - a sample submission file in the correct format
- **train_labels** - targets corresponding (by `id`) to the cadence snippet files found in the `train/` folder
- **old_leaky_data** - full pre-relaunch data, including test labels; you should not assume this data is helpful (it may or may not be).

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
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (112 lines)
            old_leaky_data.zip (23.6 GB)
            sample_submission.csv (6001 lines)
            sample_submission.csv.zip (60.0 kB)
            test.zip (4.5 GB)
            train.zip (4.7 GB)
            train_labels.csv (54001 lines)
            train_labels.csv.zip (529.5 kB)
            old_leaky_data/
                test_labels_old.csv (35848 lines)
                train_labels_old.csv (50166 lines)
                test_old/
                    0/
                        00034db451c4.npy (838.8 kB)
                        0006316b5ca0.npy (838.8 kB)
                        ... and 2197 other files
                    1/
                        10038983cab1.npy (838.8 kB)
                        100865aff453.npy (838.8 kB)
                        ... and 2278 other files
                    ... and 14 other folders
                train_old/
                    0/
                        00034abb3629.npy (838.8 kB)
                        0004300a0b9b.npy (838.8 kB)
                        ... and 3143 other files
                    1/
                        1000e00b26db.npy (838.8 kB)
                        100148224705.npy (838.8 kB)
                        ... and 3142 other files
                    ... and 14 other folders
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
            test/
                0/
                    0016fd6c09d476d.npy (838.8 kB)
                    0017643c1c5c254.npy (838.8 kB)
                    ... and 374 other files
                1/
                    1001ca1d08f9235.npy (838.8 kB)
                    1016de9cec2dc8a.npy (838.8 kB)
                    ... and 353 other files
                ... and 15 other folders
            train/
                0/
                    0000799a2b2c42d.npy (838.8 kB)
                    00042890562ff68.npy (838.8 kB)
                    ... and 3335 other files
                1/
                    100105755d4c5b1.npy (838.8 kB)
                    1001a55ebce86f2.npy (838.8 kB)
                    ... and 3392 other files
                ... and 15 other folders
        input/
            description.md (112 lines)
            old_leaky_data.zip (23.6 GB)
            sample_submission.csv (6001 lines)
            sample_submission.csv.zip (60.0 kB)
            test.zip (4.5 GB)
            train.zip (4.7 GB)
            train_labels.csv (54001 lines)
            train_labels.csv.zip (529.5 kB)
            old_leaky_data/
                test_labels_old.csv (35848 lines)
                train_labels_old.csv (50166 lines)
                test_old/
                    0/
                        00034db451c4.npy (838.8 kB)
                        0006316b5ca0.npy (838.8 kB)
                        ... and 2197 other files
                    1/
                        10038983cab1.npy (838.8 kB)
                        100865aff453.npy (838.8 kB)
                        ... and 2278 other files
                    ... and 14 other folders
                train_old/
                    0/
                        00034abb3629.npy (838.8 kB)
                        0004300a0b9b.npy (838.8 kB)
                        ... and 3143 other files
                    1/
                        1000e00b26db.npy (838.8 kB)
                        100148224705.npy (838.8 kB)
                        ... and 3142 other files
                    ... and 14 other folders
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
            test/
                0/
                    0016fd6c09d476d.npy (838.8 kB)
                    0017643c1c5c254.npy (838.8 kB)
                    ... and 374 other files
                1/
                    1001ca1d08f9235.npy (838.8 kB)
                    1016de9cec2dc8a.npy (838.8 kB)
                    ... and 353 other files
                ... and 15 other folders
            train/
                0/
                    0000799a2b2c42d.npy (838.8 kB)
                    00042890562ff68.npy (838.8 kB)
                    ... and 3335 other files
                1/
                    100105755d4c5b1.npy (838.8 kB)
                    1001a55ebce86f2.npy (838.8 kB)
                    ... and 3392 other files
                ... and 15 other folders
        working/
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
```

-> data/old_leaky_data/test_labels_old.csv has 35847 rows and 2 columns.
The columns are: id, target

-> data/old_leaky_data/train_labels_old.csv has 50165 rows and 2 columns.
The columns are: id, target

-> data/sample_submission.csv has 6000 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/old_leaky_data/test_labels_old.csv has 35847 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/old_leaky_data/train_labels_old.csv has 50165 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/sample_submission.csv has 6000 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/train_labels.csv has 54000 rows and 2 columns.
The columns are: id, target

-> data/train_labels.csv has 54000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.7571666913491772

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The notebook fails because it tries to read submission files from other Kaggle datasets that are not present in your environment, so none of the `data1..data6` DataFrames get created and everything downstream errors. I keep the ensemble logic structure but make it robust: load whichever external submissions exist if available, otherwise fall back to a safe baseline using `sample_submission.csv` with a constant probability. I also ensure all frames are aligned by `id` before blending (prevents silent row-order mismatches) and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by the fallback path, which loads and featurizes every one of the 54k training `.npy` files sequentially; this is pure I/O + NumPy reduction work and easily exceeds 600s. To keep the exact same fallback model/feature logic, I (1) make the “need_fallback” check much cheaper and correct by pre-indexing external submissions and avoiding repeated merges, and (2) parallelize feature extraction with a process pool while keeping the exact same feature computations. I also avoid any redundant dtype conversions/allocations inside `extract_features` and minimize per-file overhead, while leaving the logistic regression and ensemble weights unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd




## === cell 1
def try_read_submission(path: str):
    if path is None:
        return None
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "id" not in df.columns or "target" not in df.columns:
            return None
        df = df[["id", "target"]].copy()
        df["id"] = df["id"].astype(str)
        df["target"] = (
            pd.to_numeric(df["target"], errors="coerce").fillna(0.5).astype(float)
        )
        return df
    return None


sample_path = "/kaggle/input/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/data/sample_submission.csv"
sample = pd.read_csv(sample_path)[["id", "target"]].copy()
sample["id"] = sample["id"].astype(str)



## === cell 2
data1 = try_read_submission(
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv"
)
data2 = try_read_submission("../input/seti-bl-spatial-info-tf-tpu/submission.csv")
data3 = try_read_submission("../input/seti-bl-tf-starter-tpu/submission.csv")
data4 = try_read_submission("../input/seti-learned-image-resizing/submission.csv")
data5 = try_read_submission(
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv"
)
data6 = try_read_submission(
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv"
)

available = {
    "data1": data1 is not None,
    "data2": data2 is not None,
    "data3": data3 is not None,
    "data4": data4 is not None,
    "data5": data5 is not None,
    "data6": data6 is not None,
}
print("Available external submissions:", available)



## === cell 3
from sklearn.linear_model import LogisticRegression

train_labels_path = "/kaggle/input/train_labels.csv"
if not os.path.exists(train_labels_path):
    train_labels_path = "/kaggle/data/train_labels.csv"
train_labels = pd.read_csv(train_labels_path)
train_labels["id"] = train_labels["id"].astype(str)

train_root = "/kaggle/input/train"
test_root = "/kaggle/input/test"
if not os.path.exists(train_root):
    train_root = "/kaggle/data/train"
if not os.path.exists(test_root):
    test_root = "/kaggle/data/test"


def id_to_path(root: str, id_: str) -> str:
    return os.path.join(root, id_[0], f"{id_}.npy")


_A_IDX = np.array([0, 2, 4])
_B_IDX = np.array([1, 3, 5])


def extract_features(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)
    A = x[_A_IDX]
    B = x[_B_IDX]

    feat = np.empty(13, dtype=np.float32)
    feat[0] = x.mean()
    feat[1] = x.std()
    feat[2] = np.median(x)
    feat[3] = A.mean()
    feat[4] = A.std()
    feat[5] = B.mean()
    feat[6] = B.std()
    feat[7] = feat[3] - feat[5]
    feat[8] = feat[4] - feat[6]
    feat[9] = np.mean(np.abs(A) - np.abs(B))
    feat[10] = np.percentile(A, 99) - np.percentile(B, 99)
    feat[11] = np.percentile(A, 95) - np.percentile(B, 95)
    feat[12] = np.mean(A * A) - np.mean(B * B)
    return feat


def build_feature_matrix(ids, root: str) -> np.ndarray:
    from concurrent.futures import ProcessPoolExecutor

    ids = list(ids)

    def _one(id_):
        p = id_to_path(root, id_)
        arr = np.load(p, mmap_mode="r")
        return extract_features(arr)

    max_workers = min(8, (os.cpu_count() or 2))
    X = np.empty((len(ids), 13), dtype=np.float32)

    chunksize = max(16, len(ids) // (max_workers * 8) if len(ids) else 16)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i, feat in enumerate(ex.map(_one, ids, chunksize=chunksize)):
            X[i] = feat
    return X


def _needs_fallback_for_any_id(dfs, sample_ids: pd.Series) -> bool:
    sample_ids = sample_ids.astype(str)
    sample_list = sample_ids.tolist()
    sample_set = set(sample_list)

    for df in dfs:
        if df is None:
            return True
        if df.shape[0] != len(sample_list):
            return True

        if df["id"].equals(sample_ids):
            if df["target"].isna().any():
                return True
            continue

        df_ids = df["id"].astype(str)
        if len(set(df_ids.tolist())) != len(sample_list):
            return True
        if not sample_set.issubset(set(df_ids.tolist())):
            return True

        s = pd.Series(df["target"].values, index=df_ids.values)
        if s.reindex(sample_list).isna().any():
            return True

    return False


external_list = [data1, data2, data3, data4, data5, data6]
need_fallback = _needs_fallback_for_any_id(external_list, sample["id"])
print("Need fallback model:", need_fallback)

if need_fallback:
    train_ids = train_labels["id"].tolist()
    y = train_labels["target"].astype(int).values

    X_train = build_feature_matrix(train_ids, train_root)

    clf = LogisticRegression(
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        random_state=0,
        max_iter=300,
    )
    clf.fit(X_train, y)

    test_ids = sample["id"].tolist()
    X_test = build_feature_matrix(test_ids, test_root)
    fallback_pred = clf.predict_proba(X_test)[:, 1].astype(float)

    fallback_frame = sample.copy()
    fallback_frame["target"] = np.clip(fallback_pred, 0.0, 1.0)

    print(
        "Fallback model predictions summary:",
        pd.Series(fallback_frame["target"]).describe().to_dict(),
    )
else:
    fallback_frame = sample.copy()
    fallback_frame["target"] = 0.5




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
AttributeError: Can't pickle local object 'build_feature_matrix.<locals>._one'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4010320124.py in <cell line: 0>()
    115     y = train_labels["target"].astype(int).values
    116 
--> 117     X_train = build_feature_matrix(train_ids, train_root)
    118 
    119     clf = LogisticRegression(

/tmp/ipykernel_11/4010320124.py in build_feature_matrix(ids, root)
     68 
     69     with ProcessPoolExecutor(max_workers=max_workers) as ex:
---> 70         for i, feat in enumerate(ex.map(_one, ids, chunksize=chunksize)):
     71             X[i] = feat
     72     return X

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/multiprocessing/queues.py in _feed(buffer, notempty, send_bytes, writelock, reader_close, writer_close, ignore_epipe, onerror, queue_sem)
    242 
    243                         # serialize the data before acquiring the lock
--> 244                         obj = _ForkingPickler.dumps(obj)
    245                         if wacquire is None:
    246                             send_bytes(obj)

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object 'build_feature_matrix.<locals>._one'

## === cell 4
def ensure_frame(df):
    if df is None:
        return fallback_frame.copy()
    s = pd.Series(df["target"].values, index=df["id"].values)
    out = sample[["id"]].copy()
    out["target"] = s.reindex(out["id"].values).to_numpy()
    out["target"] = np.where(
        np.isnan(out["target"]), fallback_frame["target"].to_numpy(), out["target"]
    ).astype(float)
    return out


data1 = ensure_frame(data1)
data2 = ensure_frame(data2)
data3 = ensure_frame(data3)
data4 = ensure_frame(data4)
data5 = ensure_frame(data5)
data6 = ensure_frame(data6)

assert len(data1) == len(sample) == 6000
for d in (data1, data2, data3, data4, data5, data6):
    assert d["id"].equals(sample["id"])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1838938660.py in <cell line: 0>()
     13 
     14 
---> 15 data1 = ensure_frame(data1)
     16 data2 = ensure_frame(data2)
     17 data3 = ensure_frame(data3)

/tmp/ipykernel_11/1838938660.py in ensure_frame(df)
      3 def ensure_frame(df):
      4     if df is None:
----> 5         return fallback_frame.copy()
      6     s = pd.Series(df["target"].values, index=df["id"].values)
      7     out = sample[["id"]].copy()

NameError: name 'fallback_frame' is not defined

## === cell 5
data1.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2462497009.py in <cell line: 0>()
----> 1 data1.head()
      2 

AttributeError: 'NoneType' object has no attribute 'head'

## === cell 6
data2.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2979661811.py in <cell line: 0>()
----> 1 data2.head()
      2 

AttributeError: 'NoneType' object has no attribute 'head'

## === cell 7
pred = sample.copy()
pred["target"] = (
    0.76 * data5["target"].values
    + 0.13 * data4["target"].values
    + 0.11 * data1["target"].values
    + 0.00 * data2["target"].values
    + 0.00 * data3["target"].values
)

pred["target"] = pred["target"].clip(0.0, 1.0)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1972799730.py in <cell line: 0>()
      1 pred = sample.copy()
      2 pred["target"] = (
----> 3     0.76 * data5["target"].values
      4     + 0.13 * data4["target"].values
      5     + 0.11 * data1["target"].values

TypeError: 'NoneType' object is not subscriptable

## === cell 8
pred.to_csv("submission.csv", index=False)
print(pred.head())
print("Wrote submission.csv with shape:", pred.shape)
print("Prediction summary:", pd.Series(pred["target"]).describe().to_dict())
