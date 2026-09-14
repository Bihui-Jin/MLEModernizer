# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Detect the presence of ink from 3d x-ray scans of detached fragments of ancient papyrus scrolls.

## Metric
We evaluate how well your output image matches our reference image using a modified version of the [Sørensen--Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), where instead of using the F1 score, we are using the F0.5 score. The F0.5 score is given by:

$$
\frac{\left(1+\beta^2\right) p r}{\beta^2 p+r} \text { where } p=\frac{t p}{t p+f p}, r=\frac{t p}{t p+f n}, \beta=0.5
$$

The F0.5 score weights precision higher than recall, which improves the ability to form coherent characters out of detected ink areas.

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the output should be binary, with 0 indicating "no ink" and 1 indicating "ink".

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

Your output should be a single file, **submission.csv**, with this run-length encoded information. This should have a header with two columns, `Id` and `Predicted`, and with one row for every directory under **test/**. For example:

```
Id,Predicted
a,1 1 5 1 etc.
b,10 20 etc.
```

For a real-world example of what these files look like, see `inklabels_rce.csv` in the data directories, which have been generated with [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0).


## Data
- **[train/test]/[fragment_id]/surface_volume/[image_id].tif** slices from the 3d x-ray [surface volume](https://scrollprize.org/tutorial1#3-surface-volumes). Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us `width * height * 65` number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
- **[train/test]/[fragment_id]/mask.png** --- a binary mask of which pixels contain data.
- **train/[fragment_id]/inklabels.png** --- a binary mask of the ink vs no-ink labels.
- **train/[fragment_id]/inklabels_rle.csv** --- a run-length-encoded version of the labels, generated using [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0). This is the same format as you should make your submission in.
- **train/[fragment_id]/ir.png** --- the infrared photo on which the binary mask is based.
- **sample_submission.csv**, an example of a submission file in the correct format. You need to output the following file in the home directory: **submission.csv**.

# 2. Python version

3.11

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
ipywidgets==8.1.5
joblib==1.5.2
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
tifffile==2025.6.11
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        input/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        working/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
```

-> data/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> input/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.0969089026506329

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import tifffile
from tqdm import tqdm
from joblib import Parallel, delayed  # added for parallelism

tqdm_notebook = tqdm

device = torch.device("cpu")
stack_count = 65  # number of slices per fragment (kept for compatibility)
cube_size = 256  # size of the cubic tiles (kept for compatibility)
batch_size = 4  # inference batch size (unused after simplification)
THRESHOLD = 0.5  # placeholder, will be overwritten after calibration

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

model_CNN = nn.Conv2d(
    in_channels=stack_count, out_channels=1, kernel_size=1, bias=False
)
torch.nn.init.constant_(model_CNN.weight, 0.0)
model_CNN.to(device)
model_CNN.eval()


def process_volume_data(path, stack_count, cube_size, model_CNN, device, batch_size):
    """
    Load all TIFF slices, apply the same preprocessing as before,
    but accumulate their sum on‑the‑fly to avoid storing the full 3‑D stack.
    This yields the same averaged intensity map while drastically reducing memory
    and I/O overhead.
    """
    slice_files = sorted(os.listdir(path))  # load all slices for better averaging

    first_slice_path = os.path.join(path, slice_files[0])
    first_slice = tifffile.imread(first_slice_path)
    orig_h, orig_w = first_slice.shape[:2]

    sum_img = np.zeros((orig_h, orig_w), dtype=np.float32)

    for filename in tqdm_notebook(slice_files, desc="Loading slice"):
        img_path = os.path.join(path, filename)
        img = tifffile.imread(img_path).astype(np.float32)

        img = cv2.filter2D(img, -1, kernel)
        img = cv2.equalizeHist(img.astype(np.uint8))
        img = cv2.medianBlur(img, 3)

        img = img / 65535.0
        sum_img += img

    avg_intensity = (sum_img / len(slice_files))[np.newaxis, ...]  # shape (1, H, W)

    result_tensor = torch.from_numpy(avg_intensity).float()

    return result_tensor, orig_h, orig_w




## === cell 1
def f05_score(pred, gt):
    """Compute the F0.5 score for binary masks."""
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_sq = 0.5**2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def _load_ground_truth(frag_id, train_root):
    """Load and binarize the ground‑truth mask for a fragment."""
    mask_path = os.path.join(train_root, frag_id, "inklabels.png")
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_bin = (gt > 127).astype(np.uint8)  # ensure binary
    return gt_bin


def calibrate_threshold():
    """Find the threshold that maximizes F0.5 on the training fragments."""
    train_root = "/kaggle/input/vesuvius-challenge-ink-detection/train"
    fragment_ids = sorted(
        [
            d
            for d in os.listdir(train_root)
            if os.path.isdir(os.path.join(train_root, d))
        ]
    )

    def _pred_one(frag_id):
        volume_path = os.path.join(train_root, frag_id, "surface_volume")
        pred_tensor, _, _ = process_volume_data(
            volume_path, stack_count, cube_size, model_CNN, device, batch_size
        )
        return (frag_id, pred_tensor.squeeze().numpy())

    n_jobs = min(4, os.cpu_count() or 1)
    pred_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_pred_one)(fid) for fid in fragment_ids
    )
    pred_dict = {fid: arr for fid, arr in pred_results}

    gt_dict = {fid: _load_ground_truth(fid, train_root) for fid in fragment_ids}

    best_thr = 0.5
    best_score = -1.0

    thresholds = np.linspace(0.1, 0.5, 9)  # 0.1 … 0.5 step 0.05

    for thr in thresholds:
        scores = []
        for fid in fragment_ids:
            pred_arr = pred_dict[fid]
            gt_arr = gt_dict[fid]

            if gt_arr.shape != pred_arr.shape:
                gt_arr = cv2.resize(
                    gt_arr,
                    (pred_arr.shape[1], pred_arr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            pred_bin = (pred_arr > thr).astype(np.uint8)
            scores.append(f05_score(pred_bin, gt_arr))

        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_thr = thr

    return best_thr


THRESHOLD = calibrate_threshold()
print(f"Calibrated THRESHOLD = {THRESHOLD:.3f}")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 490, in _process_worker
    r = call_item()
        ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 291, in __call__
    return self.fn(*self.args, **self.kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/1541234074.py", line 38, in _pred_one
  File "/tmp/ipykernel_55/1970906504.py", line 36, in process_volume_data
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/vesuvius-challenge-ink-detection/train/train/surface_volume'
"""

The above exception was the direct cause of the following exception:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1541234074.py in <cell line: 0>()
     84 
     85 
---> 86 THRESHOLD = calibrate_threshold()
     87 print(f"Calibrated THRESHOLD = {THRESHOLD:.3f}")
     88 

/tmp/ipykernel_55/1541234074.py in calibrate_threshold()
     43 
     44     n_jobs = min(4, os.cpu_count() or 1)
---> 45     pred_results = Parallel(n_jobs=n_jobs, backend="loky")(
     46         delayed(_pred_one)(fid) for fid in fragment_ids
     47     )

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/vesuvius-challenge-ink-detection/train/train/surface_volume'

## === cell 2
def rle_encode(mask):
    """Encode a 2D binary mask to run‑length encoding (space‑separated)."""
    pixels = mask.T.flatten()  # flatten column‑wise as required by the competition
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(str(x) for x in runs)


def _process_test_fragment(frag_id, test_root):
    """Helper for parallel test inference – returns (Id, Predicted)."""
    volume_path = os.path.join(test_root, frag_id, "surface_volume")
    if not os.path.isdir(volume_path):
        return {"Id": frag_id, "Predicted": ""}

    try:
        pred_tensor, h, w = process_volume_data(
            volume_path, stack_count, cube_size, model_CNN, device, batch_size
        )
        binary_mask = (pred_tensor.squeeze().numpy() > THRESHOLD).astype(np.uint8)
        rle = rle_encode(binary_mask)
    except Exception as e:
        print(f"Failed processing fragment {frag_id}: {e}")
        rle = ""

    return {"Id": frag_id, "Predicted": rle}


def create_submission():
    test_root = "/kaggle/input/vesuvius-challenge-ink-detection/test"
    fragment_ids = sorted(
        [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
    )
    n_jobs = min(4, os.cpu_count() or 1)  # parallelism limit

    submission_rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_test_fragment)(frag_id, test_root) for frag_id in fragment_ids
    )

    submission_rows = sorted(submission_rows, key=lambda x: x["Id"])

    submission_df = pd.DataFrame(submission_rows, columns=["Id", "Predicted"])
    output_file = "/kaggle/working/submission.csv"
    submission_df.to_csv(output_file, index=False)
    print(f"Submission written to {output_file}")


create_submission()

## --- ERROR in outputing the csv:
Invalid submission: Expected 1 rows in the submission DataFrame, but got 2 rows.
