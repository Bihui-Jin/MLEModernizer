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

0.0909200423095562

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
import glob
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from ipywidgets import interact, fixed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from scipy.ndimage import morphology
from torch.utils.data import random_split
from torchvision.transforms import ToTensor, Normalize
import gc
from torch.utils.data import TensorDataset, SubsetRandomSampler
import tifffile

gc.collect()




## === cell 1
learning_rate = 0.0001
batch_size = 32
cube_size = 128
epochs = 1000
seed = 42
stack_count = 32
seed_everything(seed)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
block_size = cube_size

rle_threshold = 0.5  # default value; adjust if needed




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3202380995.py in <cell line: 0>()
      5 seed = 42
      6 stack_count = 32
----> 7 seed_everything(seed)
      8 kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
      9 block_size = cube_size

NameError: name 'seed_everything' is not defined

## === cell 2
def _choose_test_path():
    """Return the first existing test directory from a list of candidates."""
    candidates = [
        "/kaggle/input/vesuvius-challenge-ink-detection/test",
        "./input/vesuvius-challenge-ink-detection/test",
        "./vesuvius-challenge-ink-detection/test",
        "./test",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("No valid test directory found among candidates.")


base_test_path = _choose_test_path()


def process_volume_data(path, stack_count, cube_size, model_CNN, device, batch_size):
    """Read a fragment's surface volume, run the UNet on tiled cubes and return the
    reconstructed probability map."""
    if not os.path.isdir(path):
        return np.zeros((1, 1), dtype=np.float32)
    files = sorted(os.listdir(path))
    if len(files) == 0:
        return np.zeros((1, 1), dtype=np.float32)

    first_file = files[0]
    sample_img = tifffile.imread(os.path.join(path, first_file))
    if sample_img.ndim == 3:
        sample_img = sample_img[..., 0]
    h, w = sample_img.shape

    stack_np = np.zeros((stack_count, h, w), dtype=np.float32)
    for idx, filename in enumerate(files[:stack_count]):
        slice_path = os.path.join(path, filename)
        slice_data = tifffile.imread(slice_path)
        if slice_data.ndim == 3:
            slice_data = slice_data[..., 0]
        stack_np[idx] = slice_data.astype(np.float32) / 255.0

    pad_h = (cube_size - h % cube_size) % cube_size
    pad_w = (cube_size - w % cube_size) % cube_size
    if pad_h or pad_w:
        stack_np = np.pad(
            stack_np,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
        )
    padded_h, padded_w = stack_np.shape[1], stack_np.shape[2]

    volume_tensor = torch.from_numpy(stack_np).to(device)  # (C, H, W)
    recon = torch.zeros((1, padded_h, padded_w), dtype=torch.float32, device=device)

    block_coords = [
        (i, j)
        for i in range(0, padded_h, cube_size)
        for j in range(0, padded_w, cube_size)
    ]

    with torch.no_grad():
        for batch_start in range(0, len(block_coords), batch_size):
            batch_coords = block_coords[batch_start : batch_start + batch_size]
            batch_blocks = []
            for y, x in batch_coords:
                blk = volume_tensor[:, y : y + cube_size, x : x + cube_size].unsqueeze(
                    0
                )
                batch_blocks.append(blk)
            batch_tensor = torch.cat(batch_blocks, dim=0)  # (B, C, H, W)

            if device == "cuda" and batch_tensor.dtype != torch.float16:
                batch_tensor = batch_tensor.half()

            outputs = model_CNN(batch_tensor)  # (B, 1, H, W)
            for out, (y, x) in zip(outputs, batch_coords):
                recon[:, y : y + cube_size, x : x + cube_size] = out.squeeze(1)

    recon = recon[:, :h, :w].squeeze(0).cpu().numpy().astype(np.float32)
    return recon


def rle_from_prob(img, thresh=rle_threshold):
    """Convert a probability map to run‑length encoding using the configurable threshold."""
    binary = (img >= thresh).astype(np.uint8)
    flat = binary.ravel()
    padded = np.concatenate([[0], flat, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0] + 1  # 1‑based indexing
    ends = np.where(diff == -1)[0] + 1
    lengths = ends - starts
    return starts, lengths


fragment_ids = sorted(
    [
        d
        for d in os.listdir(base_test_path)
        if os.path.isdir(os.path.join(base_test_path, d))
    ]
)

submission_rows = []
for frag_id in fragment_ids:
    vol_path = os.path.join(base_test_path, frag_id, "surface_volume")
    prob_map = process_volume_data(
        vol_path,
        stack_count,
        cube_size,
        model_CNN,
        device,
        batch_size,
    )
    starts, lengths = rle_from_prob(prob_map)  # uses rle_threshold
    rle_string = " ".join(map(str, itertools.chain.from_iterable(zip(starts, lengths))))
    submission_rows.append((frag_id, rle_string))

submission_path = os.path.join(output_path, "submission.csv")
with open(submission_path, "w") as f:
    f.write("Id,Predicted\n")
    for fid, rle_str in submission_rows:
        f.write(f"{fid},{rle_str}\n")

print(f"Results written to {submission_path}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/652095161.py in <cell line: 0>()
     13 
     14 
---> 15 base_test_path = _choose_test_path()
     16 
     17 

/tmp/ipykernel_55/652095161.py in _choose_test_path()
      8     ]
      9     for p in candidates:
---> 10         if os.path.isdir(p):
     11             return p
     12     raise FileNotFoundError("No valid test directory found among candidates.")

NameError: name 'os' is not defined

## === cell 3
print(pd.read_csv(os.path.join(output_path, "submission.csv")))

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3736833742.py in <cell line: 0>()
----> 1 print(pd.read_csv(os.path.join(output_path, "submission.csv")))

NameError: name 'pd' is not defined
