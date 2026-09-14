# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
simpleitk==2.5.2
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from tqdm import tqdm
from os.path import join, exists
from pandas import read_csv, DataFrame
import SimpleITK as sitk

import os
import random
import numpy as np
import pandas as pd

import torch
from torch import cuda, device
from torch import Size, Tensor, no_grad, load
from torch.nn.functional import interpolate
import torch.nn as nn
import torch.jit as jit
from torch.utils.data import DataLoader
from torchvision.models.densenet import densenet121


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

KAGGLE_DATA = r"../input/rsna-2022-cervical-spine-fracture-detection"
N_SPLITS = 5
KERNEL_TYPE = "densenet121"
RESIZE_H = 150
RESIZE_W = 150
RESIZE_C = 50
BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
LR = 1e-5
OUT_DIM = 8
EPOCHS = 100

test_df = read_csv(join(KAGGLE_DATA, "test.csv"))
test_img_path = join(KAGGLE_DATA, "test_images")

print("test shape:", test_df.shape)

try:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(
        max(1, (os.cpu_count() or 4) // 2)
    )
except Exception:
    pass

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
except Exception:
    pass




## === cell 1
class CustomDenseNet(nn.Module):
    def __init__(self, inp_c, count_labels):
        super(CustomDenseNet, self).__init__()
        self.prep_layer = nn.Conv2d(
            in_channels=inp_c, out_channels=3, kernel_size=3, padding="same"
        )
        self.backbone = densenet121(num_classes=count_labels)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.backbone(x)
        x = self.out(x)
        return x


class RSNAStudyDataset:
    def __init__(self, study_uids, series_files, img_path, transform=None):
        self.study_uids = np.asarray(study_uids)
        self.series_files = series_files
        self.img_path = img_path
        self.transform = transform
        self._series_reader = None

    def __len__(self):
        return len(self.study_uids)

    def _get_series_reader(self):
        if self._series_reader is None:
            r = sitk.ImageSeriesReader()
            try:
                r.MetaDataDictionaryArrayUpdateOff()
                r.LoadPrivateTagsOff()
            except Exception:
                pass
            self._series_reader = r
        return self._series_reader

    @staticmethod
    def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        mn = float(arr.min())
        mx = float(arr.max())
        if mx <= mn:
            return np.zeros(arr.shape, dtype=np.uint8)
        out = (arr - mn) * (255.0 / (mx - mn))
        return out.clip(0.0, 255.0).astype(np.uint8, copy=False)

    def __getitem__(self, index):
        uid = self.study_uids[index]
        files = self.series_files[uid]

        sr = self._get_series_reader()
        sr.SetFileNames(files)
        img3d = sr.Execute()
        vol_np = sitk.GetArrayViewFromImage(img3d)  # (z,y,x)
        vol_u8 = self._rescale_to_uint8(vol_np)
        vol = torch.from_numpy(vol_u8).to(torch.float32)

        if self.transform:
            vol = self.transform(vol)
        return vol, uid


class TorchDeviceManager:
    def __init__(self):
        self.GPU_DEVICES = {}
        self.CPU_DEVICE = device("cpu")
        if cuda.is_available():
            print("TORCH: CUDA IS AVAILABLE\nGPU DEVICES:")
            for i in range(cuda.device_count()):
                self.GPU_DEVICES[i] = f"cuda:{i}"
                print(f"    {i}: {cuda.get_device_name(self.GPU_DEVICES[i])}")
        else:
            print("TORCH: CUDA IS NOT AVAILABLE")


class ValidationTransforms(jit.ScriptModule):
    def __init__(self, c, h, w):
        super().__init__()
        self.window = Size([c, h, w])

    @jit.script_method
    def forward(self, x):
        x = interpolate(x[None, None, :], size=self.window).clamp(min=0.0, max=255.0)[
            0, 0
        ]
        x = x / 255.0
        return x


def pd_factorize_stable(arr):
    codes, uniques = pd.factorize(arr, sort=False)
    return codes.astype(np.int64, copy=False), np.asarray(uniques, dtype=object)


class PipelineRSNA:
    def __init__(self, test_labels_path=None, test_img_path=None, test_transform=None):
        self.target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
        self.test_cols = ["row_id", "StudyInstanceUID", "prediction_type"]
        self.test_transform = test_transform
        self.target_map = {self.target_cols[i]: i for i in range(len(self.target_cols))}
        print(f"Load data from: {test_labels_path}")
        self.test_set = read_csv(test_labels_path)
        self.test_img_path = test_img_path

        self._ptype_to_idx = {k: v for k, v in self.target_map.items()}

        self._uids = self.test_set["StudyInstanceUID"].to_numpy()
        self._uid_codes, self._unique_uids = pd_factorize_stable(self._uids)

        self._predtype_idx = (
            self.test_set["prediction_type"].map(self._ptype_to_idx).to_numpy(np.int64)
        )

        self._study_start, self._study_count = self._compute_group_ranges(
            self._uid_codes
        )

        self._series_row_slices = list(
            zip(self._study_start.tolist(), self._study_count.tolist())
        )

        self._series_files = self._build_or_load_series_file_cache_fast()

    @staticmethod
    def _compute_group_ranges(uid_codes: np.ndarray):
        n = uid_codes.shape[0]
        if n == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        change = np.empty(n, dtype=bool)
        change[0] = True
        np.not_equal(uid_codes[1:], uid_codes[:-1], out=change[1:])
        starts = np.flatnonzero(change).astype(np.int64, copy=False)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = n
        counts = (ends - starts).astype(np.int64, copy=False)
        return starts, counts

    def _build_or_load_series_file_cache_fast(self):
        cache_path = "gdcm_series_files_cache.npz"
        if exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                keys = data["keys"]
                vals = data["vals"]
                series_files = {
                    k: list(v) for k, v in zip(keys.tolist(), vals.tolist())
                }
                if isinstance(series_files, dict) and len(series_files) > 0:
                    print(
                        f"Loaded series file cache: {cache_path} (entries={len(series_files)})"
                    )
                    return series_files
            except Exception:
                pass

        series_files = {}
        uids = self._unique_uids  # already unique, stable order

        for uid in tqdm(uids, desc="Index DICOM series"):
            folder = join(self.test_img_path, uid)
            names = []
            with os.scandir(folder) as it:
                for e in it:
                    if e.is_file() and e.name.endswith(".dcm"):
                        names.append(e.name)
            names.sort(key=lambda n: int(n[:-4]) if n[:-4].isdigit() else n)
            series_files[uid] = [join(folder, nm) for nm in names]

        try:
            np.savez_compressed(
                cache_path,
                keys=np.array(list(series_files.keys()), dtype=object),
                vals=np.array(list(series_files.values()), dtype=object),
            )
            print(
                f"Saved series file cache: {cache_path} (entries={len(series_files)})"
            )
        except Exception:
            pass
        return series_files

    @staticmethod
    def _collate_study(batch):
        imgs = [b[0].contiguous() for b in batch]
        uids = [b[1] for b in batch]
        return torch.stack(imgs, dim=0), uids

    def test(self, device, model_path=None):
        print(f"Test: {model_path}")

        model = CustomDenseNet(RESIZE_C, OUT_DIM).eval().to(device)

        if model_path is not None and exists(model_path):
            state = load(model_path, map_location=device)
            model.load_state_dict(state, strict=True)
            print("Loaded model weights.")
        else:
            print(
                "WARNING: model weights not found; running with randomly initialized model."
            )

        study_uids_in_order = self._unique_uids

        study_set = RSNAStudyDataset(
            study_uids=study_uids_in_order,
            series_files=self._series_files,
            img_path=self.test_img_path,
            transform=self.test_transform,
        )

        cpu_count = os.cpu_count() or 4
        num_workers = min(8, max(2, cpu_count // 2))

        g = torch.Generator()
        g.manual_seed(42)

        study_loader = DataLoader(
            study_set,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            collate_fn=self._collate_study,
            num_workers=num_workers,
            pin_memory=cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            generator=g,
        )

        out = np.empty(len(self.test_set), dtype=np.float64)

        with no_grad():
            code = 0
            for img_batch, _uids in tqdm(study_loader, desc="Infer studies"):
                img_batch = img_batch.to(device, non_blocking=True, dtype=torch.float32)
                pred_cpu = (
                    model(img_batch).clamp(1e-6, 1 - 1e-6).detach().cpu().numpy()
                )  # (B, 8)

                bsz = pred_cpu.shape[0]
                for i in range(bsz):
                    s, c = self._series_row_slices[code]
                    sl = slice(s, s + c)
                    out[sl] = pred_cpu[i, self._predtype_idx[sl]]
                    code += 1

        sub = DataFrame(
            {"row_id": self.test_set["row_id"].to_numpy(), "fractured": out}
        )
        sub.to_csv("submission.csv", index=False)
        print("Wrote submission.csv with rows:", len(sub))




## === cell 2
transform = ValidationTransforms(RESIZE_C, RESIZE_H, RESIZE_W)

pipe = PipelineRSNA(
    test_labels_path=join(KAGGLE_DATA, "test.csv"),
    test_img_path=join(KAGGLE_DATA, "test_images"),
    test_transform=transform,
)

pipe.test(
    device("cuda" if cuda.is_available() else "cpu"),
    model_path=r"../input/baseline-pretrained/densenet121_e4.pt",
)
