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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.041385369226159

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_MODE", "offline")



## === cell 1
try:
    import pylibjpeg  # noqa: F401
except Exception:
    print("pylibjpeg not available; proceeding with fallback DICOM decoding.")



## === cell 2
import gc
import glob
import os
from functools import lru_cache

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torchvision as tv
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm.notebook import tqdm

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
plt.rcParams["figure.figsize"] = (20, 5)

WEIGHTS = tv.models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT
RSNA_2022_PATH = "/kaggle/input/rsna-breast-cancer-detection"
TRAIN_IMAGES_PATH = f"{RSNA_2022_PATH}/train_images"
TEST_IMAGES_PATH = f"{RSNA_2022_PATH}/test_images"
EFFNET_CHECKPOINTS_PATH = "/kaggle/input/breast-cancer-models"

MODEL_NAMES = [f"effnetv2-f{i}" for i in range(5)]

try:
    from kaggle_secrets import UserSecretsClient  # noqa: F401

    IS_KAGGLE = True
except Exception:
    IS_KAGGLE = False

os.environ["WANDB_MODE"] = "offline"

if not IS_KAGGLE:
    print("Running locally")
    RSNA_2022_PATH = "/mnt/rsna2022"
    TRAIN_IMAGES_PATH = "/mnt/rsna2022/train_images"
    TEST_IMAGES_PATH = "/mnt/rsna2022/test_images"
    METADATA_PATH = "/home/vslaykovsky/Downloads/"
    EFFNET_CHECKPOINTS_PATH = "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 2



## === cell 3
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
torch.backends.cudnn.benchmark = False




## === cell 4
def load_df_test():
    df_test = pd.read_csv(f"{RSNA_2022_PATH}/test.csv")
    return df_test


df_test = load_df_test()
df_test.head()



## === cell 5
df_sample = pd.read_csv(f"{RSNA_2022_PATH}/sample_submission.csv")
assert list(df_sample.columns) == ["prediction_id", "cancer"]
df_sample.head()




## === cell 6
def _dicom_to_uint8(ds: dicom.Dataset) -> np.ndarray:
    """
    Robust DICOM -> uint8 image conversion:
    - Avoid forcing PhotometricInterpretation changes (can trigger pydicom color conversion errors).
    - Apply RescaleSlope/Intercept when present.
    - Use a robust percentile window then map to [0, 255].
    """
    arr = ds.pixel_array  # may raise if compressed decoding not supported
    arr = np.asarray(arr)

    if arr.ndim == 3:
        arr = arr[0]

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr.astype(np.float32) * slope + intercept

    phot = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if phot == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = np.percentile(arr, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(arr.min()) if arr.size else 0.0
        hi = float(arr.max()) if arr.size else 1.0
        if hi <= lo:
            hi = lo + 1.0

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-6)
    arr = (arr * 255.0).astype(np.uint8)
    return arr


@lru_cache(maxsize=8192)
def _load_dicom_rgb_cached(path: str) -> np.ndarray:
    ds = dicom.dcmread(path, stop_before_pixels=False, force=True)
    data_u8 = _dicom_to_uint8(ds)

    if data_u8.ndim == 2:
        img_rgb = cv2.cvtColor(data_u8, cv2.COLOR_GRAY2RGB)
    else:
        if data_u8.shape[-1] == 3:
            img_rgb = data_u8
        else:
            img_rgb = cv2.cvtColor(data_u8[..., 0], cv2.COLOR_GRAY2RGB)

    return img_rgb


def load_dicom(path):
    """
    Supports loading both regular and compressed DICOMs when decoder plugins are available.
    If decoding fails, caller should handle exception (Dataset returns a black image).
    """
    img_rgb = _load_dicom_rgb_cached(path)
    return img_rgb, None


try:
    if False:
        for fname in glob.glob(f"{RSNA_2022_PATH}/train_images/10006/*.dcm"):
            im, meta = load_dicom(fname)
            plt.figure()
            plt.imshow(im)
            plt.axis("off")
            break
except Exception as e:
    print("DICOM preview failed (non-fatal):", repr(e))




## === cell 7
class EffnetDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None, image_size=512):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.path = path
        self.transforms = transforms
        self.image_size = image_size

        pids = self.df["patient_id"].astype(str).to_numpy()
        iids = self.df["image_id"].astype(str).to_numpy()
        base = self.path
        self._dcm_paths = [
            os.path.join(base, pid, f"{iid}.dcm") for pid, iid in zip(pids, iids)
        ]

    def __getitem__(self, i):
        dcm_path = self._dcm_paths[i]

        try:
            img = load_dicom(dcm_path)[0]
        except Exception:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = torch.from_numpy(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.df)


ds_test = EffnetDataSet(df_test, TEST_IMAGES_PATH, WEIGHTS.transforms())
X = ds_test[2]
X.shape



## === cell 8
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()




## === cell 9
class EffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.efficientnet_v2_s(weights=WEIGHTS)
        self.model = create_feature_extractor(effnet, ["flatten"])
        self.nn_cancer = torch.nn.Sequential(
            torch.nn.Linear(1280, 1),
        )

    def forward(self, x):
        x = self.model(x)["flatten"]
        return self.nn_cancer(x)

    def predict(self, x):
        canc = self.forward(x)
        return torch.sigmoid(canc)


model = EffnetModel()
_ = model.predict(torch.randn(1, 3, 512, 512))
del model




## === cell 10
def load_model(model, name, path="."):
    """
    Original code expects external .tph checkpoints. In this environment/path, they may not exist.
    We keep the same architecture and will only load if the file is present; otherwise we fall back
    to ImageNet weights already in WEIGHTS (constructed in EffnetModel()).
    """
    ckpt_path = os.path.join(path, f"{name}.tph")
    if os.path.exists(ckpt_path):
        data = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(data)
    else:
        pass
    return model




## === cell 11
effnet_models = [
    load_model(EffnetModel(), name, EFFNET_CHECKPOINTS_PATH).to(DEVICE)
    for name in MODEL_NAMES
]
len(effnet_models)



## === cell 12
for m in effnet_models:
    m.eval()



## === cell 13
from typing import List


def predict_effnet(models: List[EffnetModel], ds, max_batches=1e9):
    cpu_cnt = os.cpu_count() or 2
    if DEVICE == "cuda":
        num_workers = min(8, cpu_cnt)
        prefetch_factor = 4
        persistent_workers = True
    else:
        num_workers = 0 if IS_KAGGLE else min(2, cpu_cnt)
        prefetch_factor = None
        persistent_workers = False

    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
    )
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        dl_kwargs["prefetch_factor"] = prefetch_factor

    dl_test = torch.utils.data.DataLoader(ds, **dl_kwargs)

    for m in models:
        m.eval()

    use_shared_backbone = True
    base_backbone = models[0].model

    if use_shared_backbone:
        Ws = []
        bs = []
        for m in models:
            lin = m.nn_cancer[0]
            Ws.append(lin.weight.detach().squeeze(0))  # [1280]
            bs.append(lin.bias.detach().squeeze(0))  # []
        W = torch.stack(Ws, dim=0).to(DEVICE)  # [M,1280]
        b = torch.stack(bs, dim=0).to(DEVICE)  # [M]

    with torch.no_grad():
        predictions = []
        for idx, X in enumerate(tqdm(dl_test, miniters=10)):
            if idx >= int(max_batches):
                break

            X = X.to(DEVICE, non_blocking=True)

            if use_shared_backbone:
                try:
                    feats = base_backbone(X)["flatten"]  # [B,1280]
                    logits = feats @ W.T  # [B,M]
                    logits = logits + b.unsqueeze(0)  # [B,M]
                    pred = torch.sigmoid(logits).mean(dim=1)  # [B]
                except Exception:
                    pred = torch.zeros((X.shape[0],), device=DEVICE)
                    for m in models:
                        y1 = m.predict(X).squeeze(-1)
                        pred += y1 / len(models)
            else:
                pred = torch.zeros((X.shape[0],), device=DEVICE)
                for m in models:
                    y1 = m.predict(X).squeeze(-1)
                    pred += y1 / len(models)

            predictions.append(pred)

        return torch.cat(predictions).float().cpu().numpy()


predict_effnet([EffnetModel().to(DEVICE)], ds_test, max_batches=2).shape



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/748747425.py in predict_effnet(models, ds, max_batches)
     55                     feats = base_backbone(X)["flatten"]  # [B,1280]
---> 56                     logits = feats @ W.T  # [B,M]
     57                     logits = logits + b.unsqueeze(0)  # [B,M]

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

During handling of the above exception, another exception occurred:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/748747425.py in <cell line: 0>()
     73 
     74 
---> 75 predict_effnet([EffnetModel().to(DEVICE)], ds_test, max_batches=2).shape
     76 

/tmp/ipykernel_55/748747425.py in predict_effnet(models, ds, max_batches)
     60                     pred = torch.zeros((X.shape[0],), device=DEVICE)
     61                     for m in models:
---> 62                         y1 = m.predict(X).squeeze(-1)
     63                         pred += y1 / len(models)
     64             else:

/tmp/ipykernel_55/897987069.py in predict(self, x)
     13 
     14     def predict(self, x):
---> 15         canc = self.forward(x)
     16         return torch.sigmoid(canc)
     17 

/tmp/ipykernel_55/897987069.py in forward(self, x)
     10     def forward(self, x):
     11         x = self.model(x)["flatten"]
---> 12         return self.nn_cancer(x)
     13 
     14     def predict(self, x):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

## === cell 14
effnet_pred = predict_effnet(effnet_models, ds_test)

df_effnet_pred = pd.DataFrame(data=effnet_pred, columns=["cancer"])
df_effnet_pred.head()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/748747425.py in predict_effnet(models, ds, max_batches)
     55                     feats = base_backbone(X)["flatten"]  # [B,1280]
---> 56                     logits = feats @ W.T  # [B,M]
     57                     logits = logits + b.unsqueeze(0)  # [B,M]

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

During handling of the above exception, another exception occurred:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2310572379.py in <cell line: 0>()
----> 1 effnet_pred = predict_effnet(effnet_models, ds_test)
      2 
      3 df_effnet_pred = pd.DataFrame(data=effnet_pred, columns=["cancer"])
      4 df_effnet_pred.head()
      5 

/tmp/ipykernel_55/748747425.py in predict_effnet(models, ds, max_batches)
     60                     pred = torch.zeros((X.shape[0],), device=DEVICE)
     61                     for m in models:
---> 62                         y1 = m.predict(X).squeeze(-1)
     63                         pred += y1 / len(models)
     64             else:

/tmp/ipykernel_55/897987069.py in predict(self, x)
     13 
     14     def predict(self, x):
---> 15         canc = self.forward(x)
     16         return torch.sigmoid(canc)
     17 

/tmp/ipykernel_55/897987069.py in forward(self, x)
     10     def forward(self, x):
     11         x = self.model(x)["flatten"]
---> 12         return self.nn_cancer(x)
     13 
     14     def predict(self, x):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

## === cell 15
df_effnet_pred.describe()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/578161262.py in <cell line: 0>()
----> 1 df_effnet_pred.describe()
      2 

NameError: name 'df_effnet_pred' is not defined

## === cell 16
df_test_pred = pd.concat(
    [df_test[["prediction_id", "patient_id", "image_id"]], df_effnet_pred], axis=1
)
df_test_pred = df_test_pred.sort_values(["patient_id", "image_id"])
df_test_pred = df_test_pred.groupby("prediction_id", as_index=False)["cancer"].mean()

df_test_pred = df_sample[["prediction_id"]].merge(
    df_test_pred, on="prediction_id", how="left"
)
df_test_pred["cancer"] = df_test_pred["cancer"].fillna(0.0).clip(0.0, 1.0)
df_test_pred.head()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/552712453.py in <cell line: 0>()
      1 df_test_pred = pd.concat(
----> 2     [df_test[["prediction_id", "patient_id", "image_id"]], df_effnet_pred], axis=1
      3 )
      4 df_test_pred = df_test_pred.sort_values(["patient_id", "image_id"])
      5 df_test_pred = df_test_pred.groupby("prediction_id", as_index=False)["cancer"].mean()

NameError: name 'df_effnet_pred' is not defined

## === cell 17
out_path = "submission.csv"
df_test_pred.to_csv(out_path, index=False)

sub = pd.read_csv(out_path)
assert list(sub.columns) == ["prediction_id", "cancer"]
assert len(sub) == len(df_sample)
print("Wrote", out_path, "with shape", sub.shape)
print(sub.head())

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1257527009.py in <cell line: 0>()
      1 out_path = "submission.csv"
----> 2 df_test_pred.to_csv(out_path, index=False)
      3 
      4 sub = pd.read_csv(out_path)
      5 assert list(sub.columns) == ["prediction_id", "cancer"]

NameError: name 'df_test_pred' is not defined
