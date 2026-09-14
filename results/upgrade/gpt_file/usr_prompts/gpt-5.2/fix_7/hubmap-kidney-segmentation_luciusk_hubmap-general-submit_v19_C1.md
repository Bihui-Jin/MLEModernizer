# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect functional tissue units (FTUs) across different tissue preparation pipelines. An FTU is defined as a "three-dimensional block of cells centered around a capillary, such that each cell in this block is within diffusion distance from any other cell in the same block".

## Metric
Dice coefficient.

## Submission Format
Use run-length encoding on the pixel values. Submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The file should contain a header and have the following format:

```
img,pixels\
1,1 1 5 1\
2,1 1\
3,1 1\
etc.
```

## Dataset
The training set includes annotations in both RLE-encoded and unencoded (JSON) forms. The annotations denote segmentations of glomeruli.

Both the training and public test sets also include anatomical structure segmentations. They are intended to help you identify the various parts of the tissue.

### File structure
The JSON files are structured as follows, with each feature having:

-   A `type` (`Feature`) and object type `id` (`PathAnnotationObject`). Note that these fields are the same between all files and do not offer signal.
-   A `geometry` containing a `Polygon` with `coordinates` for the feature's enclosing volume
-   Additional `properties`, including the name and color of the feature in the image.
-   The `IsLocked` field is the same across file types (locked for glomerulus, unlocked for anatomical structure) and is not signal-bearing.

Note that the objects themselves do NOT have unique IDs. The expected prediction for a given image is an RLE-encoded mask containing ALL objects in the image. The mask, as mentioned in the Evaluation page, should be binary when encoded - with `0` indicating the lack of a masked pixel, and `1` indicating a masked pixel.

`train.csv` contains the unique IDs for each image, as well as an RLE-encoded representation of the mask for the objects in the image.

`HuBMAP-20-dataset_information.csv` contains additional information (including anonymized patient data) about each image.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
            HuBMAP-20-dataset_information.csv (16 lines)
            HuBMAP-20-dataset_information.csv.zip (987 Bytes)
            description.md (211 lines)
            sample_submission.csv (4 lines)
            sample_submission.csv.zip (238 Bytes)
            test.zip (3.3 GB)
            train.csv (13 lines)
            train.csv.zip (5.0 MB)
            train.zip (16.5 GB)
            hubmap-kidney-segmentation/
                HuBMAP-20-dataset_information.csv (16 lines)
                HuBMAP-20-dataset_information.csv.zip (987 Bytes)
                ... and 7 other files
                hubmap-kidney-segmentation/
                test/
                    0486052bb-anatomical-structure.json (161 lines)
                    0486052bb.json (11619 lines)
                    ... and 8 other files
                    test/
                train/
                    1e2425f28-anatomical-structure.json (127 lines)
                    1e2425f28.json (30956 lines)
                    ... and 34 other files
                    train/
            test/
                0486052bb-anatomical-structure.json (161 lines)
                0486052bb.json (11619 lines)
                ... and 8 other files
                test/
            train/
                1e2425f28-anatomical-structure.json (127 lines)
                1e2425f28.json (30956 lines)
                ... and 34 other files
                train/
        input/
            HuBMAP-20-dataset_information.csv (16 lines)
            HuBMAP-20-dataset_information.csv.zip (987 Bytes)
            description.md (211 lines)
            sample_submission.csv (4 lines)
            sample_submission.csv.zip (238 Bytes)
            test.zip (3.3 GB)
            train.csv (13 lines)
            train.csv.zip (5.0 MB)
            train.zip (16.5 GB)
            hubmap-kidney-segmentation/
                HuBMAP-20-dataset_information.csv (16 lines)
                HuBMAP-20-dataset_information.csv.zip (987 Bytes)
                ... and 7 other files
                hubmap-kidney-segmentation/
                test/
                    0486052bb-anatomical-structure.json (161 lines)
                    0486052bb.json (11619 lines)
                    ... and 8 other files
                    test/
                train/
                    1e2425f28-anatomical-structure.json (127 lines)
                    1e2425f28.json (30956 lines)
                    ... and 34 other files
                    train/
            test/
                0486052bb-anatomical-structure.json (161 lines)
                0486052bb.json (11619 lines)
                ... and 8 other files
                test/
                    0486052bb-anatomical-structure.json (161 lines)
                    0486052bb.json (11619 lines)
                    ... and 8 other files
                    test/
            train/
                1e2425f28-anatomical-structure.json (127 lines)
                1e2425f28.json (30956 lines)
                ... and 34 other files
                train/
                    1e2425f28-anatomical-structure.json (127 lines)
                    1e2425f28.json (30956 lines)
                    ... and 34 other files
                    train/
        working/
            hubmap-kidney-segmentation/
                HuBMAP-20-dataset_information.csv (16 lines)
                HuBMAP-20-dataset_information.csv.zip (987 Bytes)
                ... and 7 other files
                hubmap-kidney-segmentation/
                test/
                    0486052bb-anatomical-structure.json (161 lines)
                    0486052bb.json (11619 lines)
                    ... and 8 other files
                    test/
                train/
                    1e2425f28-anatomical-structure.json (127 lines)
                    1e2425f28.json (30956 lines)
                    ... and 34 other files
                    train/
```

-> data/HuBMAP-20-dataset_information.csv has 15 rows and 16 columns.
The columns are: image_file, width_pixels, height_pixels, anatomical_structures_segmention_file, glomerulus_segmentation_file, patient_number, race, ethnicity, sex, age, weight_kilograms, height_centimeters, bmi_kg/m^2, laterality, percent_cortex... and 1 more columns

-> data/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv has 15 rows and 16 columns.
The columns are: image_file, width_pixels, height_pixels, anatomical_structures_segmention_file, glomerulus_segmentation_file, patient_number, race, ethnicity, sex, age, weight_kilograms, height_centimeters, bmi_kg/m^2, laterality, percent_cortex... and 1 more columns

-> data/hubmap-kidney-segmentation/sample_submission.csv has 3 rows and 2 columns.
The columns are: id, predicted

-> data/hubmap-kidney-segmentation/test/0486052bb-anatomical-structure.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string"
      },
      "id": {
        "type": "string"
      },
      "geometry": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string"
          },
          "coordinates": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "integer"
                }
              }
            }
          }
        },
        "required": [
          "coordinates",
          "type"
        ]
      },
      "properties": {
        "type": "object",
        "properties": {
          "classification": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string"
              },
              "colorRGB": {
                "type": "integer"
              }
            },
            "required": [
              "colorRGB",
              "name"
            ]
          },
          "isLocked": {
            "type": "boolean"
          },
          "measurements": {
            "type": "array"
          }
        },
        "required": [
          "classification",
          "isLocked",
          "measurements"
        ]
      }
    },
    "required": [
      "geometry",
      "id",
      "properties",
      "type"
    ]
  }
}

-> data/hubmap-kidney-segmentation/test/0486052bb.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string"
      },
      "id": {
        "type": "string"
      },
      "geometry": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string"
          },
          "coordinates": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "number"
                }
              }
            }
          }
        },
        "required": [
          "coordinates",
          "type"
        ]
      },
      "properties": {
        "type": "object",
        "properties": {
          "classification": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string"
              },
              "colorRGB": {
                "type": "integer"
              }
            },
            "required": [
              "colorRGB",
              "name"
            ]
          },
          "isLocked": {
            "type": "boolean"
          },
          "measurements": {
            "type": "array"
          }
        },
        "required": [
          "classification",
          "isLocked",
          "measurements"
        ]
      }
    },
    "required": [
      "geometry",
      "id",
      "properties",
      "type"
    ]
  }
}

-> data/hubmap-kidney-segmentation/test/095bf7a1f-anatomical-structure.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string"
      },
      "id": {
        "type": "string"
      },
      "geometry": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string"
          },
          "coordinates": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "integer"
                }
              }
            }
          }
        },
        "required": [
          "coordinates",
          "type"
        ]
      },
      "properties": {
        "type": "object",
        "properties": {
          "classification": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string"
              },
              "colorRGB": {
                "type": "integer"
              }
            },
            "required": [
              "colorRGB",
              "name"
            ]
          },
          "isLocked": {
            "type": "boolean"
          },
          "measurements": {
            "type": "array"
          }
        },
        "required": [
          "classification",
          "isLocked",
          "measurements"
        ]
      }
    },
    "required": [
      "geometry",
      "id",
      "properties",
      "type"
    ]
  }
}

-> data/hubmap-kidney-segmentation/test/095bf7a1f.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string"
      },
      "id": {
        "type": "string"
      },
      "geometry": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string"
          },
          "coordinates": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "number"
                }
              }
            }
          }
        },
        "required": [
          "coordinates",
          "type"
        ]
      },
      "properties": {
        "type": "object",
        "properties": {
          "classification": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string"
              },
              "colorRGB": {
                "type": "integer"
              }
            },
            "required": [
              "colorRGB",
              "name"
            ]
          },
          "isLocked": {
            "type": "boolean"
          },
          "measurements": {
            "type": "array"
          }
        },
        "required": [
          "classification",
          "isLocked",
          "measurements"
        ]
      }
    },
    "required": [
      "geometry",
      "id",
      "properties",
      "type"
    ]
  }
}

-> data/hubmap-kidney-segmentation/test/8242609fa-anatomical-structure.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string"
      },
      "id": {
        "type": "string"
      },
      "geometry": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string"
          },
          "coordinates": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "integer"
                }
              }
            }
          }
        },
        "required": [
          "coordinates",
          "type"
        ]
      },
      "properties": {
        "type": "object",
        "properties": {
          "classification": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string"
              },
              "colorRGB": {
                "type": "integer"
              }
            },
            "required": [
              "colorRGB",
              "name"
            ]
          },
          "isLocked": {
            "type": "boolean"
          },
          "measurements": {
            "type": "array"
          }
        },
        "required": [
          "classification",
          "isLocked",
          "measurements"
        ]
      }
    },
    "required": [
      "geometry",
      "id",
      "properties",
      "type"
    ]
  }
}

-> (stopped after 10 files for performance)

# 5. Target score

0.9274917681347176

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by reading the TIFFs with `tifffile` and using NumPy slicing to extract windows, keeping the exact tiling/blank-tile filtering logic intact. I also fix runtime errors caused by deprecated `np.float`, missing imports due to the first-cell crash, and replace deprecated `F.upsample` with `F.interpolate` (same semantics) for stability. Finally, I ensure the submission is always created with exactly the sample submission’s IDs (3 rows) and correct `id,predicted` columns, even if an image read fails, so Kaggle accepts the CSV.'
- What this solution (achieved 0.03226) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not installed) by adding a lightweight fallback `Unet` implementation that keeps the same call pattern (`HuBMAP().forward -> logits`), so inference can run end-to-end. I also fix the missing model-weight paths by conditionally loading weights only when they exist; if none exist, the code still generate a valid `submission.csv` (but this likely not meet the target score because it’s effectively untrained). Finally, I add small safety fixes for device/half usage and make sure the submission uses exactly the sample submission IDs and required columns.'
- What this solution (achieved 0.03226) has done: 'Your score is low because the code is effectively running with random/untrained weights (the referenced checkpoint paths don’t exist in this environment), so the generated masks are mostly noise. To move toward the target with minimal semantic change, I keep the same tiling/inference/RLE pipeline but (1) load the available pretrained model weights shipped with the competition dataset (the common `resnet34` checkpoint), (2) build the exact same UNet architecture those weights expect (no fallback unless the file is missing), and (3) use mild test-time augmentation already supported by your wrapper (horizontal/vertical flips) to lift Dice without changing the core approach. These changes should substantially increase the score and stay within Kaggle constraints while still producing a valid `submission.csv`.'
- What this solution (achieved 0.03226) has done: 'Your current score is low because the intended pretrained checkpoint path doesn’t exist, so the model runs with random weights; the smallest change that moves score toward your target is to (1) automatically discover and load any `.pth` checkpoints shipped under the dataset’s `ckpts/` folder, and (2) ensure the architecture matches those checkpoints by requiring `segmentation_models_pytorch`’s ResNet34 UNet when a checkpoint is found (fallback only if no checkpoint exists). I’m also keeping your exact tiling, blank-tile filtering, upsampling, TTA logic, thresholding, and RLE pipeline unchanged, so evaluation semantics stay the same. Finally, I keep the submission IDs aligned to `sample_submission.csv` exactly as before so the CSV is always valid.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import warnings
import subprocess

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)


def _ensure_smp_available():
    try:
        import segmentation_models_pytorch as smp  # noqa: F401

        return True
    except Exception:
        pass

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "segmentation-models-pytorch",
                "pretrainedmodels",
            ]
        )
        import segmentation_models_pytorch as smp  # noqa: F401

        return True
    except Exception:
        return False


HAS_SMP = _ensure_smp_available()
if HAS_SMP:
    import segmentation_models_pytorch as smp  # type: ignore
else:
    smp = None



## === cell 1
sz = 256  # the size of tiles (at reduced resolution)
reduce = 4  # reduce the original images by 4 times
TH = 0.35  # threshold for positive predictions

DATA_CANDIDATES = [
    "../input/hubmap-kidney-segmentation",  # expected location for .tiff files
    "../input/hubmap-kidney-segmentation/test",  # sometimes contains .tiff in other mirrors
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test",
    "../input",
]

CKPT_DIRS = [
    "../input/hubmap-kidney-segmentation/ckpts",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/ckpts",
    "../input/ckpts",
]

df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, float) and np.isnan(enc):
            continue
        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start : start + length] = 1 + m
    return img.reshape(shape).T


def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(" ".join(str(x) for x in runs))
    return encs


def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

s_th = 40  # saturation blanking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def _find_tiff_path(idx: str):
    fn = idx + ".tiff"
    for base in DATA_CANDIDATES:
        fp = os.path.join(base, fn)
        if os.path.exists(fp):
            return fp
        fp = os.path.join(base, "test", fn)
        if os.path.exists(fp):
            return fp
        fp = os.path.join(base, "hubmap-kidney-segmentation", fn)
        if os.path.exists(fp):
            return fp
        fp = os.path.join(base, "hubmap-kidney-segmentation", "test", fn)
        if os.path.exists(fp):
            return fp
    return None


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.reduce = reduce
        self.sz = reduce * sz  # tile size at full resolution

        fp = _find_tiff_path(idx)
        if fp is None:
            raise FileNotFoundError(
                f"Could not find TIFF for id={idx} in {DATA_CANDIDATES}"
            )

        img = tiff.imread(fp)

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            if img.shape[0] in (3, 4) and img.shape[1] > 32 and img.shape[2] > 32:
                img = np.moveaxis(img[:3], 0, -1)
            if img.shape[-1] > 3:
                img = img[..., :3]
        else:
            raise ValueError(f"Unexpected tiff shape for {idx}: {img.shape}")

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8, copy=False)

        self.full_img = img
        self.shape = self.full_img.shape[:2]  # (H,W)

        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = self.full_img[
            p00:p01, p10:p11, :
        ]

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // reduce, self.sz // reduce),
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img2tensor((img / 255.0 - mean) / std), -1
        else:
            return img2tensor((img / 255.0 - mean) / std), idx




## === cell 4
class Model_pred:
    def __init__(self, models, dl, tta: bool = False, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.no_grad():
            for x, y in iter(self.dl):
                if (y >= 0).sum() > 0:  # exclude empty tiles
                    x = x[y >= 0].to(device)
                    y = y[y >= 0]
                    if self.half:
                        x = x.half()

                    total = None
                    denom = 0

                    for model in self.models:
                        p = torch.sigmoid(model(x)).detach()
                        total = p if total is None else (total + p)
                        denom += 1

                    if self.tta:
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            for model in self.models:
                                p = model(xf)
                                p = torch.flip(p, f)
                                p = torch.sigmoid(p).detach()
                                total += p
                                denom += 1

                    py = total / float(denom)

                    py = F.interpolate(
                        py, scale_factor=reduce, mode="bilinear", align_corners=False
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    for i in range(len(py)):
                        yield py[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 5
class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffX != 0 or diffY != 0:
            x1 = F.pad(
                x1,
                [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class _UNetFallback(nn.Module):
    def __init__(self, in_channels=3, classes=1, base=32):
        super().__init__()
        self.inc = _DoubleConv(in_channels, base)
        self.down1 = _Down(base, base * 2)
        self.down2 = _Down(base * 2, base * 4)
        self.down3 = _Down(base * 4, base * 8)
        self.down4 = _Down(base * 8, base * 16)

        self.up1 = _Up(base * 16 + base * 8, base * 8)
        self.up2 = _Up(base * 8 + base * 4, base * 4)
        self.up3 = _Up(base * 4 + base * 2, base * 2)
        self.up4 = _Up(base * 2 + base, base)

        self.outc = nn.Conv2d(base, classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class HuBMAP(nn.Module):
    def __init__(self, force_smp: bool = False):
        super(HuBMAP, self).__init__()
        if force_smp and (not HAS_SMP):
            raise RuntimeError(
                "segmentation_models_pytorch is required to load the discovered checkpoints, but it's not available."
            )

        if HAS_SMP:
            self.cnn_model = smp.Unet("resnet34", encoder_weights=None, classes=1)
        else:
            self.cnn_model = _UNetFallback(in_channels=3, classes=1, base=32)

    def forward(self, imgs):
        return self.cnn_model(imgs)




## === cell 6
def _discover_ckpts(ckpt_dirs):
    paths = []
    for ckpt_dir in ckpt_dirs:
        if not os.path.isdir(ckpt_dir):
            continue
        for root, _, files in os.walk(ckpt_dir):
            for fn in files:
                if fn.lower().endswith(".pth"):
                    paths.append(os.path.join(root, fn))
    return sorted(list(dict.fromkeys(paths)))


MODELS = _discover_ckpts(CKPT_DIRS)

models = []
existing_paths = [p for p in MODELS if os.path.exists(p)]

if len(existing_paths) > 0 and (not HAS_SMP):
    raise RuntimeError(
        f"Found {len(existing_paths)} checkpoint(s) but segmentation_models_pytorch is unavailable; cannot load."
    )

if len(existing_paths) == 0:
    model = HuBMAP(force_smp=False).to(device)
    model.float().eval()
    models = [model]
else:
    for path in existing_paths:
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model = HuBMAP(force_smp=True)
        model.load_state_dict(state_dict, strict=True)
        model.float()
        model.eval()
        model.to(device)
        models.append(model)
        del state_dict
        gc.collect()

gc.collect()



## === cell 7
names, preds = [], []

use_tta = True

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]
    try:
        ds = HuBMAPDataset(idx)
        dl = DataLoader(
            ds,
            batch_size=bs,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            num_workers=0,
        )
        mp = Model_pred(models, dl, tta=use_tta)

        mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
        for p, i in iter(mp):
            mask[i.item()] = p.squeeze(-1) > TH

        mask = (
            mask.view(ds.n0max, ds.n1max, ds.sz, ds.sz)
            .permute(0, 2, 1, 3)
            .reshape(ds.n0max * ds.sz, ds.n1max * ds.sz)
        )
        mask = mask[
            ds.pad0
            // 2 : -(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz,
            ds.pad1
            // 2 : -(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz,
        ]

        rle = rle_encode_less_memory(mask.numpy().astype(np.uint8))
    except Exception:
        rle = ""

    names.append(idx)
    preds.append(rle)

    if "ds" in locals():
        del ds
    if "dl" in locals():
        del dl
    if "mask" in locals():
        del mask
    gc.collect()



## === cell 8
sub = pd.DataFrame({"id": names, "predicted": preds})

sub = df_sample[["id"]].merge(sub, on="id", how="left")
sub["predicted"] = sub["predicted"].fillna("")

sub.to_csv("submission.csv", index=False)

print(sub.head())
print(f"Saved submission.csv with {len(sub)} rows")
print(f"segmentation_models_pytorch available: {HAS_SMP}")
print(f"Discovered checkpoints in {CKPT_DIRS}: {len(MODELS)}")
print(f"Loaded model checkpoints: {len(existing_paths)} / {len(MODELS)}")
print(f"Using TTA: {use_tta}")
