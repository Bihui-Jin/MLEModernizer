# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import sys
import gc
import json
import warnings

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

import numpy as np
import pandas as pd
import cv2

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
sz = 256  # tile size (model input at reduced resolution)
reduce = 4  # read tiles at full-res then downscale by 4

TH = 0.3

DATA = "../input/hubmap-kidney-segmentation/test/"
TRAIN_CSV = "../input/hubmap-kidney-segmentation/train.csv"

df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if enc is None:
            continue
        if isinstance(enc, (float, np.floating)) and np.isnan(enc):
            continue
        s = str(enc).split()
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
    if pixels.size == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

s_th = 40
p_th = 1000 * (sz // 256) ** 2


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def read_tiff_opencv(path: str) -> np.ndarray:
    """
    Uses OpenCV to read TIFFs (works here after raising OPENCV_IO_MAX_IMAGE_PIXELS).
    Returns HWC uint8 with 3 channels.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.ndim == 3:
        if img.shape[2] >= 3:
            img = img[:, :, :3]
        else:
            img = np.repeat(img, 3, axis=2)
    else:
        raise ValueError(f"Unexpected image ndim={img.ndim} for {path}")

    if img.dtype != np.uint8:
        mx = float(np.max(img)) if img.size else 0.0
        if mx > 255.0:
            img = (img.astype(np.float32) / (mx + 1e-9) * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def _poly_to_mask(poly_coords, h: int, w: int) -> np.ndarray:
    pts = np.asarray(poly_coords, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return np.zeros((h, w), dtype=np.uint8)
    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


_ANATOMICAL_MASK_CACHE = {}


def load_anatomical_prior_mask(idx: str, shape_hw) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    key = (idx, h, w)
    if key in _ANATOMICAL_MASK_CACHE:
        return _ANATOMICAL_MASK_CACHE[key]

    json_path = os.path.join(DATA, f"{idx}-anatomical-structure.json")
    if not os.path.exists(json_path):
        out = np.zeros((h, w), dtype=np.uint8)
        _ANATOMICAL_MASK_CACHE[key] = out
        return out

    try:
        with open(json_path, "r") as f:
            feats = json.load(f)
    except Exception:
        out = np.zeros((h, w), dtype=np.uint8)
        _ANATOMICAL_MASK_CACHE[key] = out
        return out

    mask = np.zeros((h, w), dtype=np.uint8)
    if isinstance(feats, list):
        for feat in feats:
            try:
                geom = feat.get("geometry", {})
                if geom.get("type", "") != "Polygon":
                    continue
                coords = geom.get("coordinates", None)
                if not coords or not isinstance(coords, list) or not coords[0]:
                    continue
                outer = coords[0]
                pts = np.asarray(outer, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
                    continue
                pts = np.round(pts).astype(np.int32)
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                cv2.fillPoly(mask, [pts], 1)
            except Exception:
                continue

    _ANATOMICAL_MASK_CACHE[key] = mask
    return mask


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path = os.path.join(DATA, idx + ".tiff")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Missing tiff: {self.path}")

        self.data = read_tiff_opencv(self.path)
        self.shape = self.data.shape[:2]  # (H,W)

        self.prior_full = load_anatomical_prior_mask(idx, self.shape)

        self.reduce = reduce
        self.sz = reduce * sz  # full-res tile size

        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        self._x0 = -self.pad0 // 2 + np.arange(self.n0max, dtype=np.int32) * self.sz
        self._y0 = -self.pad1 // 2 + np.arange(self.n1max, dtype=np.int32) * self.sz

    def __len__(self):
        return self.n0max * self.n1max

    def tile_origin(self, tile_id: int):
        n0, n1 = tile_id // self.n1max, tile_id % self.n1max
        return int(self._x0[n0]), int(self._y0[n1])

    def __getitem__(self, tile_id):
        n0, n1 = tile_id // self.n1max, tile_id % self.n1max
        x0 = int(self._x0[n0])
        y0 = int(self._y0[n1])

        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        patch = self.data[p00:p01, p10:p11]
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = patch

        prior = np.zeros((self.sz, self.sz), np.uint8)
        prior_patch = self.prior_full[p00:p01, p10:p11]
        prior[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = prior_patch

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )
            prior = cv2.resize(
                prior,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_NEAREST,
            )

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)

        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            ret_id = -1
        else:
            ret_id = int(tile_id)

        return (
            img2tensor((img / 255.0 - mean) / std),
            ret_id,
            img2tensor(prior[None, ...] / 1.0),
        )




## === cell 4
class Model_pred:
    def __init__(self, models, dl, tta: bool = False, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        out_hw = (int(self.dl.dataset.sz), int(self.dl.dataset.sz))

        with torch.no_grad():
            for batch in iter(self.dl):
                if len(batch) == 3:
                    x, y, prior = batch
                else:
                    x, y = batch
                    prior = None

                if (y >= 0).sum() > 0:
                    keep = y >= 0
                    x = x[keep].to(device, non_blocking=True)
                    y = y[keep]
                    if prior is not None:
                        prior = prior[keep].to(device, non_blocking=True)

                    if self.half:
                        x = x.half()
                        if prior is not None:
                            prior = prior.half()

                    py = None
                    for model in self.models:
                        p = model(x, prior=prior)
                        p = torch.sigmoid(p).detach()
                        py = p if py is None else (py + p)

                    if self.tta:
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            priorf = torch.flip(prior, f) if prior is not None else None
                            for model in self.models:
                                p = model(xf, prior=priorf)
                                p = torch.flip(p, f)
                                py += torch.sigmoid(p).detach()
                        py /= 1 + len(flips)

                    py /= len(self.models)

                    py = F.interpolate(
                        py, size=out_hw, mode="bilinear", align_corners=False
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    for i in range(len(py)):
                        yield py[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 5
class HuBMAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_img = nn.Parameter(torch.tensor(1.0))
        self.w_prior = nn.Parameter(torch.tensor(3.0))
        self.b = nn.Parameter(torch.tensor(-2.0))

    def forward(self, imgs, prior=None):
        x = imgs.float()
        g = x[:, 1:2] * float(std[1]) + float(mean[1])
        g = g.clamp(0.0, 1.0)

        blur = F.avg_pool2d(g, kernel_size=5, stride=1, padding=2)
        resp = (blur - g).abs()

        if prior is None:
            prior_f = torch.zeros_like(g)
        else:
            prior_f = prior.float().clamp(0.0, 1.0)

        logits = self.w_img * (0.5 * g + 0.5 * resp) + self.w_prior * prior_f + self.b
        return logits




## === cell 6
def rle_positive_fraction(rle: str, height: int, width: int) -> float:
    if rle is None or (isinstance(rle, (float, np.floating)) and np.isnan(rle)):
        return 0.0
    s = str(rle).split()
    if len(s) < 2:
        return 0.0
    total = 0
    for i in range(1, len(s), 2):
        total += int(s[i])
    return float(total) / float(height * width)


train_pos_frac = None
try:
    df_train = pd.read_csv(TRAIN_CSV)
    info_path = "../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
    if os.path.exists(info_path):
        df_info = pd.read_csv(info_path)
        info_map = df_info.set_index("image_file")[
            ["height_pixels", "width_pixels"]
        ].to_dict("index")
        fracs = []
        for _, r in df_train.iterrows():
            img_id = r["id"]
            key = f"{img_id}.tiff"
            if key in info_map:
                h = int(info_map[key]["height_pixels"])
                w = int(info_map[key]["width_pixels"])
                fracs.append(rle_positive_fraction(r["encoding"], h, w))
        if len(fracs) > 0:
            train_pos_frac = float(np.mean(fracs))
except Exception:
    train_pos_frac = None

if train_pos_frac is None or not np.isfinite(train_pos_frac) or train_pos_frac <= 0:
    train_pos_frac = 0.02

print(f"Using target positive fraction for calibration: {train_pos_frac:.6f}")

models = []
for _ in range(1):
    model = HuBMAP().to(device)
    model.eval()
    models.append(model)

gc.collect()




## === cell 7
def _to_int_index(x):
    if torch.is_tensor(x):
        return int(x.item())
    return int(x)


def _make_loader(ds: Dataset):
    nw = 0
    if os.name != "nt":
        nw = min(4, (os.cpu_count() or 2))
    return DataLoader(
        ds,
        batch_size=bs,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        num_workers=nw,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )


def _alloc_stitched_prob(ds: HuBMAPDataset) -> torch.Tensor:
    H_full = int(ds.n0max * ds.sz)
    W_full = int(ds.n1max * ds.sz)
    return torch.zeros((H_full, W_full), dtype=torch.float32)


def _crop_to_original(ds: HuBMAPDataset, prob_full: torch.Tensor) -> torch.Tensor:
    r0 = ds.pad0 // 2
    r1 = ds.pad0 - r0
    c0 = ds.pad1 // 2
    c1 = ds.pad1 - c0
    H_full, W_full = prob_full.shape
    return prob_full[
        r0 : (H_full - r1) if r1 > 0 else H_full,
        c0 : (W_full - c1) if c1 > 0 else W_full,
    ]


names, preds = [], []

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]
    ds = HuBMAPDataset(idx)
    dl = _make_loader(ds)
    mp = Model_pred(models, dl)

    prob_full = _alloc_stitched_prob(ds)

    for p, tile_id in iter(mp):
        tile_idx = _to_int_index(tile_id)
        if tile_idx < 0 or tile_idx >= len(ds):
            continue

        tile = p.squeeze(-1).clamp(0.0, 1.0).to(torch.float32)
        if tile.ndim == 3:
            tile = tile.squeeze()
        if tile.ndim != 2 or tile.shape[0] != ds.sz or tile.shape[1] != ds.sz:
            tile_t = tile
            if tile_t.ndim == 2:
                tile_t = tile_t[None, None, ...]
            elif tile_t.ndim == 3:
                tile_t = tile_t[None, ...]
            else:
                tile_t = tile.view(1, 1, -1, 1)
            tile_t = F.interpolate(
                tile_t, size=(ds.sz, ds.sz), mode="bilinear", align_corners=False
            )
            tile = tile_t[0, 0].contiguous()

        x0, y0 = ds.tile_origin(tile_idx)

        xs0 = max(0, x0)
        ys0 = max(0, y0)
        xs1 = min(x0 + ds.sz, ds.shape[0] + ds.pad0)  # padded extent
        ys1 = min(y0 + ds.sz, ds.shape[1] + ds.pad1)

        cx0 = x0 + ds.pad0 // 2
        cy0 = y0 + ds.pad1 // 2

        wx0 = max(0, cx0)
        wy0 = max(0, cy0)
        wx1 = min(cx0 + ds.sz, prob_full.shape[0])
        wy1 = min(cy0 + ds.sz, prob_full.shape[1])

        tx0 = wx0 - cx0
        ty0 = wy0 - cy0
        tx1 = tx0 + (wx1 - wx0)
        ty1 = ty0 + (wy1 - wy0)

        if wx1 > wx0 and wy1 > wy0:
            prob_full[wx0:wx1, wy0:wy1] = tile[tx0:tx1, ty0:ty1]

    prob = _crop_to_original(ds, prob_full)

    flat = prob.flatten()
    k = int(train_pos_frac * flat.numel())
    if k <= 0:
        thr = TH
    else:
        vals, _ = torch.topk(flat, k=min(k, flat.numel()), largest=True, sorted=True)
        thr = float(vals[-1].item()) if vals.numel() > 0 else TH

    mask = (prob > thr).to(torch.uint8)
    rle = rle_encode_less_memory(mask.numpy())

    names.append(idx)
    preds.append(rle)

    del mask, prob, prob_full, ds, dl, flat
    gc.collect()




## === cell 8
sub = df_sample.copy()
pred_map = dict(zip(names, preds))
sub["predicted"] = sub["id"].map(pred_map).fillna("")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print(f"Wrote submission.csv with {len(sub)} rows")
print(
    "Unique non-empty predictions:", (sub["predicted"].astype(str).str.len() > 0).sum()
)
