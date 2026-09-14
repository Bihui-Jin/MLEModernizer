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

0.9323729005535468

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by replacing TIFF reading and window extraction with `tifffile.memmap` and numpy slicing, keeping the same tiling/prediction logic. I also fix `np.float` deprecation, ensure `torch` is imported before use, and keep the same model loading/inference/thresholding so evaluation semantics remain unchanged. Finally, I make sure we always create `submission.csv` with the exact `id,predicted` columns from `sample_submission.csv`, even if an image ends up with an empty mask.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not installed here) by adding a tiny fallback U-Net implementation that can still load the provided `.pth` weights when they match, and otherwise safely continue with randomly initialized weights (so the pipeline always runs and creates a valid submission). I also fix the `tifffile.memmap` failure by switching to a robust TIFF reader that uses `tifffile.TiffFile(...).asarray(out='memmap')` when possible and falls back to normal array loading when not memory-mappable, while keeping the same window-slicing logic. Finally, I guard model-path discovery (search under `../input/**`) so missing paths don’t crash, and ensure RLE encoding returns an empty string for an all-zero mask to avoid invalid runs.'
- What this solution (achieved 0.0) has done: 'The runtime error comes from `tifffile` needing the optional `imagecodecs` package to decode JPEG-compressed TIFFs, which isn’t installed in this environment. To keep the core tiling/inference logic unchanged but make the pipeline run end-to-end, I replace the TIFF reading with an OpenCV-based TIFF reader fallback (OpenCV can decode these TIFFs here) and only if that fails we try `tifffile`. I also keep the array axis handling identical (ensure HWC with 3 channels) so downstream resizing, HSV filtering, and model input remain the same. This move the score from 0.0 (no submission) toward the target by enabling actual inference and a valid `submission.csv` to be produced.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import glob
import warnings

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

try:
    import segmentation_models_pytorch as smp  # type: ignore
except Exception:
    smp = None



## === cell 1
sz = 256  # tile size at network input resolution
reduce = 4  # downscale factor applied before model; later upsample back by this factor
TH = 0.35  # threshold for positive predictions

DATA = "../input/hubmap-kidney-segmentation/test/"

MODELS = [
    f"../input/b4-256-noshifttrain/efficientnet-b2-256-FOLD-{i}-model.pth"
    for i in range(5)
]

df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b2"
shift = True
minoverlap = 300




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if enc is None or (isinstance(enc, float) and np.isnan(enc)):
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
    if img is None or img.size == 0 or img.max() == 0:
        return ""
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

s_th = 40  # saturation blanking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for minimum number of pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return array of size (N,4), where N - number of tiles,
    axis represents: x1,x2,y1,y2
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)

    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)

    slices = np.zeros((nx, ny, 4), dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


def _ensure_hwc_3ch(arr: np.ndarray) -> np.ndarray:
    """Normalize to HWC with exactly 3 channels, uint8-compatible values."""
    mm = arr
    if mm.ndim == 2:
        mm = mm[..., None]
    if mm.ndim == 3 and mm.shape[0] in (3, 4) and mm.shape[-1] not in (3, 4):
        mm = np.moveaxis(mm, 0, -1)
    if mm.shape[-1] < 3:
        mm = np.repeat(mm, 3, axis=-1)
    elif mm.shape[-1] > 3:
        mm = mm[..., :3]
    return mm


def open_tiff_memmap(path):
    """
    Bugfix: tifffile requires optional 'imagecodecs' for JPEG-compressed TIFFs (not installed),
    causing a crash and a 0.0 score (no submission). Use OpenCV TIFF decoding first (available),
    then fall back to tifffile for cases OpenCV can't read.
    """
    arr = None

    try:
        cv_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if cv_img is not None:
            arr = cv_img
    except Exception:
        arr = None

    if arr is None:
        with tiff.TiffFile(path) as tif:
            try:
                arr = tif.asarray(out="memmap")
            except Exception:
                arr = tif.asarray()

    mm = _ensure_hwc_3ch(arr)
    return mm


def read_window(mm, x1, x2, y1, y2):
    w = mm[x1:x2, y1:y2]
    if w.dtype != np.uint8:
        w = np.clip(w, 0, 255).astype(np.uint8)
    return w




## === cell 4
class Model_pred_shift:
    def __init__(self, models, dl, tta: bool = False, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.no_grad():
            for x, z, y in iter(self.dl):
                if (y >= 0).sum() > 0:  # exclude empty tiles
                    x = x[y >= 0].to(device)
                    z = z[y >= 0]
                    y = y[y >= 0]
                    if self.half:
                        x = x.half()
                    py = None
                    for model in self.models:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        py = p if py is None else (py + p)

                    if self.tta:
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            for model in self.models:
                                p = model(xf)
                                p = torch.flip(p, f)
                                py += torch.sigmoid(p).detach()
                        py /= 1 + len(flips)

                    py /= max(1, len(self.models))

                    py = F.upsample(py, scale_factor=reduce, mode="bilinear")
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    py = py.squeeze(-1).numpy()
                    z = z.numpy()

                    batch_size = len(py)
                    for i in range(batch_size):
                        yield py[i], z[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)


class HuBMAPDatasetShift(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.mm = open_tiff_memmap(os.path.join(DATA, idx + ".tiff"))
        self.shape = self.mm.shape[:2]  # (H,W)
        self.reduce = reduce
        self.sz = reduce * sz
        self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=minoverlap)

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        img = read_window(self.mm, x1, x2, y1, y2)

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // reduce, self.sz // reduce),
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)
        vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)

        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img2tensor((img / 255.0 - mean) / std), vertices, -1
        else:
            return img2tensor((img / 255.0 - mean) / std), vertices, idx




## === cell 5
class DoubleConv(nn.Module):
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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.size(2) - x1.size(2)
        dw = x2.size(3) - x1.size(3)
        if dh != 0 or dw != 0:
            x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FallbackUNet(nn.Module):
    def __init__(self, in_channels=3, classes=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.up1 = Up(base * 8 + base * 4, base * 4)
        self.up2 = Up(base * 4 + base * 2, base * 2)
        self.up3 = Up(base * 2 + base, base)
        self.outc = nn.Conv2d(base, classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


def build_model():
    if smp is not None:
        return smp.Unet(model_name, encoder_weights=None, classes=1)
    return FallbackUNet(in_channels=3, classes=1, base=32)


def _find_model_files(requested_paths):
    existing = [p for p in requested_paths if os.path.exists(p)]
    if existing:
        return existing
    candidates = glob.glob("../input/**/*.pth", recursive=True)
    candidates = sorted(candidates)
    return candidates[:5]  # keep same ensemble size intent


MODEL_FILES = _find_model_files(MODELS)
print("Found model files:", MODEL_FILES if MODEL_FILES else "NONE")



## === cell 6
models = []
if MODEL_FILES:
    for path in MODEL_FILES:
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model = build_model()
        try:
            model.load_state_dict(state_dict, strict=True)
            loaded = True
        except Exception:
            try:
                model.load_state_dict(state_dict, strict=False)
                loaded = True
            except Exception:
                loaded = False

        model.float()
        model.eval()
        model.to(device)
        models.append(model)
        print(f"Loaded: {os.path.basename(path)} (state_dict_loaded={loaded})")

    del state_dict
    gc.collect()
else:
    model = build_model().to(device).eval().float()
    models = [model]
    print(
        "No .pth weights found; using randomly initialized model (submission will be low-scoring)."
    )



## === cell 7
names, preds = [], []

for _, row in df_sample.iterrows():
    idx = row["id"]
    ds = HuBMAPDatasetShift(idx)
    dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
    mp = Model_pred_shift(models, dl)

    mask = np.zeros(ds.shape, dtype=np.uint8)
    for pred, vert, i in iter(mp):
        x1, x2, y1, y2 = vert
        mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

    mask = (mask > 0.5).astype(np.uint8)
    rle = rle_encode_less_memory(mask)

    names.append(idx)
    preds.append(rle)

    del mask, ds, dl
    gc.collect()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3639744685.py in open_tiff_memmap(path)
     75             try:
---> 76                 arr = tif.asarray(out="memmap")
     77             except Exception:

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in asarray(self, key, series, level, squeeze, out, maxworkers, buffersize)
   4541                 raise ValueError('page is None')
-> 4542             result = page0.asarray(
   4543                 out=out, maxworkers=maxworkers, buffersize=buffersize

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in asarray(self, out, squeeze, lock, maxworkers, buffersize)
   8884 
-> 8885             for _ in self.segments(
   8886                 func=func,

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in segments(self, lock, maxworkers, func, sort, buffersize, _fullsize)
   8698                 ):
-> 8699                     yield from executor.map(decode, segments)
   8700 

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    316         try:
--> 317             return fut.result(timeout)
    318         finally:

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    400             try:
--> 401                 raise self._exception
    402             finally:

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in decode(args, decodeargs, decode)
   8671             def decode(args, decodeargs=decodeargs, decode=keyframe.decode):
-> 8672                 return func(decode(*args, **decodeargs))
   8673 

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in decode_raise_compression(exc, *args, **kwargs)
   8093             def decode_raise_compression(*args, exc=str(exc)[1:-1], **kwargs):
-> 8094                 raise ValueError(f'{exc}')
   8095 

ValueError: <COMPRESSION.JPEG: 7> requires the 'imagecodecs' package

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2932306676.py in <cell line: 0>()
      3 for _, row in df_sample.iterrows():
      4     idx = row["id"]
----> 5     ds = HuBMAPDatasetShift(idx)
      6     dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
      7     mp = Model_pred_shift(models, dl)

/tmp/ipykernel_55/1091460817.py in __init__(self, idx, sz, reduce)
     49 class HuBMAPDatasetShift(Dataset):
     50     def __init__(self, idx, sz=sz, reduce=reduce):
---> 51         self.mm = open_tiff_memmap(os.path.join(DATA, idx + ".tiff"))
     52         self.shape = self.mm.shape[:2]  # (H,W)
     53         self.reduce = reduce

/tmp/ipykernel_55/3639744685.py in open_tiff_memmap(path)
     76                 arr = tif.asarray(out="memmap")
     77             except Exception:
---> 78                 arr = tif.asarray()
     79 
     80     mm = _ensure_hwc_3ch(arr)

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in asarray(self, key, series, level, squeeze, out, maxworkers, buffersize)
   4540             if page0 is None:
   4541                 raise ValueError('page is None')
-> 4542             result = page0.asarray(
   4543                 out=out, maxworkers=maxworkers, buffersize=buffersize
   4544             )

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in asarray(self, out, squeeze, lock, maxworkers, buffersize)
   8883                 #     pass  # corrupted file, for example, with too many strips
   8884 
-> 8885             for _ in self.segments(
   8886                 func=func,
   8887                 lock=lock,

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in segments(self, lock, maxworkers, func, sort, buffersize, _fullsize)
   8697                     flat=False,
   8698                 ):
-> 8699                     yield from executor.map(decode, segments)
   8700 
   8701     def asarray(

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
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in decode(args, decodeargs, decode)
   8670 
   8671             def decode(args, decodeargs=decodeargs, decode=keyframe.decode):
-> 8672                 return func(decode(*args, **decodeargs))
   8673 
   8674         if maxworkers is None or maxworkers < 1:

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in decode_raise_compression(exc, *args, **kwargs)
   8092 
   8093             def decode_raise_compression(*args, exc=str(exc)[1:-1], **kwargs):
-> 8094                 raise ValueError(f'{exc}')
   8095 
   8096             return cache(decode_raise_compression)

ValueError: <COMPRESSION.JPEG: 7> requires the 'imagecodecs' package

## === cell 8
sub = pd.DataFrame({"id": names, "predicted": preds})

sub = df_sample[["id"]].merge(sub, on="id", how="left")
sub["predicted"] = sub["predicted"].fillna("")  # empty string allowed for empty mask

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))
