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

0.9405835609248696

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by replacing TIFF tile reading with `tifffile.memmap`, while keeping the same tiling/reduction/prediction logic. I also fix `np.float` deprecation checks and ensure `torch` is imported before it’s used so later cells don’t crash. I add a safe fallback for missing model weight files (produce a valid all-empty submission rather than erroring) so a `.csv` is always generated. Finally, I keep the same RLE encoding semantics and write `submission.csv` with the required `id,predicted` columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is coming from producing an effectively empty/invalid submission because no model weights are found at the specified paths, so every `predicted` value becomes `""`. To move the score upward toward the target, the smallest valid change is to (1) point `DATA` and the model weight paths to the actual existing `/kaggle/input/...` locations, and (2) keep the same model/tiling/inference/RLE logic but ensure the “no-mask” case is encoded as an empty string (as expected by this competition) while non-empty masks are RLE-encoded. I’m also keeping strict submission alignment with `sample_submission.csv` ids and adding a tiny guard so missing TIFFs don’t crash the run (still yields a valid row). These changes preserve your core approach and should move the score from 0.0 to a non-zero value.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because no model weights are being loaded (your code falls back to predicting empty masks for every image), so the smallest improvement is to actually load the provided fold checkpoints if they exist in the Kaggle input tree. I keep your exact model/tiling/inference/RLE logic, but add a tiny, safe search for the `.pth` files under `/kaggle/input` when the hardcoded path doesn’t exist, and then load all found weights (still the same architecture). I also ensure the output submission is aligned to `sample_submission.csv` ids (as you already do) and keep the empty-mask encoding as `""` (competition-expected). These minimal path/weights fixes should move the score from 0.0 upward toward your target without changing evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from effectively predicting empty masks for all images because the checkpoint paths are very likely wrong for this environment (so no weights are loaded and the code falls back to `predicted=""`). I keep your exact model/tiling/inference/RLE logic, but make the weight discovery more reliable by (1) expanding the search to any `.pth` under `/kaggle/input` and (2) preferring checkpoints that match your expected naming/fold pattern, so real weights get loaded and predictions become non-empty. I also keep submission alignment to `sample_submission.csv` ids and add a tiny safeguard to avoid double-counting overlapping tiles by switching the tile aggregation from `+=` to `|=` (still the same thresholded binary semantics, but prevents overlap from inflating values). These changes are minimal, preserve evaluation semantics, and should move the score upward toward the target by producing actual segmentations instead of empty masks.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still coming from effectively submitting empty masks (either because no weights are actually found/loaded, or because the model outputs are being tiled/stitched onto the wrong pixel coordinates for this competition’s RLE convention). I keep your exact model, tiling, thresholding, and RLE logic, but make two minimal score-critical fixes: (1) reliably load checkpoints by searching for `.pth` and also `.pt` under `/kaggle/input` (some datasets use `.pt`), and (2) fix the axis/order mismatch between dataset shape, grid creation, and mask indexing so tiles land in the correct places before RLE. These changes preserve the same inference semantics (same model forward + sigmoid + TH) but prevent “all-empty / wrong-position” submissions that score ~0.0. The script still always write a valid `submission.csv` with `id,predicted`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with submitting empty masks for every test image because no real checkpoints are being loaded and/or the model package import path is wrong, so the inference loop never produces positive pixels. I make the smallest score-relevant changes to (1) load `segmentation_models_pytorch` reliably (it’s available as an installed package here, so don’t depend on a Kaggle Dataset path), and (2) make checkpoint discovery stricter by only accepting state_dicts that actually match your Unet so we don’t silently “load” incompatible weights and then effectively predict garbage/empties. Everything else (tiling, preprocessing, model, sigmoid+TH, stitching with `|=`, and RLE encoding) stays the same to preserve your core logic and evaluation semantics. This should move the score upward from 0.0 toward your target by producing non-empty, valid masks when compatible weights exist in `/kaggle/input`.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import warnings

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

try:
    import segmentation_models_pytorch as smp  # installed package

    _HAS_SMP = True
except Exception:
    smp = None
    _HAS_SMP = False

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



## === cell 1
sz = 256  # the size of tiles (after reduction)
reduce = 4  # reduce the original images by 4 times
TH = 0.5  # threshold for positive predictions

DATA = "/kaggle/input/hubmap-kidney-segmentation/test/"

MODELS = [
    f"/kaggle/input/skfoldalldata/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(1)
]

df_sample = pd.read_csv(
    "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
)
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"  # efficientnet-b4, se_resnext50_32x4d
minoverlap = 300
TTA = False




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if enc is None or (isinstance(enc, (float, np.floating)) and np.isnan(enc)):
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
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

s_th = 40  # saturation blancking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape_hw, window=256, min_overlap=32):
    """
    Return array of size (N,4): x1,x2,y1,y2 for an image of shape (H,W).
    """
    h, w = shape_hw
    step = window - min_overlap

    nx = h // step + 1
    x1 = np.linspace(0, h, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = max(0, h - window)
    x2 = (x1 + window).clip(0, h)

    ny = w // step + 1
    y1 = np.linspace(0, w, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = max(0, w - window)
    y2 = (y1 + window).clip(0, w)

    slices = np.zeros((nx, ny, 4), dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


class HuBMAPDataset(Dataset):
    """
    Keep image shape as (H,W) consistently so tile coords/mask indexing match RLE convention.
    """

    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path = os.path.join(DATA, idx + ".tiff")

        if not os.path.exists(self.path):
            self.data = None
            self.reduce = reduce
            self.sz = reduce * sz
            self.layout = "HWC"
            self.shape = (self.sz, self.sz)  # (H,W)
            self.mask_grid = np.zeros((0, 4), dtype=np.int64)
            return

        self.data = tiff.memmap(self.path)  # (H,W,3) or (3,H,W)
        self.reduce = reduce
        self.sz = reduce * sz  # tile size on original resolution
        self._infer_shape_and_layout()
        self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=minoverlap)

    def _infer_shape_and_layout(self):
        arr = self.data
        if arr.ndim == 3:
            if arr.shape[-1] == 3:
                self.layout = "HWC"
                self.shape = (int(arr.shape[0]), int(arr.shape[1]))  # (H,W)
            elif arr.shape[0] == 3:
                self.layout = "CHW"
                self.shape = (int(arr.shape[1]), int(arr.shape[2]))  # (H,W)
            else:
                self.layout = "HWC"
                self.shape = (int(arr.shape[0]), int(arr.shape[1]))
        else:
            raise ValueError(f"Unexpected tiff shape for {self.path}: {arr.shape}")

    def __len__(self):
        return len(self.mask_grid)

    def _read_window(self, x1, x2, y1, y2):
        if self.layout == "HWC":
            img = np.asarray(self.data[x1:x2, y1:y2, :])
        else:  # CHW
            img = np.asarray(self.data[:, x1:x2, y1:y2])
            img = np.moveaxis(img, 0, -1)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        return img

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        img = self._read_window(x1, x2, y1, y2)

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // reduce, self.sz // reduce),
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)

        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img2tensor((img / 255.0 - mean) / std), vertices, -1
        else:
            return img2tensor((img / 255.0 - mean) / std), vertices, idx


class Model_pred:
    def __init__(self, models, dl, tta: bool = TTA, half: bool = False):
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
                    py /= len(self.models)

                    py = F.interpolate(
                        py, scale_factor=reduce, mode="bilinear", align_corners=False
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu()
                    py = py.squeeze(-1).numpy()
                    z = z.numpy()

                    for i in range(len(py)):
                        yield py[i], z[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)


def _find_weight_files_fallback():
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return []

    candidates = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if low.endswith(".pth") or low.endswith(".pt"):
                candidates.append(os.path.join(dirpath, fn))

    if not candidates:
        return []

    def _rank(p):
        low = os.path.basename(p).lower()
        score = 0
        if "efficientnet-b4" in low:
            score += 10
        if "unet" in low:
            score += 5
        if "bceloss" in low or "bce" in low:
            score += 2
        if "fold" in low:
            score += 1
        if "model" in low:
            score += 1
        return (-score, len(p), p)

    candidates = sorted(candidates, key=_rank)
    return candidates[:5]


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    return None


def _state_dict_compatible(model, state_dict):
    try:
        model_state = model.state_dict()
        matched = 0
        for k, v in state_dict.items():
            if k in model_state:
                matched += 1
                if tuple(v.shape) != tuple(model_state[k].shape):
                    return False
        return matched > 0
    except Exception:
        return False


models = []
_model_paths_existing = []

if _HAS_SMP:
    for path in MODELS:
        if os.path.exists(path):
            _model_paths_existing.append(path)
    if len(_model_paths_existing) == 0:
        _model_paths_existing = _find_weight_files_fallback()

if _HAS_SMP and len(_model_paths_existing) > 0:
    for path in _model_paths_existing:
        try:
            obj = torch.load(path, map_location=torch.device("cpu"))
        except Exception:
            continue

        state_dict = _extract_state_dict(obj)
        if state_dict is None:
            continue

        model = smp.Unet(model_name, encoder_weights=None, classes=1)

        if not _state_dict_compatible(model, state_dict):
            continue

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) == len(model.state_dict()):
            continue

        model.float()
        model.eval()
        model.to(device)
        models.append(model)

        del obj, state_dict
else:
    models = []

names, preds = [], []

for _, row in df_sample.iterrows():
    idx = row["id"]
    names.append(idx)

    if len(models) == 0:
        preds.append("")
        continue

    ds = HuBMAPDataset(idx)
    if len(ds) == 0:
        preds.append("")
        continue

    dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
    mp = Model_pred(models, dl)

    mask = np.zeros(ds.shape, dtype=np.uint8)
    for pred, vert, i in iter(mp):
        x1, x2, y1, y2 = vert
        tile_bin = pred > TH
        mask[x1:x2, y1:y2] |= tile_bin.astype(np.uint8)

    mask = (mask > 0).astype(np.uint8)
    preds.append("" if mask.sum() == 0 else rle_encode_less_memory(mask))

    del mask, ds, dl
    gc.collect()



## === cell 4
df = pd.DataFrame({"id": names, "predicted": preds})
df.to_csv("submission.csv", index=False)
print(df.head())
print("Loaded models:", len(models), "from candidates:", _model_paths_existing)
print("Wrote submission.csv with shape:", df.shape)
