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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

0.8556657225019251

# 6. Current score

0.049

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.049) has done: 'I fix the TIFF reading failure by ensuring we never fall back to `tifffile` for JPEG-compressed TIFFs (which requires the unavailable `imagecodecs`), and by using a safe OpenCV-based fallback that reads embedded previews when present. I also ensure the script always produces a valid `submission.csv` with the exact required columns (`id`, `predicted`) even if a specific test image cannot be decoded, by emitting an empty mask RLE for that image instead of crashing. These changes are execution-blocking bug fixes and should be score-neutral relative to the intended pipeline. I keep the model/training logic unchanged.'
- What this solution (achieved 0.04885) has done: 'Your current 0.049 score is consistent with a prediction/metric mismatch: you train with `BCEWithLogitsLoss` but during validation/inference you threshold raw logits at `> 0` without applying `sigmoid`, which usually produces poorly calibrated masks and very low Dice. I keep the exact same model, data pipeline, training loop, and loss, but (1) apply `sigmoid` before thresholding in both `evaluate()` and `make_submission()`, and (2) use a standard 0.5 threshold on probabilities to align with binary-mask Dice evaluation. These are minimal semantic fixes (not architecture/training changes) and should move the score substantially toward your target by making the produced RLE masks reflect the learned probabilities correctly. The script still run end-to-end and write a valid `submission.csv` with `id,predicted`.'
- What this solution (achieved 0.0) has done: 'Your current gap to the target is large (0.04885 → 0.8557), so we need a small but high-impact fix that doesn’t change the model/training core: the biggest remaining issue is that inference is stitching *tile indices* back into the full image incorrectly. Right now `idx` is a 1D tile id, but you upsample each tile to `ds.reduce` and then write it into a buffer sized `ds.sz` (pre-reduce), causing silent shape/index mismatches and heavily corrupted full masks, which destroys Dice. I keep the same tiling, model, loss, and thresholding, but fix reconstruction by storing tiles at the *actual predicted tile size* and only assembling at the end, plus a tiny safety cast for `idx` to avoid tensor/int edge cases. This should legitimately move the score substantially toward the target without changing architecture or training semantics.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 strongly suggests the submission masks are effectively empty or badly mis-assembled at full resolution; the biggest low-risk fix is to reconstruct the full-size mask directly into the final canvas (shape `ds.shape`) using each tile’s `(n0,n1)` position and the dataset’s padding math, instead of reshaping a `(len(ds), tile, tile)` buffer (which silently breaks whenever some tiles are filtered out with `idx=-1`). I keep the same dataset, tiling, model, training, loss, and 0.5 sigmoid-thresholding, but change inference stitching to paste only valid predicted tiles into the correct region of the final mask, respecting padding and cropping. This should move Dice substantially upward toward your target by producing correctly aligned binary masks for RLE encoding. The submission file still be written as `submission.csv` with columns `id,predicted`.'
- What this solution (achieved 0.049) has done: 'I fix the submission formatting to exactly match the competition’s required columns (`id`, `predicted`) and ensure the CSV is ordered identically to `sample_submission.csv`, because a wrong schema/column naming can prevent scoring or yield an effectively invalid submission. I also make the RLE encoding and decoding consistently use the competition’s 1-indexed, column-major convention (Fortran order) without relying on fragile transposes, to avoid subtle off-by-orientation bugs that can crater Dice while keeping the same model and inference thresholding. Finally, I remove the duplicate `collate_fn` redefinition ambiguity by keeping one consistent version, so inference stitching never accidentally changes behavior across cells. These are minimal, execution-safe changes that preserve your model/training core logic and are directly tied to getting a valid, correctly evaluated submission.'
- What this solution (achieved 0.0) has done: 'Your current gap to the target is very large (0.049 → 0.8557), and the most likely remaining high-impact issue without changing the model/training core is a metric mismatch caused by background padding being treated as real negative labels. I keep the exact same U-Net, tiling, loss, and training loop, but add a per-tile “valid pixel” mask so padded pixels (outside the real image area) are excluded from loss and from validation Dice computation. This change aligns optimization and evaluation to only the true tissue/image area, which typically boosts Dice substantially for padded-tile pipelines. I also keep inference stitching the same, but ensure padding is naturally excluded because we only paste the real crop region.'
- What this solution (achieved 0.049) has done: 'Your current 0.0 score is most consistent with an evaluation-killing bug in the Dice computation (and likely your debugging signals): FP/FN are swapped in `evaluate()`, so your reported “Dice” isn’t actually Dice and can hide that the model is effectively predicting the wrong thing. I make the smallest semantic fix to compute TP/FP/FN correctly on the valid (non-padded) pixels and compute Dice from those counts, without changing the model, loss, training loop, tiling, or inference thresholding. This should move your real leaderboard Dice upward toward your target by ensuring you are validating the same thing Kaggle scores, letting you iterate from a meaningful baseline. I also keep the submission generation unchanged except for a tiny safety guard to ensure IDs remain strings and aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import gc
import json
import csv

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

DATA_ROOT = "/kaggle/input/hubmap-kidney-segmentation"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")

_train_df = None
_rle_map = None


def _load_train_rle_map():
    global _train_df, _rle_map
    if _rle_map is not None:
        return _rle_map
    if os.path.exists(TRAIN_CSV):
        _train_df = pd.read_csv(TRAIN_CSV)
        if "encoding" in _train_df.columns:
            enc_col = "encoding"
        elif "pixels" in _train_df.columns:
            enc_col = "pixels"
        else:
            enc_col = _train_df.columns[1]
        _rle_map = dict(
            zip(
                _train_df["id"].astype(str).tolist(),
                _train_df[enc_col].fillna("").astype(str).tolist(),
            )
        )
    else:
        _rle_map = {}
    return _rle_map


def rle_decode(mask_rle: str, shape):
    """
    Decode RLE string into a binary mask (H, W).
    Competition convention: 1-indexed, column-major (top-to-bottom, left-to-right),
    which corresponds to flattening in Fortran order.
    """
    h, w = shape
    if mask_rle is None:
        return np.zeros((h, w), dtype=np.uint8)
    s = str(mask_rle).strip()
    if s == "" or s.lower() == "nan":
        return np.zeros((h, w), dtype=np.uint8)
    nums = np.asarray(s.split(), dtype=np.int64)
    if nums.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    starts = nums[0::2] - 1
    lengths = nums[1::2]
    ends = starts + lengths
    img = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((h, w), order="F")


def _read_tiff_bgr_uint8(file_path: str) -> np.ndarray:
    """
    Bugfix: Avoid tifffile fallback which can require 'imagecodecs' for JPEG-compressed TIFFs.
    Strategy:
      1) Try cv2.imread (works for many TIFFs).
      2) Try skimage.io.imread (often works without imagecodecs in this environment).
      3) Try OpenCV's imdecode on raw bytes (can read embedded preview/strips for some TIFFs).
    If all fail, raise.
    """
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    arr = None
    try:
        from skimage import io as skio

        arr = skio.imread(file_path)
    except Exception:
        arr = None

    if arr is None:
        try:
            with open(file_path, "rb") as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            img2 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img2 is not None:
                return img2
        except Exception:
            pass

    if arr is None:
        raise RuntimeError(
            f"Failed to read TIFF (cv2 + skimage + cv2.imdecode failed). Path: {file_path}"
        )

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[0] >= 3 and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr[:3, ...], (1, 2, 0))
    else:
        raise RuntimeError(f"Unexpected TIFF shape {arr.shape} for file: {file_path}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            mx = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
            if mx <= 1.0 + 1e-6:
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        else:
            info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
            if info is not None and info.max > 0:
                arr = (
                    (arr.astype(np.float32) / float(info.max) * 255.0)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
            else:
                arr = arr.astype(np.uint8)

    arr = arr[..., ::-1].copy()
    return arr


class HuBMAPDatasetPreprocessing(Dataset):
    def __init__(self, image, category="train", use_compressed=False):
        self.folder = DATA_ROOT
        self.image = str(image)
        self.category = category
        self.use_compressed = use_compressed
        self.reduce = 4 if not use_compressed else 1  # keep original behavior
        self._image = None

        self._get_image()

        self.sz = 256 * self.reduce
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        if self.category == "train":
            rle_map = _load_train_rle_map()
            rle = rle_map.get(self.image, "")
            self.mask_full = rle_decode(rle, self.shape).astype(np.uint8)  # (H, W), 0/1
        else:
            self.mask_full = None

    def _get_image(self):
        if self._image is not None:
            return self._image
        file_path = os.path.join(self.folder, self.category, f"{self.image}.tiff")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing image file: {file_path}")

        img = _read_tiff_bgr_uint8(file_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {file_path}")
        self._image = img
        self.shape = (img.shape[0], img.shape[1])  # (H, W)
        return self._image

    def close(self):
        self._image = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return int(self.n0max * self.n1max)

    def __getitem__(self, idx):
        image = self._get_image()
        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz

        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = image[p00:p01, p10:p11]

        valid = np.zeros((self.sz, self.sz), np.uint8)
        valid[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = 1

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )
            valid = cv2.resize(
                valid,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.category == "train":
            mask = np.zeros((self.sz, self.sz, 3), np.uint8)
            mpatch = np.zeros((self.sz, self.sz), np.uint8)
            mpatch[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = self.mask_full[
                p00:p01, p10:p11
            ]
            if self.reduce != 1:
                mpatch = cv2.resize(
                    mpatch,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask[..., 0] = mpatch
            mask[..., 1] = valid
        else:
            mask = torch.zeros(1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)

        result_tensor = torch.from_numpy(
            ((img.astype(np.float32) / 255.0) - mean) / std
        ).float()
        result_tensor = result_tensor.permute(2, 0, 1).contiguous()

        if (s > 40).sum() <= 1000 or img.sum() <= 1000:
            return result_tensor, mask, -1
        else:
            return result_tensor, mask, idx




## === cell 1
"""
dataset = HuBMAPDatasetPreprocessing('0486052bb', use_compressed=True)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(2, 5)
c = 0
for i in range(100):
    sample, mask, idx = dataset[i]
    sample = sample.permute(1,2,0)
    if idx == -1 or (isinstance(mask, np.ndarray) and mask.sum() == 0):
        continue
    if c == 5:
        break
    plots[0][c].imshow((sample.numpy() * std + mean).clip(0,1))
    plots[0][c].set_title(f'Sample #{c}')
    plots[1][c].imshow(mask[...,0], cmap="gray")
    plots[1][c].set_title(f'Mask #{c}')
    c += 1
"""


## === cell 2
"""
# Visualize test patches if needed
import matplotlib.pyplot as plt
dataset_test = HuBMAPDatasetPreprocessing('0486052bb', category='test')
fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(1, 5)
c = 0
for i in range(100):
    sample, mask, idx = dataset_test[i]
    if idx == -1:
        continue
    if c == 5:
        break
    plots[c].imshow((sample.permute(1,2,0).numpy() * std + mean).clip(0,1))
    plots[c].set_title(f'Sample #{c}')
    c += 1
"""


## === cell 3
import torch
from torch import nn, optim
from torch.utils import data
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.skip_conf = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x2 = self.skip_conf(x2)
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x.clone())


class Model(nn.Module):
    """The base class for the encoder-decoder architecture."""

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, X):
        x1 = self.inc(X)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits




## === cell 4
import collections

OUTPUT_PATH = "/kaggle/working/models"


def collate_fn(x):
    x = list(filter(lambda z: z[-1] != -1, x))
    if not x:
        return []
    return data.dataloader.default_collate(x)


VALSET = ["e79de561c", "cb2d976f4"]


def get_tiff_images():
    input_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
    for images in glob.glob(input_folder):
        bname = os.path.basename(images)
        name = os.path.splitext(bname)[0]
        if name not in VALSET:
            dataset = HuBMAPDatasetPreprocessing(name, use_compressed=True)
            yield dataset


def get_val_set():
    for image in VALSET:
        dataset = HuBMAPDatasetPreprocessing(image, use_compressed=True)
        yield dataset


def dice_loss(pred, target, device):
    numerator = 2 * torch.sum(pred.float() * target.float())
    denominator = torch.sum(pred.float() + target.float())
    return numerator / denominator


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=True,
        device=None,
        alpha=0.7,
        beta=0.3,
        gamma=0.75,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        Tversky = (TP + smooth) / (TP + (self.alpha) * FP + (self.beta) * FN + smooth)
        FocalTversky = (1 - Tversky) ** self.gamma
        return FocalTversky


class Trainer:
    def __init__(self):
        self.is_cuda_available = torch.cuda.is_available()
        dev = "cuda:0" if self.is_cuda_available else "cpu"
        print("Device", dev)
        self.device = torch.device(dev)
        self.model = Model().to(self.device)
        self.epochs = 20
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self._original_model_path = "../input/hubmapmodel/model.pkl"
        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)
        self.eval_metrics = []

        def _make_bce():
            return torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor([5]).to(self.device),
                reduction="none",
            )

        if os.path.exists(self._original_model_path):
            self._data_dict = torch.load(
                self._original_model_path, map_location=self.device
            )
            self.start_positions = int(self._data_dict.get("epoch", 0))
            self.loss = self._data_dict.get("loss", _make_bce())
            if (
                isinstance(self.loss, torch.nn.BCEWithLogitsLoss)
                and getattr(self.loss, "reduction", "mean") != "none"
            ):
                self.loss = _make_bce()
            if "optimizer_state_dict" in self._data_dict:
                self.optim.load_state_dict(self._data_dict["optimizer_state_dict"])
            if "model_state_dict" in self._data_dict:
                self.model.load_state_dict(self._data_dict["model_state_dict"])
        else:
            self.start_positions = 0
            self.loss = _make_bce()

    def evaluate(self, epoch):
        val_set = get_val_set()
        TP = torch.tensor(0.0, device=self.device)
        FP = torch.tensor(0.0, device=self.device)
        FN = torch.tensor(0.0, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for ds in val_set:
                print(f"started processing image {ds.image}")
                loader = data.DataLoader(ds, 8, pin_memory=True, collate_fn=collate_fn)
                for batch_ndx, sample in enumerate(loader):
                    if not sample:
                        continue
                    target, mask, idx = sample
                    target = target.to(self.device, non_blocking=True)
                    mask = mask.to(self.device, non_blocking=True)

                    gt = mask[:, :, :, 0] > 0  # (B,H,W) bool
                    valid = mask[:, :, :, 1] > 0  # (B,H,W) bool

                    res = self.model(target)  # logits (B,1,H,W)
                    prob = torch.sigmoid(res).squeeze(1)  # (B,H,W)
                    pred = prob > 0.5  # bool (B,H,W)

                    pv = pred[valid]
                    gv = gt[valid]
                    TP += (pv & gv).sum().float()
                    FP += (pv & (~gv)).sum().float()
                    FN += ((~pv) & gv).sum().float()

                    del mask, target, res, prob, pred, gt, valid
                    torch.cuda.empty_cache()
                    gc.collect()

        denom = (2.0 * TP + FP + FN).clamp_min(1.0)
        score = (2.0 * TP) / denom
        print(f"epoch {epoch} evaluation", score)
        print(
            f"epoch {epoch} TP={int(TP.item())}, FP={int(FP.item())}, FN={int(FN.item())}"
        )
        self.eval_metrics.append(
            (
                float(score.detach().cpu()),
                int(TP.item()),
                int(FP.item()),
                int(FN.item()),
            )
        )
        self.model.train()
        gc.collect()

    def predict(self, image):
        transforms = []
        image = image.to(self.device)
        first_res = self.model(image)
        for t in transforms:
            res = self.model(t(image))
            first_res += res
            del res
            gc.collect()
        denom = max(1, (len(transforms) + 1))
        preds = first_res / denom
        del image, first_res
        gc.collect()
        return preds

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        print(f"start position {self.start_positions}")
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH, exist_ok=True)

        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")
            losses_stats = collections.defaultdict(int)

            for dataset in get_tiff_images():
                try:
                    loader = data.DataLoader(
                        dataset, 12, pin_memory=True, collate_fn=collate_fn
                    )
                    for batch_ndx, sample in enumerate(loader):
                        self.optim.zero_grad()
                        if not sample:
                            continue
                        target, mask, idx = sample
                        target = target.to(self.device, non_blocking=True)
                        mask = mask.to(self.device, non_blocking=True)

                        y = (mask[:, :, :, 0] > 0).float().unsqueeze(1)  # (B,1,H,W)
                        valid = (mask[:, :, :, 1] > 0).float().unsqueeze(1)  # (B,1,H,W)

                        logits = self.model(target)
                        loss_map = self.loss(logits, y)  # (B,1,H,W), reduction='none'
                        denom = valid.sum().clamp_min(1.0)
                        loss_result = (loss_map * valid).sum() / denom

                        loss_result.backward()
                        self.optim.step()

                        del mask, target, logits, loss_result, loss_map, y, valid
                        torch.cuda.empty_cache()
                        gc.collect()
                    print(f"data processed for {dataset.image}")
                finally:
                    del dataset, loader
                    torch.cuda.empty_cache()
                    gc.collect()

            self.evaluate(i)
            print("losses_stats", dict(losses_stats))
            with open(os.path.join(OUTPUT_PATH, "eval_metrics.csv"), "w+") as fw:
                fw.write(
                    "\n".join(
                        "\t".join(map(str, metric)) for metric in self.eval_metrics
                    )
                )
            print(f"saving epoch {i}")
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                },
                os.path.join(OUTPUT_PATH, f"model_{i}.pkl"),
            )
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                },
                self._model_path,
            )


def display_predictions(masks):
    import matplotlib.pyplot as plt

    masks = masks.squeeze()
    masks = masks > 0.5
    mask = 255 * masks
    plt.imshow(mask)




## === cell 5
torch.cuda.empty_cache()
trainer = Trainer()



## === cell 6
import pandas as pd


def _load_sample_submission():
    submission_file = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    sub_df = pd.read_csv(submission_file)
    if "img" in sub_df.columns:
        id_col = "img"
        pred_col = "pixels" if "pixels" in sub_df.columns else sub_df.columns[1]
    elif "id" in sub_df.columns:
        id_col = "id"
        pred_col = "predicted" if "predicted" in sub_df.columns else sub_df.columns[1]
    else:
        id_col = sub_df.columns[0]
        pred_col = sub_df.columns[1]
    return sub_df, id_col, pred_col


def get_test_dataset():
    sub_df, id_col, _ = _load_sample_submission()
    for image_id in sub_df[id_col].astype(str).tolist():
        print(f"openning {image_id}")
        try:
            dataset = HuBMAPDatasetPreprocessing(image_id, category="test")
            yield dataset
        except Exception as e:
            print(f"[WARN] Failed to open test image {image_id}: {e}")
            yield ("__FAILED__", image_id)


def rle_encode_less_memory(img):
    """
    img: numpy array or torch tensor, 1 - mask, 0 - background
    Returns run length as string formatted.
    Competition convention: 1-indexed, column-major (top-to-bottom, left-to-right),
    which corresponds to flattening in Fortran order.
    """
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = img.astype(np.uint8)
    if img.size == 0:
        return ""
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def make_submission(model):
    names, preds = [], []
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    with torch.no_grad():
        for item in get_test_dataset():
            if isinstance(item, tuple) and item[0] == "__FAILED__":
                image_id = str(item[1])
                names.append(image_id)
                preds.append("")  # empty RLE => empty mask
                continue

            ds = item
            print(ds.image, len(ds))
            loader = data.DataLoader(ds, 32, collate_fn=collate_fn, pin_memory=True)

            full_mask = np.zeros(ds.shape, dtype=np.uint8)

            for batch_num, batch in enumerate(loader):
                if not batch:
                    continue
                print(f"processing batch {batch_num}")
                image, _, idx = batch
                image = image.to(dev, non_blocking=True)

                logits = model(image)  # (B,1,h,w)
                prob = torch.sigmoid(logits)  # (B,1,h,w)

                if ds.reduce != 1:
                    prob = torch.nn.functional.interpolate(
                        prob,
                        scale_factor=ds.reduce,
                        mode="bilinear",
                        align_corners=False,
                    )
                prob = prob.squeeze(1)  # (B,ds.sz,ds.sz)
                binmask = (prob > 0.5).to(torch.uint8)

                if torch.is_tensor(idx):
                    idx_list = idx.detach().cpu().tolist()
                else:
                    idx_list = list(idx)

                binmask_np = binmask.detach().cpu().numpy()

                for i, ndx in enumerate(idx_list):
                    ndx = int(ndx)
                    n0, n1 = ndx // ds.n1max, ndx % ds.n1max
                    x0 = -ds.pad0 // 2 + n0 * ds.sz
                    y0 = -ds.pad1 // 2 + n1 * ds.sz

                    p00, p01 = max(0, x0), min(x0 + ds.sz, ds.shape[0])
                    p10, p11 = max(0, y0), min(y0 + ds.sz, ds.shape[1])

                    tx0, tx1 = p00 - x0, p01 - x0
                    ty0, ty1 = p10 - y0, p11 - y0

                    if (p01 > p00) and (p11 > p10):
                        patch = binmask_np[i, tx0:tx1, ty0:ty1]
                        full_mask[p00:p01, p10:p11] |= patch.astype(np.uint8)

                del image, logits, prob, binmask, binmask_np
                torch.cuda.empty_cache()
                gc.collect()
                print(f"batch {batch_num} is done")

            rle = rle_encode_less_memory(full_mask)
            names.append(str(ds.image))
            preds.append(rle)

            del full_mask, ds
            gc.collect()

    sub_df, id_col, pred_col = _load_sample_submission()
    out = pd.DataFrame({id_col: [str(x) for x in names], pred_col: preds})

    order = sub_df[id_col].astype(str).tolist()
    out = out.set_index(id_col).reindex(order).reset_index()

    if list(sub_df.columns) == ["id", "predicted"]:
        out.columns = ["id", "predicted"]

    out.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with", len(out), "rows")
    print("Columns:", out.columns.tolist())


model = Model()
loaded = False

local_ckpt = "/kaggle/working/models/model.pkl"
if os.path.exists(local_ckpt):
    data_dict = torch.load(local_ckpt, map_location=torch.device("cpu"))
    if "model_state_dict" in data_dict:
        model.load_state_dict(data_dict["model_state_dict"])
        loaded = True

if not loaded:
    model.load_state_dict(trainer.model.state_dict())
    loaded = True

make_submission(model)



## === cell 7
"""
Optional debugging/visualization cell kept commented out in original.
"""
