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

0.869651476780555

# 6. Current score

0.07486

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Your code didn’t yield a Kaggle score because it’s producing an invalid submission format: it writes columns `img,pixels` but this competition’s `sample_submission.csv` requires `id,predicted`. I make the smallest possible changes to (1) write the correct column names and (2) ensure the row order matches the sample submission IDs exactly (preventing silent misalignment). I also fix a mask reconstruction bug caused by filtering tiles in `collate_fn` without compensating for missing indices (your current `mask_tiles.view(...)` can break or scramble tiles), by assembling tiles into a full padded canvas using the tile coordinates derived from `idx`. These changes preserve your core model and inference semantics (still logits > 0 threshold, same tiling) while producing a valid CSV.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission being *valid CSV-wise* but *semantically wrong for this competition*: you’re segmenting the TIFF pixels, but HuBMAP 20 expects predictions for the *provided JSON annotation coordinate space* (and the sample submission only has 3 test IDs with no TIFFs here). The smallest score-moving fix is to keep your model and tiling logic untouched, but switch inference input from missing/nonexistent `test/*.tiff` to the available `test/*.json` anatomical-structure polygons by rasterizing them into a 3‑channel “image” tensor and running your same model on that. Additionally, ensure we read the submission IDs exactly and always emit `id,predicted` with correct ordering. These changes preserve your architecture/training/inference semantics (same UNet, same >0 threshold, same RLE) while making the pipeline actually produce meaningful masks for the provided test set, which should move the Dice score upward from 0.0 toward your target.'
- What this solution (achieved 0.07989) has done: 'I make the submission semantically valid for HuBMAP kidney segmentation by switching test-time inference back to the actual test TIFF images (the metric is computed on TIFF pixel space, and the provided TIFFs exist inside the competition dataset). This is a minimal score-moving change because it preserves your model, tiling, normalization, thresholding, and RLE encoding; it only fixes the input source used for predictions (JSON anatomical-structure masks are not the evaluation target). I also ensure the submission uses the exact `id,predicted` columns and ID order from `sample_submission.csv`, and I keep your tile filtering/collation behavior unchanged to avoid altering core semantics. Finally, I add a tiny safeguard so if reading a TIFF fails for any ID, it falls back to the previous JSON-as-image path rather than producing an empty/invalid submission.'
- What this solution (achieved 0.0) has done: 'Your current score gap to the target is very large, so we need a real (but still minimal) score-moving fix without changing your UNet, loss, or tiling approach. The biggest issue is that your inference assembles a full mask in *tile grid space* but never upsamples it back to the original TIFF resolution when `use_compressed=True` (`reduce=4`), so the submitted RLE is at 1/4 scale and badly mismatched to the evaluation masks. I keep the same model output thresholding and tile filtering, but (1) reconstruct the mask at reduced scale, (2) crop padding, then (3) upsample back to the original image size using nearest-neighbor before RLE encoding. I also make the `collate_fn` in inference keep the `idx` tensor aligned (by filtering within the collate and returning stacked tensors), which avoids rare misalignment edge cases when some tiles are dropped.'
- What this solution (achieved 0.02632) has done: 'I make two minimal, score-moving fixes while preserving your UNet, tiling, thresholding, and loss semantics. First, your `reduce` logic is currently inverted (`reduce=4 if not use_compressed else 1`), which makes you train/infer at full resolution when `use_compressed=True`; I correct it to `reduce=4 if use_compressed else 1` so the existing “compressed” path behaves as intended and your mask reconstruction/upsample becomes consistent. Second, your evaluation code computes FP/FN with swapped terms (it doesn’t affect training but can mislead you); I correct that without changing the actual predictions. Everything else (data paths, model, collate filtering, logits>0 threshold, RLE, `id,predicted` submission ordering) stays the same and it still write `submission.csv`.'
- What this solution (achieved 0.07388) has done: 'Your score is far below the target, so we should make the smallest changes that correct likely *inference-time semantic mismatches* without touching your UNet, loss, tiling scheme, or training loop. The main issue is that your tile placement during reconstruction ignores the dataset’s padded-origin offset (the negative `x0,y0` in `__getitem__`), so tiles get shifted in the assembled canvas; I reconstruct using the same `x0,y0` math as the dataset and then crop/upsample as you already do. Second, your RLE encoder forcibly zeroes the first/last pixel, which can truncate valid positives at borders; I switch to the standard Kaggle-safe sentinel concatenation (same encoding semantics, no border corruption). These two fixes are minimal and should move Dice up materially while keeping your core model/training/inference intact and still writing a valid `submission.csv` with `id,predicted` in sample order.'
- What this solution (achieved 1e-05) has done: 'Your current gap to the target is very large, so the most likely “minimal-but-real” score movement comes from fixing inference-time reconstruction/encoding mismatches rather than changing the UNet or training. I (1) correct the tile-to-canvas placement math to use tile-grid coordinates directly (removing the extra pad-origin shift that can systematically misplace tiles), (2) apply the same background-tile filter to the *predictions* (force skipped tiles to 0 instead of leaving default canvas values that can become inconsistent), and (3) fix RLE encoding to match the competition’s required pixel order (top-to-bottom then left-to-right, i.e., flatten without transpose). These changes keep your model, thresholding (logits > 0), tiling, and loss/training intact, but make the submitted mask align better with the evaluation space, which should increase Dice toward your target.'
- What this solution (achieved 0.07486) has done: 'Your current score (1e-05) is far below the target, so we should focus on a minimal, high-impact semantic fix in inference rather than changing the UNet or training. The most likely cause is incorrect RLE pixel ordering: the competition numbers pixels top-to-bottom then left-to-right, which corresponds to flattening the mask in Fortran order (`order='F'`), not default C order. I change only the RLE encoder to use Fortran order (and keep the same sentinel padding), while keeping your tiling, thresholding (`logits>0`), reconstruction, and submission schema unchanged. This should move Dice materially upward toward your target without altering core model logic.'

# 9. Code solution

## === cell 0
import os
import glob
import json
import gc
import csv

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import torchvision
import matplotlib.pyplot as plt

mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])


def _read_tiff_robust(path: str) -> np.ndarray:
    """
    Fix: OpenCV can fail on very large TIFFs due to pixel guard. Use tifffile first (fast and robust),
    then skimage.io.imread, then finally OpenCV as last resort.
    """
    img = None

    try:
        import tifffile

        img = tifffile.imread(path)
    except Exception:
        img = None

    if img is None:
        try:
            from skimage import io

            img = io.imread(path)
        except Exception:
            img = None

    if img is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Could not read TIFF: {path}")

    if img.ndim == 2:
        img = img[..., None]

    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))

    return img


def _read_json_features(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _rasterize_json_polygons(features, h: int, w: int) -> np.ndarray:
    """
    Minimal helper used in both train mask creation and test anatomical-structure fallback:
    rasterize polygons to a binary mask in the JSON coordinate space.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for feat in features:
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", None)
        if not coords:
            continue
        try:
            exterior = coords[0]
        except Exception:
            continue
        if exterior is None or len(exterior) == 0:
            continue
        pts = np.asarray(exterior, dtype=np.float32)
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask


class HuBMAPDatasetPreprocessing(Dataset):
    """
    rasterio is not available in this Kaggle environment.
    Replace rasterio-based window reads with robust TIFF reads + numpy slicing.
    Replace rasterio.mask.mask with polygon rasterization via cv2.fillPoly.
    Core logic preserved: same tiling, padding, downscale (reduce), normalization, and filter.
    """

    def __init__(
        self, image, category="train", use_compressed=False, apply_jitter=False
    ):
        self.mask_folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.use_compressed = use_compressed
        self.image = image
        self.category = category

        self.reduce = 4 if use_compressed else 1

        self.apply_jitter = apply_jitter

        self._img = None
        self._mask = None
        self._get_image()

    def _get_image_path(self):
        return os.path.join(self.folder, self.category, f"{self.image}.tiff")

    def _get_masks(self):
        file_path = os.path.join(self.mask_folder, self.category, f"{self.image}.json")
        with open(file_path, "r") as f:
            return json.load(f)

    def _rasterize_polygons(self, h, w):
        feats = self._get_masks()
        return _rasterize_json_polygons(feats, h, w)

    def _get_image(self):
        if self._img is not None:
            return self._img

        file_path = self._get_image_path()
        self._img = _read_tiff_robust(file_path)

        if self._img.ndim == 2:
            self._img = np.repeat(self._img[..., None], 3, axis=2)
        elif self._img.shape[2] > 3:
            self._img = self._img[:, :, :3]
        elif self._img.shape[2] == 1:
            self._img = np.repeat(self._img, 3, axis=2)

        self.shape = self._img.shape[:2]  # (H, W)
        self.sz = 256 * self.reduce
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        if self.category == "train":
            if self._mask is None:
                self._mask = self._rasterize_polygons(
                    self.shape[0], self.shape[1]
                ).astype(bool)

        return self._img

    def close(self):
        pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
        self._img = None
        self._mask = None

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        img_full = self._get_image()
        H, W = self.shape
        n0, n1 = idx // self.n1max, idx % self.n1max

        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz, H)
        p10, p11 = max(0, y0), min(y0 + self.sz, W)

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = img_full[
            p00:p01, p10:p11, :
        ].astype(np.uint8)

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        if self.category == "train":
            mask = np.zeros((self.sz, self.sz), np.uint8)
            m = self._mask[p00:p01, p10:p11].astype(np.uint8)
            mask[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = m
            if self.reduce != 1:
                mask = cv2.resize(
                    mask,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_AREA,
                )
            mask = mask[..., None]  # (H,W,1)
        else:
            mask = torch.zeros(1)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        _, s, _ = cv2.split(hsv)

        result_tensor = torch.from_numpy((img / 255.0 - mean) / std).float()
        result_tensor = result_tensor.permute(2, 0, 1).float()

        if self.category == "train" and self.apply_jitter is True:
            jitter = torchvision.transforms.ColorJitter(brightness=0.1)
            result_tensor = jitter(result_tensor)

        if (s > 40).sum() <= 1000 or img.sum() <= 1000:
            return result_tensor, mask, -1
        else:
            return result_tensor, mask, idx




## === cell 1
"""
dataset = HuBMAPDatasetPreprocessing('0486052bb', use_compressed=True)

fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(2, 5)
c = 0
for i in range(100):
    sample, mask, idx = dataset[i]
    sample = sample.permute(1,2,0) * 255
    
    if idx == -1 or (isinstance(mask, np.ndarray) and mask.sum() == 0):
        continue
    if c == 5:
        break
    plots[0][c].imshow(sample.int().numpy())
    plots[0][c].set_title(f'Sample #{c}')
    plots[1][c].imshow(mask[..., 0] if isinstance(mask, np.ndarray) else mask)
    plots[1][c].set_title(f'Sample #{c}')
    c += 1
"""


## === cell 2
"""

dataset_test = HuBMAPDatasetPreprocessing('0486052bb', 'test', use_compressed=True)
fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(2, 5)
c = 0
for i in range(100):
    sample, mask, idx = dataset_test[i]
    sample = sample.permute(1,2,0)
    if idx == -1:
        continue
    if c == 5:
        break
    plots[0][c].imshow(sample)
    plots[0][c].set_title(f'Sample #{c}')
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
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x.clone())


class Model(nn.Module):
    """The base class for the encoder-decoder architecture."""

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, **kwargs):
        super(Model, self).__init__(**kwargs)
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
import numpy

OUTPUT_PATH = "/kaggle/working/models"


def collate_fn(x):
    x = list(filter(lambda t: t[-1] != -1, x))
    if len(x) == 0:
        return []
    return data.dataloader.default_collate(x)


VALSET = ["e79de561c", "cb2d976f4"]


def get_tiff_images():
    input_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
    for images in glob.glob(input_folder):
        bname = os.path.basename(images)
        name = os.path.splitext(bname)[0]
        if name not in VALSET:
            dataset = HuBMAPDatasetPreprocessing(
                name, apply_jitter=False, use_compressed=True
            )
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
        super(FocalTverskyLoss, self).__init__()

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
        self.epochs = 30
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self._original_model_path = "../input/hubmapmodel/model (13).pkl"

        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)
        self.eval_metrics = []

        self.loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([5]).to(self.device)
        )

        if os.path.exists(self._original_model_path):
            if not self.is_cuda_available:
                self._data_dict = torch.load(
                    self._original_model_path, map_location=torch.device("cpu")
                )
            else:
                self._data_dict = torch.load(self._original_model_path)
            self.start_positions = self._data_dict.get("epoch", 0)
            if "optimizer_state_dict" in self._data_dict:
                self.optim.load_state_dict(self._data_dict["optimizer_state_dict"])
            if "model_state_dict" in self._data_dict:
                self.model.load_state_dict(self._data_dict["model_state_dict"])
        else:
            self.start_positions = 0

    def evaluate(self, epoch, display=False):
        val_set = get_val_set()
        numerator, denominator = 0, 0
        TP, FP, FN = 0, 0, 0
        for ds in val_set:
            print(f"started processing image {ds.image}")
            loader = data.DataLoader(ds, 9, pin_memory=True, collate_fn=collate_fn)
            for batch_ndx, sample in enumerate(loader):
                if not sample:
                    continue
                target, mask, idx = sample
                target = target.to(self.device)
                mask = mask.to(self.device)
                mask = (mask[:, :, :, :1] > 0).squeeze()
                res = self.model(target)
                result = (res > 0).squeeze()
                for i in range(len(mask)):
                    inputs = mask[i].int()
                    targets = result[i].int()
                    tp = inputs * targets
                    fp = targets * (1 - inputs)
                    fn = (1 - targets) * inputs
                    TP += tp.sum()
                    FP += fp.sum()
                    FN += fn.sum()
                    numerator += 2 * torch.sum(mask[i].float() * result[i].float())
                    denominator += torch.sum(mask[i].float() + result[i].float())
                del mask, target, result, res
                torch.cuda.empty_cache()
                gc.collect()

        print(
            f"epoch {epoch} evaluation",
            numerator / denominator if denominator != 0 else 0,
        )
        print(f"epoch {epoch} TP={TP}, FP={FP}, FN={FN}")
        self.eval_metrics.append(
            (numerator / denominator if denominator != 0 else 0, TP, FP, FN)
        )
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
        preds = first_res / (len(transforms) + 1)
        del image, first_res
        gc.collect()
        return preds

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        print(f"start position {self.start_positions}")
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")
            for dataset in get_tiff_images():
                loader = None
                try:
                    loader = data.DataLoader(
                        dataset, 12, pin_memory=True, collate_fn=collate_fn
                    )
                    for batch_ndx, sample in enumerate(loader):
                        self.optim.zero_grad()
                        if not sample:
                            continue
                        target, mask, idx = sample
                        mask = mask[:, :, :, :1] > 0
                        result = self.model.forward(target.to(self.device)).to(
                            self.device
                        )
                        mask = mask.to(self.device).float().permute(0, 3, 1, 2)
                        loss_result = self.loss(result, mask)
                        loss_result.backward()
                        self.optim.step()
                        del mask, target
                        torch.cuda.empty_cache()
                        gc.collect()
                    print(f"data processed for {dataset.image}")
                finally:
                    del dataset
                    if loader is not None:
                        del loader
                    torch.cuda.empty_cache()
                    gc.collect()

            self.evaluate(i)
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


def display_predictions(tp, fp, fn, subplot):
    result = np.zeros(fp.shape)
    result += tp.cpu().numpy() * 255
    result += fn.cpu().numpy() * 64
    subplot.imshow(result)




## === cell 5
torch.cuda.empty_cache()
trainer = Trainer()




## === cell 6
import pandas as pd


def collate_fn(x):
    x = [t for t in x if t[-1] != -1]
    if len(x) == 0:
        return []
    images = torch.stack([t[0] for t in x], dim=0)
    masks = torch.stack(
        [t[1] if torch.is_tensor(t[1]) else torch.from_numpy(t[1]) for t in x], dim=0
    )
    idxs = torch.tensor([int(t[2]) for t in x], dtype=torch.long)
    return images, masks, idxs


def get_test_ids_in_submission_order():
    sample_path = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    df = pd.read_csv(sample_path)
    id_col = "id" if "id" in df.columns else df.columns[0]
    return df[id_col].astype(str).tolist(), df, id_col


class HuBMAPJsonAsImageDataset(Dataset):
    """
    Fallback-only dataset: build an 'image' from the available anatomical-structure JSON polygon mask.
    Kept so submission is always produced even if a TIFF is unreadable for some ID.
    """

    def __init__(self, image_id: str, category="test", use_compressed=True):
        self.folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.image = image_id
        self.category = category
        self.use_compressed = use_compressed
        self.reduce = 1
        self._img = None

        info_path = os.path.join(self.folder, "HuBMAP-20-dataset_information.csv")
        info = pd.read_csv(info_path)
        row = info[
            info["image_file"].astype(str).str.replace(".tiff", "", regex=False)
            == image_id
        ]
        if len(row) == 0:
            self.shape = (1024, 1024)
        else:
            self.shape = (
                int(row.iloc[0]["height_pixels"]),
                int(row.iloc[0]["width_pixels"]),
            )

        self.sz = 256 * self.reduce
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        self._build_image_from_anatomical_json()

    def _build_image_from_anatomical_json(self):
        h, w = self.shape
        json_path = os.path.join(
            self.folder, self.category, f"{self.image}-anatomical-structure.json"
        )
        feats = _read_json_features(json_path)
        anat = _rasterize_json_polygons(feats, h, w).astype(np.uint8)  # 0/1
        img = np.stack([anat * 255, anat * 255, anat * 255], axis=-1).astype(np.uint8)
        self._img = img

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        img_full = self._img
        H, W = self.shape
        n0, n1 = idx // self.n1max, idx % self.n1max

        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz, H)
        p10, p11 = max(0, y0), min(y0 + self.sz, W)

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = img_full[
            p00:p01, p10:p11, :
        ]

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        _, s, _ = cv2.split(hsv)

        result_tensor = torch.from_numpy((img / 255.0 - mean) / std).float()
        result_tensor = result_tensor.permute(2, 0, 1).float()

        if (s > 40).sum() <= 1000 or img.sum() <= 1000:
            return result_tensor, torch.zeros(1), -1
        else:
            return result_tensor, torch.zeros(1), idx


def get_test_dataset():
    """
    Run inference on the real test TIFFs (evaluation is on TIFF pixel masks).
    Keep same tiling/filtering/normalization as dataset class.
    Fallback to JSON-as-image only if TIFF is missing/unreadable for an ID.
    """
    ids, _, _ = get_test_ids_in_submission_order()
    for image_id in ids:
        print(f"opening {image_id}")
        try:
            dataset = HuBMAPDatasetPreprocessing(
                image_id, category="test", use_compressed=True
            )
        except Exception as e:
            print(
                f"TIFF load failed for {image_id} ({e}); falling back to anatomical JSON image."
            )
            dataset = HuBMAPJsonAsImageDataset(
                image_id, category="test", use_compressed=True
            )
        yield dataset


def rle_encode_less_memory(img: np.ndarray) -> str:
    """
    Score-moving minimal fix: correct pixel order to match the competition statement.
    "Pixels are numbered from top to bottom, then left to right" corresponds to flattening
    in Fortran order (column-major) for a (H,W) array.
    Keeping the same sentinel padding and run computation.
    """
    pixels = img.flatten(order="F").astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _load_best_available_weights(model: nn.Module):
    candidates = [
        "/kaggle/working/models/model.pkl",
        "/kaggle/working/model.pkl",
    ]
    ckpt_path = None
    for p in candidates:
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        print("No checkpoint found in /kaggle/working; using untrained model weights.")
        return model

    print(f"Loading checkpoint: {ckpt_path}")
    data_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    if isinstance(data_dict, dict) and "model_state_dict" in data_dict:
        model.load_state_dict(data_dict["model_state_dict"], strict=True)
    else:
        model.load_state_dict(data_dict, strict=True)
    return model


def make_submission(model):
    ids, sample_df, id_col = get_test_ids_in_submission_order()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    model.eval()

    pred_map = {}

    with torch.no_grad():
        for ds in get_test_dataset():
            print(ds.image, len(ds))
            loader = data.DataLoader(ds, 32, collate_fn=collate_fn, pin_memory=True)

            tile_sz = ds.sz // ds.reduce

            full_h = ds.n0max * tile_sz
            full_w = ds.n1max * tile_sz
            full_mask = np.zeros((full_h, full_w), dtype=np.uint8)

            for batch_num, data_ in enumerate(loader):
                if not data_:
                    continue
                print(f"processing batch {batch_num}")
                image, _, idx = data_
                image = image.to(dev)

                result = model(image).squeeze(1)  # (B, H, W) logits

                for i, ndx in enumerate(idx.tolist()):
                    n0 = int(ndx) // ds.n1max
                    n1 = int(ndx) % ds.n1max

                    y_canvas = int(n0 * tile_sz)
                    x_canvas = int(n1 * tile_sz)

                    full_mask[
                        y_canvas : y_canvas + tile_sz, x_canvas : x_canvas + tile_sz
                    ] = ((result[i] > 0).to(torch.uint8).cpu().numpy())

                del result, image
                torch.cuda.empty_cache()
                gc.collect()
                print(f"batch {batch_num} is done")

            top = (ds.pad0 // 2) // ds.reduce
            left = (ds.pad1 // 2) // ds.reduce
            bottom_crop = (ds.pad0 - ds.pad0 // 2) // ds.reduce
            right_crop = (ds.pad1 - ds.pad1 // 2) // ds.reduce

            if bottom_crop > 0:
                mask_small = full_mask[top:-bottom_crop, :]
            else:
                mask_small = full_mask[top:, :]

            if right_crop > 0:
                mask_small = mask_small[:, left:-right_crop]
            else:
                mask_small = mask_small[:, left:]

            target_h, target_w = ds.shape  # original TIFF H,W
            if (mask_small.shape[0], mask_small.shape[1]) != (target_h, target_w):
                mask = cv2.resize(
                    mask_small,
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)
            else:
                mask = mask_small.astype(np.uint8)

            rle = rle_encode_less_memory(mask)
            pred_map[ds.image] = rle

            del ds, full_mask, mask_small, mask
            gc.collect()

    predicted_col = (
        "predicted" if "predicted" in sample_df.columns else sample_df.columns[1]
    )
    out = pd.DataFrame(
        {
            "id": ids,
            predicted_col: [pred_map.get(i, "") for i in ids],
        }
    )
    out.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", out.shape, "columns:", list(out.columns))
    print("Preview:\n", out.head())


model = Model()
model = _load_best_available_weights(model)
make_submission(model)




## === cell 7
"""
(from skimage import io) visualization / debugging cell intentionally left unused
"""




## === cell 8
"""
Debug visualization cell intentionally left as comment.
"""




## === cell 9
"""
Post-hoc RLE comparison cell intentionally left as comment.
"""
