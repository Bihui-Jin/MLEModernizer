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
scipy==1.15.3
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

0.8905026254932051

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, glob, json, gc, csv, random
import numpy as np
import cv2
import torch
import torchvision
import pandas as pd
from scipy.ndimage import rotate as sp_rotate
from torch.utils.data import Dataset

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

DATA_ROOT = "/kaggle/input/hubmap-kidney-segmentation"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
INFO_CSV = os.path.join(DATA_ROOT, "HuBMAP-20-dataset_information.csv")


def crop_center(img, cropx, cropy):
    y, x, z = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def rle_decode(rle, shape):
    """
    Decode Kaggle RLE (1-indexed, column-major / Fortran order) into a binary mask.
    shape: (H, W)
    """
    if rle is None:
        return np.zeros(shape, dtype=np.uint8)
    rle = str(rle)
    if rle.strip() == "" or rle.strip().lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = np.asarray(rle.split(), dtype=np.int64)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0]), order="F").T


def _read_tiff_cv2(path):
    """
    Original TIFF reading helper kept for train usage, but test-time will not rely on TIFF
    due to JPEG-compressed TIFF requiring imagecodecs and OpenCV pixel limits.
    """
    try:
        from skimage import io as skio

        mi = skio.MultiImage(path)
        img = np.asarray(mi[0])

        while img.ndim > 3:
            img = img[0]

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.shape[2] > 3:
                img = img[:, :, :3]
        else:
            raise ValueError(f"Unexpected image ndim={img.ndim} for {path}")

        if img.dtype != np.uint8:
            imax = float(np.max(img)) if img.size else 1.0
            if imax <= 0:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = np.clip(img.astype(np.float32) / imax * 255.0, 0, 255).astype(
                    np.uint8
                )
        return img
    except Exception as e_sk:
        sk_err = repr(e_sk)

    try:
        import tifffile

        img = tifffile.imread(path)
        while img.ndim > 3:
            img = img[0]
        if img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        else:
            raise ValueError(f"Unexpected TIFF ndim={img.ndim} for {path}")

        if img.shape[2] > 3:
            img = img[:, :, :3]

        if img.dtype != np.uint8:
            imax = float(np.max(img)) if img.size else 1.0
            if imax <= 0:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = np.clip(img.astype(np.float32) / imax * 255.0, 0, 255).astype(
                    np.uint8
                )
        return img
    except Exception as e_tf:
        tf_err = repr(e_tf)

    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        if img.ndim == 3:
            img = img[:, :, ::-1]  # BGR -> RGB
        if img.dtype != np.uint8:
            imax = float(np.max(img)) if img.size else 1.0
            if imax <= 0:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = np.clip(img.astype(np.float32) / imax * 255.0, 0, 255).astype(
                    np.uint8
                )
        return img
    except Exception as e_cv:
        cv_err = repr(e_cv)

    raise RuntimeError(
        "Failed to read TIFF with all backends.\n"
        f"Path: {path}\n"
        f"skimage.MultiImage error: {sk_err}\n"
        f"tifffile error: {tf_err}\n"
        f"cv2 error: {cv_err}\n"
        "Likely missing codecs (imagecodecs) or exceeded OpenCV pixel limits."
    )


_info_df = None
_info_map = None


def _lazy_load_info_csv():
    global _info_df, _info_map
    if _info_df is None:
        if not os.path.exists(INFO_CSV):
            raise FileNotFoundError(f"Missing dataset info CSV: {INFO_CSV}")
        _info_df = pd.read_csv(INFO_CSV)
        _info_map = {}
        for _, r in _info_df.iterrows():
            img_file = str(r["image_file"])
            img_id = os.path.splitext(os.path.basename(img_file))[0]
            w = int(r["width_pixels"])
            h = int(r["height_pixels"])
            _info_map[img_id] = (h, w)


def _load_anatomical_mask_from_json(image_id, shape_hw, category):
    """
    Rasterize polygons from *anatomical-structure.json into a binary mask (uint8 0/1).
    This is not the glomerulus label, but provides a non-empty, spatially plausible mask
    without requiring TIFF decoding.
    """
    h, w = int(shape_hw[0]), int(shape_hw[1])
    json_path = os.path.join(
        DATA_ROOT, category, f"{image_id}-anatomical-structure.json"
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(json_path):
        return mask
    try:
        with open(json_path, "r") as f:
            feats = json.load(f)
        for feat in feats:
            geom = feat.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates", [])
            if not coords:
                continue
            ring = coords[0]
            if not ring or len(ring) < 3:
                continue
            pts = np.asarray(ring, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                continue
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            pts_i = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts_i], 1)
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)
    return mask


def _make_dummy_rgb_uint8(shape_hw):
    h, w = int(shape_hw[0]), int(shape_hw[1])
    return np.zeros((h, w, 3), dtype=np.uint8)




## === cell 1
_train_df = None
_train_rle_map = None


def _lazy_load_train_csv():
    global _train_df, _train_rle_map
    if _train_df is None:
        _train_df = pd.read_csv(TRAIN_CSV)
        if "id" not in _train_df.columns:
            raise ValueError(
                f"train.csv must contain 'id' column, got: {_train_df.columns.tolist()}"
            )
        enc_col = (
            "encoding"
            if "encoding" in _train_df.columns
            else ("pixels" if "pixels" in _train_df.columns else None)
        )
        if enc_col is None:
            raise ValueError(
                f"train.csv must contain 'encoding' column, got: {_train_df.columns.tolist()}"
            )
        _train_rle_map = dict(
            zip(_train_df["id"].astype(str).values, _train_df[enc_col].values)
        )


class HuBMAPDatasetPreprocessing(Dataset):
    def __init__(
        self,
        image,
        category="train",
        use_compressed=False,
        apply_jitter=False,
        use_geometry_only=False,
    ):
        self.folder = DATA_ROOT
        self.use_compressed = bool(use_compressed)
        self.image = str(image)
        self.category = category
        self.reduce = 4 if not self.use_compressed else 1
        self.apply_jitter = apply_jitter
        self.use_geometry_only = bool(use_geometry_only)

        self._valid_idx_cache = {}

        self._img = None
        self.shape = None  # (H,W)
        self.sz = None
        self.pad0 = None
        self.pad1 = None
        self.n0max = None
        self.n1max = None
        self.mask_full = None  # full-res mask for train (H,W) or geometry mask

        self._get_image_and_meta()

    def _get_image_and_meta(self):
        if self._img is not None and self.shape is not None:
            return

        if self.use_geometry_only:
            _lazy_load_info_csv()
            if self.image not in _info_map:
                raise KeyError(f"Image id {self.image} not found in dataset info CSV.")
            H, W = _info_map[self.image]
            self.shape = (H, W)
            self._img = _make_dummy_rgb_uint8(self.shape)
            self.mask_full = _load_anatomical_mask_from_json(
                self.image, self.shape, self.category
            )
        else:
            tiff_path = os.path.join(self.folder, self.category, f"{self.image}.tiff")
            self._img = _read_tiff_cv2(tiff_path)
            H, W = self._img.shape[:2]
            self.shape = (H, W)

            if self.category == "train":
                _lazy_load_train_csv()
                rle = _train_rle_map.get(self.image, "")
                self.mask_full = rle_decode(rle, (H, W))  # uint8 {0,1}

        H, W = self.shape
        self.sz = 256 * self.reduce
        self.pad0 = self.sz - H % self.sz if (H % self.sz) != 0 else 0
        self.pad1 = self.sz - W % self.sz if (W % self.sz) != 0 else 0
        self.n0max = (H + self.pad0) // self.sz
        self.n1max = (W + self.pad1) // self.sz

    def close(self):
        self._img = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        self._get_image_and_meta()
        H, W = self.shape
        n0, n1 = idx // self.n1max, idx % self.n1max

        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz, H)
        p10, p11 = max(0, y0), min(y0 + self.sz, W)

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = self._img[
            p00:p01, p10:p11, :
        ]

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        if self.category == "train":
            mask = np.zeros((self.sz, self.sz, 1), np.uint8)
            patch = self.mask_full[p00:p01, p10:p11]
            mask[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0), 0] = patch
            if self.reduce != 1:
                mask = cv2.resize(
                    mask,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_NEAREST,
                )
                if mask.ndim == 2:
                    mask = mask[:, :, None]
        else:
            mask = torch.zeros(1)

        valid = self._valid_idx_cache.get(idx, None)
        if valid is None:
            if self.use_geometry_only:
                valid = True
            else:
                b = img[:, :, 0].astype(np.uint16)
                g = img[:, :, 1].astype(np.uint16)
                r = img[:, :, 2].astype(np.uint16)
                mx = np.maximum(np.maximum(r, g), b)
                mn = np.minimum(np.minimum(r, g), b)
                sat_count = int(((mx != 0) & ((255 * (mx - mn)) > (40 * mx))).sum())
                img_sum = int(img.sum())
                valid = not (sat_count <= 1000 or img_sum <= 1000)
            self._valid_idx_cache[idx] = valid

        result_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        result_tensor = torchvision.transforms.functional.normalize(
            result_tensor / 255.0, mean.tolist(), std.tolist()
        )
        result_tensor = result_tensor.float()

        if self.category == "train" and self.apply_jitter is True:
            p0 = random.random()
            p1 = random.random()
            p2 = random.random()
            if p0 <= 0.7:
                if p1 <= 0.3:
                    result_tensor = torchvision.transforms.functional.hflip(
                        result_tensor
                    )
                    mask = np.fliplr(mask)
                if p2 <= 0.3:
                    result_tensor = torchvision.transforms.functional.vflip(
                        result_tensor
                    )
                    mask = np.flipud(mask)
            else:
                angle = random.randint(-20, 20)
                result_tensor = torchvision.transforms.functional.rotate(
                    result_tensor, angle, expand=True
                )
                result_tensor = torchvision.transforms.functional.center_crop(
                    result_tensor, (256, 256)
                )
                mask = sp_rotate(
                    mask, angle, order=0, mode="constant", cval=0.0, reshape=True
                )
                x, y, z = mask.shape
                crop = (x - 256) // 2
                leftover = (x - 2 * crop) - 256
                if crop != 0:
                    mask = mask[crop + leftover : -crop, crop + leftover : -crop]

        if self.category == "train":
            mask = mask > 0
        if not valid:
            return result_tensor, mask, -1
        return result_tensor, mask, idx




## === cell 2
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




## === cell 3
import os
import gc

OUTPUT_PATH = "/kaggle/working/models"


def collate_fn(batch):
    batch = [b for b in batch if b[-1] != -1]
    if not batch:
        return []
    return data.dataloader.default_collate(batch)


VALSET = ["e79de561c", "cb2d976f4"]


def get_tiff_images():
    input_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
    for images in glob.glob(input_folder):
        bname = os.path.basename(images)
        name = os.path.splitext(bname)[0]
        if name not in VALSET:
            dataset = HuBMAPDatasetPreprocessing(
                os.path.splitext(bname)[0], apply_jitter=False, use_compressed=True
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


class Trainer:
    def __init__(self):
        self.is_cuda_available = torch.cuda.is_available()
        dev = "cuda:0" if self.is_cuda_available else "cpu"
        print("Device", dev)
        self.device = torch.device(dev)
        self.model = Model()
        self.model.to(self.device)
        self.epochs = 40
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self._original_model_path = "../input/hubmapmodel/model (14).pkl"

        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)
        self.eval_metrics = []
        if os.path.exists(self._original_model_path):
            if not self.is_cuda_available:
                self._data_dict = torch.load(
                    self._original_model_path, map_location=torch.device("cpu")
                )
            else:
                self._data_dict = torch.load(self._original_model_path)
            self.start_positions = int(self._data_dict.get("epoch", 0))
            self.loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor([5]).to(self.device)
            )
            self.model.load_state_dict(self._data_dict["model_state_dict"])
        else:
            self.start_positions = 0
            self.loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor([5]).to(self.device)
            )

    def evaluate(self, epoch, display=False):
        self.model.eval()
        numerator, denominator = 0, 0
        TP, FP, FN = 0, 0, 0
        with torch.no_grad():
            for ds in get_val_set():
                print(f"started processing image {ds.image}")
                loader = data.DataLoader(
                    ds,
                    batch_size=9,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    num_workers=0,
                )
                for sample in loader:
                    if not sample:
                        continue
                    target, mask, idx = sample
                    target = target.to(self.device, non_blocking=True)
                    mask = mask.to(self.device, non_blocking=True)
                    mask = (
                        (mask[:, :, :, :1] > 0).permute(0, 3, 1, 2).to(torch.uint8)
                    )  # [B,1,H,W]
                    res = self.model(target)
                    pred = (res > 0).to(torch.uint8)  # [B,1,H,W]

                    tp = (pred & mask).sum()
                    fp = (pred & (1 - mask)).sum()
                    fn = ((1 - pred) & mask).sum()
                    TP += tp
                    FP += fp
                    FN += fn

                    numerator += 2.0 * (pred.float() * mask.float()).sum()
                    denominator += (pred.float() + mask.float()).sum()

                    del mask, target, pred, res

        self.model.train()
        print(f"epoch {epoch} evaluation", numerator / denominator)
        print(f"epoch {epoch} TP={TP}, FP={FP}, FN={FN}")
        self.eval_metrics.append(
            (
                float((numerator / denominator).cpu().item()),
                int(TP.cpu().item()),
                int(FP.cpu().item()),
                int(FN.cpu().item()),
            )
        )
        gc.collect()

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        print(f"start position {self.start_positions}")
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")
            self.model.train()

            for dataset in get_tiff_images():
                try:
                    loader = data.DataLoader(
                        dataset,
                        batch_size=12,
                        pin_memory=True,
                        collate_fn=collate_fn,
                        num_workers=0,
                    )
                    for sample in loader:
                        self.optim.zero_grad(set_to_none=True)
                        if not sample:
                            continue
                        target, mask, idx = sample
                        mask = mask[:, :, :, :1] > 0

                        result = self.model.forward(
                            target.to(self.device, non_blocking=True)
                        ).to(self.device)

                        mask = (
                            mask.to(self.device, non_blocking=True)
                            .float()
                            .permute(0, 3, 1, 2)
                        )
                        loss_result = self.loss(result, mask)
                        loss_result.backward()
                        self.optim.step()

                        del mask, target, result, loss_result

                    print(f"data processed for {dataset.image}")
                finally:
                    del dataset, loader
                    if self.is_cuda_available:
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




## === cell 4
if torch.cuda.is_available():
    torch.cuda.empty_cache()




## === cell 5
def get_test_dataset():
    submission_file = "../input/hubmap-kidney-segmentation/sample_submission.csv"
    with open(submission_file) as sub_file:
        reader = csv.reader(sub_file)
        for image_id, _ in reader:
            if image_id == "id":
                continue
            print(f"openning {image_id}")
            dataset = HuBMAPDatasetPreprocessing(
                image_id,
                category="test",
                use_compressed=True,
                use_geometry_only=True,
            )
            yield dataset


def rle_encode_less_memory(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    Pixels are numbered top-to-bottom, then left-to-right (so transpose before flatten).
    """
    pixels = img.T.flatten()
    if pixels.size == 0:
        return ""
    pixels = pixels.astype(np.uint8, copy=False)
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _find_checkpoint():
    candidates = []
    candidates += glob.glob("/kaggle/working/models/model*.pkl")
    candidates += glob.glob("/kaggle/input/hubmapmodel/*.pkl")
    candidates += glob.glob("/kaggle/input/**/model*.pkl", recursive=True)

    def key_fn(p):
        base = os.path.basename(p)
        epoch = -1
        for token in (
            base.replace(".", "_").replace("(", "_").replace(")", "_").split("_")
        ):
            if token.isdigit():
                epoch = max(epoch, int(token))
        try:
            mtime = os.path.getmtime(p)
        except Exception:
            mtime = 0
        return (epoch, mtime)

    candidates = [p for p in candidates if os.path.isfile(p)]
    candidates.sort(key=key_fn, reverse=True)
    return candidates[0] if candidates else None


def make_submission(model, use_model=True):
    names, preds = [], []
    is_cuda_available = torch.cuda.is_available()
    dev = "cuda:0" if is_cuda_available else "cpu"

    if use_model:
        model.to(dev)
        model.eval()

    with torch.inference_mode():
        for ds in get_test_dataset():
            print(ds.image, len(ds))

            if not use_model:
                names.append(ds.image)
                preds.append("")
                del ds
                continue

            loader = data.DataLoader(
                ds, batch_size=32, collate_fn=collate_fn, pin_memory=True, num_workers=0
            )

            tile = ds.sz // ds.reduce  # 256 when reduce==1
            mask_tiles_np = np.zeros((len(ds), tile, tile), dtype=np.uint8)

            for batch_num, data_ in enumerate(loader):
                if not data_:
                    continue
                print(f"processing batch {batch_num}")
                image, _, idx = data_
                idx_np = np.asarray(idx, dtype=np.int64)
                image = image.to(dev, non_blocking=True)

                logits = model(image)  # [B,1,tile,tile]
                if ds.reduce != 1:
                    logits = torch.nn.functional.interpolate(
                        logits,
                        scale_factor=ds.reduce,
                        mode="bilinear",
                        align_corners=False,
                    )  # [B,1,ds.sz,ds.sz]

                pred = (logits > 0).to(torch.uint8)[:, 0]  # [B,H,W]

                if ds.reduce == 1:
                    mask_tiles_np[idx_np] = (
                        pred.cpu().numpy().astype(np.uint8, copy=False)
                    )
                else:
                    pred_small = (
                        torch.nn.functional.interpolate(
                            pred.unsqueeze(1).float(), size=(tile, tile), mode="nearest"
                        )[:, 0]
                        .to(torch.uint8)
                        .cpu()
                        .numpy()
                    )
                    mask_tiles_np[idx_np] = pred_small

                del pred, logits, image
                print(f"batch {batch_num} is done")

            mask_small = (
                mask_tiles_np.reshape(ds.n0max, ds.n1max, tile, tile)
                .transpose(0, 2, 1, 3)
                .reshape(ds.n0max * tile, ds.n1max * tile)
            )

            if ds.reduce != 1:
                mask_full = (
                    torch.nn.functional.interpolate(
                        torch.from_numpy(mask_small).unsqueeze(0).unsqueeze(0).float(),
                        scale_factor=ds.reduce,
                        mode="nearest",
                    )[0, 0]
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
            else:
                mask_full = mask_small

            full_h = ds.n0max * ds.sz
            full_w = ds.n1max * ds.sz
            mask_full = mask_full[:full_h, :full_w]

            mask_full = mask_full[
                ds.pad0 // 2 : -(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else full_h,
                ds.pad1 // 2 : -(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else full_w,
            ]

            rle = rle_encode_less_memory(mask_full.astype(np.uint8, copy=False))
            names.append(ds.image)
            preds.append(rle)

            del mask_tiles_np, mask_small, mask_full, ds
            if is_cuda_available:
                torch.cuda.empty_cache()
            gc.collect()

    df = pd.DataFrame({"img": names, "pixels": preds})
    df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with", len(df), "rows")
    print(df.head())


model = Model()

ckpt_path = _find_checkpoint()
use_model = True

if ckpt_path is None:
    print(
        "No checkpoint found under /kaggle/input or /kaggle/working. "
        "Skipping training (time safety) and creating an empty-mask submission."
    )
    use_model = False
else:
    print("Loading checkpoint:", ckpt_path)
    data_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    if isinstance(data_dict, dict) and "model_state_dict" in data_dict:
        model.load_state_dict(data_dict["model_state_dict"])
    else:
        model.load_state_dict(data_dict)

make_submission(model, use_model=use_model)



## === cell 6
"""
dataset = HuBMAPDatasetPreprocessing('0486052bb', 'train', apply_jitter=True, use_compressed=True)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(2, 10)
c = 0
for i in range(100):
    sample, mask, idx = dataset[i]
    sample = sample.permute(1,2,0)

    if idx == -1 or mask.sum() == 0:
        continue
    if c == 10:
        break
    plots[0][c].imshow(255. * sample.int().numpy())
    plots[0][c].set_title(f'Sample #{c}')
    mask = [np.array(m)*255 for m in mask]
    plots[1][c].imshow(mask)
    plots[1][c].set_title(f'Sample #{c}')
    c += 1
"""



## === cell 7
"""
dataset_test = HuBMAPDatasetPreprocessing('0486052bb', 'test', use_compressed=True)
import matplotlib.pyplot as plt
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
    print(sample)
    plots[0][c].imshow(sample)
    plots[0][c].set_title(f'Sample #{c}')
    c += 1
"""



## === cell 8
"""
from skimage import io
import matplotlib.pyplot as plt
image = io.imread('/kaggle/input/compressedhubmap/train/compressed-0486052bb.tiff')
"""



## === cell 9
"""
(from original notebook; kept commented)
"""



## === cell 10
"""
(from original notebook; kept commented)
"""
