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

# 5. Code solution

## === cell 0
import os

try:
    os.system("rm -rf /kaggle/working/models/model.pkl")
except Exception:
    pass



## === cell 1
import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt
import glob
import json
import gc
from torch.utils.data import Dataset

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    cv2.setNumThreads(0)
except Exception:
    pass

TILE_CACHE_DIR = "/kaggle/working/tile_cache"
os.makedirs(TILE_CACHE_DIR, exist_ok=True)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _poly_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def rle_decode(mask_rle: str, shape):
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if mask_rle is None:
        return mask.reshape((w, h)).T
    s = str(mask_rle).strip()
    if s == "" or s.lower() == "nan":
        return mask.reshape((w, h)).T
    parts = s.split()
    starts = np.asarray(parts[0::2], dtype=np.int64) - 1
    lengths = np.asarray(parts[1::2], dtype=np.int64)
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        if lo < 0:
            lo = 0
        if hi > mask.size:
            hi = mask.size
        mask[lo:hi] = 1
    return mask.reshape((w, h)).T


def _safe_imread_anydepth(path: str) -> np.ndarray:
    img = None
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    if img is None:
        try:
            import tifffile

            img = tifffile.imread(path)
        except Exception:
            img = None

    if img is None:
        try:
            from PIL import Image

            with Image.open(path) as im:
                img = np.array(im)
        except Exception:
            img = None

    if img is None:
        raise FileNotFoundError(f"Could not read image (decoder failure): {path}")

    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.ndim == 3:
        if (
            img.shape[0] in (1, 3)
            and img.shape[2] not in (1, 3)
            and img.shape[0] < img.shape[2]
        ):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
    else:
        img = np.squeeze(img)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

    if img.dtype != np.uint8:
        img_min = float(np.nanmin(img))
        img_max = float(np.nanmax(img))
        if not np.isfinite(img_min) or not np.isfinite(img_max) or img_max <= img_min:
            img = np.zeros_like(img, dtype=np.uint8)
        else:
            img = (
                ((img.astype(np.float32) - img_min) * (255.0 / (img_max - img_min)))
                .clip(0, 255)
                .astype(np.uint8)
            )

    return img


def _saturation_from_bgr_uint8(img_bgr: np.ndarray) -> np.ndarray:
    mx = img_bgr.max(axis=2).astype(np.float32)
    mn = img_bgr.min(axis=2).astype(np.float32)
    s = np.zeros_like(mx, dtype=np.float32)
    nz = mx > 0
    s[nz] = (mx[nz] - mn[nz]) * (255.0 / mx[nz])
    return s  # float32 0..255


def _atomic_npy_save(path: str, arr):
    tmp = f"{path}.tmp_{os.getpid()}_{np.random.randint(0, 1_000_000)}"
    np.save(tmp, arr)
    if not tmp.endswith(".npy"):
        tmp_npy = tmp + ".npy"
    else:
        tmp_npy = tmp
    os.replace(tmp_npy, path)


def _safe_npy_load(path: str, mmap_mode=None):
    return np.load(path, mmap_mode=mmap_mode)


class HuBMAPDatasetPreprocessing(Dataset):
    """
    Reads TIFF and returns padded tiles.
    For train, builds a binary mask from train.csv RLE (more reliable than JSON polygons).
    """

    def __init__(
        self, image, category="train", use_compressed=False, train_rle_map=None
    ):
        self.mask_folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.use_compressed = use_compressed
        self.image = image
        self.category = category
        self.reduce = 4 if not use_compressed else 1  # preserve original behavior

        self._img = None  # HWC uint8
        self.mask = None  # HW uint8 {0,1}
        self.train_rle_map = train_rle_map or {}

        self._cached_tiles = None
        self._cached_masks = None
        self._valid_idx = None

        self._valid_bool = None

        self._get_image()
        if self.category != "train":
            self._build_or_load_valid_cache_for_inference()

    def _get_masks(self):
        file_path = os.path.join(self.mask_folder, self.category, f"{self.image}.json")
        return _load_json(file_path)

    def _read_polygons(self):
        feats = self._get_masks()
        polygons = []
        for f in feats:
            try:
                coords = f["geometry"]["coordinates"]
                poly = coords[0]
                if self.use_compressed:
                    poly = [[c[0] // 4, c[1] // 4] for c in poly]
                polygons.append(poly)
            except Exception:
                continue
        return polygons

    def _get_image(self):
        if self._img is not None:
            return self._img

        candidate_paths = []
        if self.use_compressed:
            candidate_paths.append(
                os.path.join(
                    self.folder, self.category, f"compressed-{self.image}.tiff"
                )
            )
        candidate_paths.append(
            os.path.join(self.folder, self.category, f"{self.image}.tiff")
        )

        file_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                file_path = p
                break
        if file_path is None:
            raise FileNotFoundError(
                f"Could not find image for {self.image}. Tried: {candidate_paths}"
            )

        img = _safe_imread_anydepth(file_path)

        self._img = img
        self.shape = self._img.shape[:2]  # (H, W)

        self.sz = 256 * self.reduce
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        if self.category == "train":
            if self.image in self.train_rle_map:
                self.mask = rle_decode(self.train_rle_map[self.image], self.shape)
            else:
                polygons = self._read_polygons()
                self.mask = _poly_to_mask(polygons, self.shape[0], self.shape[1])

        return self._img

    def _cache_paths(self):
        tag = (
            f"{self.category}_{self.image}_uc{int(self.use_compressed)}_r{self.reduce}"
        )
        return (
            os.path.join(TILE_CACHE_DIR, f"{tag}_tiles.npy"),
            os.path.join(TILE_CACHE_DIR, f"{tag}_masks.npy"),
            os.path.join(TILE_CACHE_DIR, f"{tag}_valid.npy"),
        )

    def _build_or_load_tile_cache(self):
        if self._cached_tiles is not None:
            return

        if self.category != "train":
            return

        tiles_path, masks_path, valid_path = self._cache_paths()
        tile_hw = self.sz // self.reduce

        if (
            os.path.exists(tiles_path)
            and os.path.exists(masks_path)
            and os.path.exists(valid_path)
        ):
            try:
                tiles_np = _safe_npy_load(tiles_path, mmap_mode="r")
                masks_np = _safe_npy_load(masks_path, mmap_mode="r")
                valid_np = _safe_npy_load(valid_path, mmap_mode=None).astype(
                    np.int32, copy=False
                )

                if tiles_np.ndim != 4 or masks_np.ndim != 4:
                    raise ValueError("cache has wrong ndim")
                if tiles_np.shape[0] != masks_np.shape[0]:
                    raise ValueError("cache tiles/masks length mismatch")

                self._cached_tiles = tiles_np
                self._cached_masks = masks_np
                self._valid_idx = valid_np
                self._valid_bool = np.zeros((len(tiles_np),), dtype=np.bool_)
                if valid_np.size:
                    self._valid_bool[valid_np] = True
                return
            except Exception:
                for p in (tiles_path, masks_path, valid_path):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                self._cached_tiles = None
                self._cached_masks = None
                self._valid_idx = None
                self._valid_bool = None

        N = self.__len__()
        tiles_np = np.empty((N, 3, tile_hw, tile_hw), dtype=np.float32)
        masks_np = np.empty((N, tile_hw, tile_hw, 3), dtype=np.uint8)
        valid = []

        img_full = self._get_image()
        for idx in range(N):
            n0, n1 = idx // self.n1max, idx % self.n1max
            x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
            p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
            p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

            img = np.zeros((self.sz, self.sz, 3), np.uint8)
            patch = img_full[p00:p01, p10:p11]
            img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = patch

            if self.reduce != 1:
                img = cv2.resize(img, (tile_hw, tile_hw), interpolation=cv2.INTER_AREA)

            mask = np.zeros((self.sz, self.sz), np.uint8)
            mpatch = self.mask[p00:p01, p10:p11]
            mask[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = mpatch.astype(
                np.uint8
            )
            if self.reduce != 1:
                mask = cv2.resize(
                    mask, (tile_hw, tile_hw), interpolation=cv2.INTER_NEAREST
                )
            mask3 = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint8)

            s = _saturation_from_bgr_uint8(img)
            if (s > 40).sum() > 1000 and img.sum() > 1000:
                valid.append(idx)

            tiles_np[idx] = ((img.astype(np.float32) / 255.0 - mean) / std).transpose(
                2, 0, 1
            )
            masks_np[idx] = mask3

        valid_np = np.asarray(valid, dtype=np.int32)

        _atomic_npy_save(tiles_path, tiles_np)
        _atomic_npy_save(masks_path, masks_np)
        _atomic_npy_save(valid_path, valid_np)

        self._cached_tiles = _safe_npy_load(tiles_path, mmap_mode="r")
        self._cached_masks = _safe_npy_load(masks_path, mmap_mode="r")
        self._valid_idx = valid_np
        self._valid_bool = np.zeros((N,), dtype=np.bool_)
        if valid_np.size:
            self._valid_bool[valid_np] = True

    def _valid_cache_paths_inference(self):
        tag = (
            f"{self.category}_{self.image}_uc{int(self.use_compressed)}_r{self.reduce}"
        )
        return os.path.join(TILE_CACHE_DIR, f"{tag}_valid_only.npy")

    def _build_or_load_valid_cache_for_inference(self):
        if self._valid_bool is not None:
            return
        valid_path = self._valid_cache_paths_inference()
        N = self.__len__()
        if os.path.exists(valid_path):
            try:
                valid_np = _safe_npy_load(valid_path).astype(np.int32, copy=False)
                vb = np.zeros((N,), dtype=np.bool_)
                if valid_np.size:
                    vb[valid_np] = True
                self._valid_idx = valid_np
                self._valid_bool = vb
                return
            except Exception:
                try:
                    os.remove(valid_path)
                except Exception:
                    pass

        img_full = self._get_image()
        tile_hw = self.sz // self.reduce
        valid = []
        for idx in range(N):
            n0, n1 = idx // self.n1max, idx % self.n1max
            x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
            p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
            p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

            img = np.zeros((self.sz, self.sz, 3), np.uint8)
            patch = img_full[p00:p01, p10:p11]
            img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = patch

            if self.reduce != 1:
                img = cv2.resize(img, (tile_hw, tile_hw), interpolation=cv2.INTER_AREA)

            s = _saturation_from_bgr_uint8(img)
            if (s > 40).sum() > 1000 and img.sum() > 1000:
                valid.append(idx)

        valid_np = np.asarray(valid, dtype=np.int32)
        _atomic_npy_save(valid_path, valid_np)
        vb = np.zeros((N,), dtype=np.bool_)
        if valid_np.size:
            vb[valid_np] = True
        self._valid_idx = valid_np
        self._valid_bool = vb

    def close(self):
        return

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        if self.category == "train":
            self._build_or_load_tile_cache()
            is_valid = False
            if getattr(self, "_valid_bool", None) is not None:
                is_valid = bool(self._valid_bool[idx])

            result_tensor = torch.from_numpy(
                np.array(self._cached_tiles[idx], copy=False)
            ).float()
            mask = np.array(self._cached_masks[idx], copy=False)

            if not is_valid:
                return result_tensor, mask, -1
            return result_tensor, mask, idx

        img_full = self._get_image()
        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        patch = img_full[p00:p01, p10:p11]
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = patch

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        mask = torch.zeros(1)

        if self._valid_bool is not None:
            if not bool(self._valid_bool[idx]):
                result_tensor = torch.from_numpy(
                    ((img.astype(np.float32) / 255.0 - mean) / std)
                ).float()
                result_tensor = result_tensor.permute(2, 0, 1).float()
                return result_tensor, mask, -1
        else:
            s = _saturation_from_bgr_uint8(img)
            if (s > 40).sum() <= 1000 or img.sum() <= 1000:
                result_tensor = torch.from_numpy(
                    ((img.astype(np.float32) / 255.0 - mean) / std)
                ).float()
                result_tensor = result_tensor.permute(2, 0, 1).float()
                return result_tensor, mask, -1

        result_tensor = torch.from_numpy(
            ((img.astype(np.float32) / 255.0 - mean) / std)
        ).float()
        result_tensor = result_tensor.permute(2, 0, 1).float()
        return result_tensor, mask, idx




## === cell 2
"""
dataset = HuBMAPDatasetPreprocessing('0486052bb', use_compressed=True)

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

    plots[0][c].imshow(sample)
    plots[0][c].set_title(f'Sample #{c}')
    plots[1][c].imshow(mask[:,:,0]*255, cmap='gray')
    plots[1][c].set_title(f'Mask #{c}')
    c += 1
"""



## === cell 3
"""
fig = plt.figure(figsize=(20, 12))
f, plots = plt.subplots(2, 5)
c = 0
for i in range(100):
    sample, mask, idx = dataset_test[i]
    if idx == -1:
        continue
    if c == 5:
        break
    plots[0][c].imshow(sample)
    plots[0][c].set_title(f'Sample #{c}')
    c += 1
"""



## === cell 4
import torch
from torch import nn, optim
from torch.utils import data
import torchvision
import torchvision.transforms.functional as F_tv  # keep import but avoid clobbering torch.nn.functional
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
        self.image_size = (128, 128)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1

        self.down3 = Down(256, 512 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, X):
        x1 = self.inc(X)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits




## === cell 5
import glob
import os
import numpy
import gc
import pandas as pd

OUTPUT_PATH = "/kaggle/working/models"


def collate_fn(x):
    ls = [t for t in x if t[-1] != -1]
    if ls:
        return data.dataloader.default_collate(ls)
    else:
        return []


VALSET = ["e79de561c", "cb2d976f4"]

TRAIN_CSV_PATH = "/kaggle/input/hubmap-kidney-segmentation/train.csv"
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_id_col = train_df.columns[0]
train_rle_col = train_df.columns[1]
TRAIN_RLE_MAP = dict(
    zip(
        train_df[train_id_col].astype(str).tolist(),
        train_df[train_rle_col].astype(str).tolist(),
    )
)


def get_tiff_images():
    input_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
    for images in glob.glob(input_folder):
        bname = os.path.basename(images)
        name = os.path.splitext(bname)[0]
        if name not in VALSET:
            dataset = HuBMAPDatasetPreprocessing(
                os.path.splitext(bname)[0],
                use_compressed=True,
                train_rle_map=TRAIN_RLE_MAP,
            )
            yield dataset


def get_val_set():
    for image in VALSET:
        dataset = HuBMAPDatasetPreprocessing(
            image, use_compressed=True, train_rle_map=TRAIN_RLE_MAP
        )
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
        self.epochs = 70
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self._original_model_path = "../input/hubmapmodel/model (5).pkl"
        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)

        if os.path.exists(self._original_model_path):
            if not self.is_cuda_available:
                self._data_dict = torch.load(
                    self._original_model_path, map_location=torch.device("cpu")
                )
            else:
                self._data_dict = torch.load(self._original_model_path)
            self.start_positions = self._data_dict["epoch"]
            self.loss = self._data_dict["loss"]
            self.optim.load_state_dict(self._data_dict["optimizer_state_dict"])
            self.model.load_state_dict(self._data_dict["model_state_dict"])
        else:
            self.start_positions = 0
            self.loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor([5]).to(self.device)
            )

    def evaluate(self, epoch):
        val_set = get_val_set()
        numerator, denominator = 0, 0
        for ds in val_set:
            loader = data.DataLoader(
                ds, 8, pin_memory=True, collate_fn=collate_fn, num_workers=0
            )
            for _, sample in enumerate(loader):
                if sample == []:
                    continue
                target, mask, _ = sample
                target = target.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                mask = (mask[:, :, :, :1] > 0).squeeze()
                res = self.predict(target)
                result = (res > 0).squeeze()

                numerator += 2 * torch.sum(mask.float() * result.float()).item()
                denominator += torch.sum(mask.float() + result.float()).item()

                del mask, target, result, res

        if denominator == 0:
            print(f"epoch {epoch} evaluation: denominator=0 (skipped)")
        else:
            print(f"epoch {epoch} evaluation", numerator / denominator)
        del ds
        gc.collect()

    def predict(self, image):
        logits = self.model(image)
        return logits

    def train(self):
        torch.autograd.set_detect_anomaly(False)

        print(f"start position {self.start_positions}")
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")
            for dataset in get_tiff_images():
                try:
                    loader = data.DataLoader(
                        dataset,
                        12,
                        pin_memory=True,
                        collate_fn=collate_fn,
                        num_workers=0,
                    )
                    for step, sample in enumerate(loader):
                        if sample == []:
                            continue
                        self.optim.zero_grad(set_to_none=True)

                        target, mask, _ = sample
                        mask = mask[:, :, :, :1] > 0

                        target = target.to(self.device, non_blocking=True)
                        mask = (
                            mask.to(self.device, non_blocking=True)
                            .float()
                            .permute(0, 3, 1, 2)
                        )

                        result = self.model.forward(target)
                        loss_result = self.loss(result, mask)
                        loss_result.backward()
                        self.optim.step()

                        del mask, target, result, loss_result

                finally:
                    try:
                        dataset.close()
                    except Exception:
                        pass
                    del dataset, loader
                    if self.is_cuda_available:
                        torch.cuda.empty_cache()
                    gc.collect()

            self.evaluate(i)

            print(f"saving epoch {i}")
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                    "loss": self.loss,
                },
                os.path.join(OUTPUT_PATH, f"model_{i}.pkl"),
            )
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                    "loss": self.loss,
                },
                self._model_path,
            )


def display_predictions(masks):
    masks = masks.squeeze()
    masks = masks > 0.5
    mask = 255 * masks
    plt.imshow(mask)




## === cell 6
torch.cuda.empty_cache()
trainer = Trainer()



## === cell 7
import pandas as pd
import csv


def get_test_dataset():
    submission_file = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    sub = pd.read_csv(submission_file)
    for image_id in sub["id"].tolist():
        dataset = HuBMAPDatasetPreprocessing(
            image_id, category="test", use_compressed=True
        )
        yield dataset


def rle_encode_less_memory(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = img.astype(np.uint8, copy=False)

    pixels = img.T.reshape(-1)  # column-major numbering
    if pixels.size == 0:
        return ""

    padded = np.empty(pixels.size + 2, dtype=np.uint8)
    padded[0] = 0
    padded[1:-1] = pixels
    padded[-1] = 0

    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes.tolist()))


def _find_checkpoint_path():
    p_pre = "../input/hubmapmodel/model (5).pkl"
    if os.path.exists(p_pre):
        return p_pre

    p_main = "/kaggle/working/models/model.pkl"
    if os.path.exists(p_main):
        return p_main
    candidates = sorted(glob.glob("/kaggle/working/models/model_*.pkl"))
    if candidates:
        return candidates[-1]
    return None


def make_submission(model):
    names, preds = [], []
    is_cuda_available = torch.cuda.is_available()
    dev = "cuda:0" if is_cuda_available else "cpu"
    model.to(dev)
    model.eval()

    num_workers = 2

    with torch.no_grad():
        for ds in get_test_dataset():
            loader = data.DataLoader(
                ds,
                32,
                collate_fn=collate_fn,
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
                pin_memory=True,
                prefetch_factor=2 if num_workers > 0 else None,
            )

            tile_hw = ds.sz // ds.reduce
            tile_mask = torch.zeros((len(ds), tile_hw, tile_hw), dtype=torch.uint8)

            for data_ in loader:
                if data_ == []:
                    continue
                image, _, idx = data_
                image = image.to(dev, non_blocking=True)

                result = torch.nn.functional.interpolate(
                    model(image),
                    scale_factor=ds.reduce,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                idx_np = idx.detach().cpu().numpy().astype(np.int64, copy=False)
                tile_mask[idx_np] = (result.detach().cpu() > 0).to(torch.uint8)

                del image, result

            full = (
                tile_mask.view(ds.n0max, ds.n1max, tile_hw, tile_hw)
                .permute(0, 2, 1, 3)
                .reshape(ds.n0max * tile_hw, ds.n1max * tile_hw)
            )

            pad0_r = ds.pad0 // ds.reduce
            pad1_r = ds.pad1 // ds.reduce
            top = pad0_r // 2
            left = pad1_r // 2
            bottom = full.shape[0] - (pad0_r - pad0_r // 2)
            right = full.shape[1] - (pad1_r - pad1_r // 2)
            full = full[top:bottom, left:right]

            rle = rle_encode_less_memory(full)
            names.append(ds.image)
            preds.append(rle)

            try:
                ds.close()
            except Exception:
                pass
            del tile_mask, full, ds
            gc.collect()

    df = pd.DataFrame({"id": names, "predicted": preds})
    df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with", len(df), "rows")


ckpt = _find_checkpoint_path()
if ckpt is None:
    print(
        "No checkpoint found. Training to create /kaggle/working/models/model.pkl ..."
    )
    trainer.train()
    ckpt = _find_checkpoint_path()

model = Model()
if ckpt is not None:
    print("Loading checkpoint:", ckpt)
    data_dict = torch.load(ckpt, map_location=torch.device("cpu"))
    if isinstance(data_dict, dict) and "model_state_dict" in data_dict:
        model.load_state_dict(data_dict["model_state_dict"], strict=True)
    else:
        model.load_state_dict(data_dict, strict=True)
else:
    print(
        "WARNING: No checkpoint found after training attempt. Using randomly initialized model weights."
    )

make_submission(model)



## === cell 8
"""
(from skimage import io
import matplotlib.pyplot as plt
subimage = image[500:1500,2500:3500]
f,subplots = plt.subplots(1, 3)
subplots[0].imshow(subimage)
activation = {} # dictionary to store the activation of a layer
torch.cuda.empty_cache()

def create_hook(name):
    def hook(m, i, o):
        activation[name] = o
    return hook

trainer = Trainer()
input = torch.Tensor(subimage).permute(2,0,1).unsqueeze(0)
prediction = trainer.predict(input)

pred = prediction.detach().cpu().squeeze().permute(1,2,0)
subplots[1].imshow(pred)
pred_2 = trainer.model(input).detach().cpu().squeeze()
counter = 0
print(pred_2)
subplots[2].imshow(pred_2 > 0)
)
"""
