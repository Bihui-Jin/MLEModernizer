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
import gc
import cv2
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io as skio

mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

_POLY_CACHE = (
    {}
)  # key: (category, image_id, use_compressed) -> list[np.ndarray float32 Nx2]
_MASK_CACHE = {}  # key: (image_id, use_compressed, h, w) -> np.ndarray bool (H,W)


class HuBMAPDatasetPreprocessing(Dataset):
    """
    Bugfixes for Kaggle runtime:
    - Avoid OpenCV full-image read of huge TIFFs (hits CV_IO_MAX_IMAGE_PIXELS).
      For category='test', load tiles lazily by reading regions from TIFF on-demand.
    - Keep core logic identical: same pad/tile/reduce, same normalization and HSV filter,
      same JSON polygon rasterization (train only).
    """

    def __init__(self, image, category="train", use_compressed=False):
        self.mask_folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.use_compressed = use_compressed
        self.image = str(image)
        self.category = category

        self.reduce = 4 if not use_compressed else 1
        self._img_full = None
        self._mask_full = None

        self._tiff = None
        self._tiff_pages = None
        self._tiff_path = None
        self._tiff_mm = None  # memmap or ndarray view for fast ROI slicing

        self._init_image_metadata()

        self._tile_rgb = np.zeros((self.sz, self.sz, 3), dtype=np.uint8)
        if self.category == "train":
            self._tile_mask3 = np.zeros((self.sz, self.sz, 3), dtype=np.uint8)

    def _get_image_path(self):
        if self.use_compressed:
            p = os.path.join(
                self.folder, self.category, f"compressed-{self.image}.tiff"
            )
            if os.path.exists(p):
                return p
            return os.path.join(self.folder, self.category, f"{self.image}.tiff")
        return os.path.join(self.folder, self.category, f"{self.image}.tiff")

    def _get_masks(self):
        file_path = os.path.join(self.mask_folder, self.category, f"{self.image}.json")
        with open(file_path, "r") as f:
            return json.load(f)

    def _read_geometry_coords(self):
        key = (self.category, self.image, bool(self.use_compressed))
        cached = _POLY_CACHE.get(key, None)
        if cached is not None:
            return cached

        feats = self._get_masks()
        polys = []
        for f in feats:
            geom = f.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates", [])
            if not coords:
                continue
            ring = coords[0]
            if len(ring) < 3:
                continue
            arr = np.asarray(ring, dtype=np.float32)  # [[x,y], ...]
            if self.use_compressed:
                arr = arr / 4.0
            polys.append(arr)

        _POLY_CACHE[key] = polys
        return polys

    def _rasterize_polygons(self, h, w):
        key = (self.image, bool(self.use_compressed), int(h), int(w))
        cached = _MASK_CACHE.get(key, None)
        if cached is not None:
            return cached

        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in self._read_geometry_coords():
            pts = np.round(poly).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            cv2.fillPoly(mask, [pts], 1)
        out = mask.astype(bool)
        _MASK_CACHE[key] = out
        return out

    def _read_tiff_rgb_uint8_full(self, path):
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.dtype != np.uint8:
                img_rgb = img_rgb.astype(np.uint8, copy=False)
            return img_rgb

        img = skio.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 3:
            img = img[:, :, :3]
        else:
            raise ValueError(f"Unexpected TIFF shape {img.shape} for {path}")

        if img.dtype != np.uint8:
            mx = float(np.max(img)) if np.max(img) > 0 else 1.0
            img = (img.astype(np.float32) / mx * 255.0).clip(0, 255).astype(np.uint8)
        return img

    def _open_tiff_for_tiled_reads(self, path):
        try:
            import tifffile
        except Exception as e:
            raise ImportError("tifffile is required for patch-based TIFF reads") from e

        self._tiff = tifffile.TiffFile(path)
        self._tiff_pages = self._tiff.pages[0]
        self._tiff_path = path

        try:
            self._tiff_mm = tifffile.memmap(path)
        except Exception:
            self._tiff_mm = None

        shape = self._tiff_pages.shape
        if len(shape) == 2:
            h, w = shape
        else:
            h, w = shape[0], shape[1]
        return (int(h), int(w))

    def _read_tile_rgb_uint8(self, p00, p01, p10, p11):
        """
        Read region [p00:p01, p10:p11] from TIFF without decoding the full image.
        Returns RGB uint8 array of shape (p01-p00, p11-p10, 3).
        """
        h = p01 - p00
        w = p11 - p10
        if h <= 0 or w <= 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        region = None

        if self._tiff_mm is not None:
            try:
                region = self._tiff_mm[p00:p01, p10:p11]
            except Exception:
                region = None

        if region is None and self._tiff_pages is not None:
            try:
                region = self._tiff_pages.asarray(
                    key=(slice(p00, p01), slice(p10, p11))
                )
            except Exception:
                region = None

        if region is None:
            img = skio.imread(self._tiff_path)
            region = img[p00:p01, p10:p11]

        if region.ndim == 2:
            region = np.stack([region, region, region], axis=-1)
        elif region.ndim == 3 and region.shape[2] >= 3:
            region = region[:, :, :3]
        else:
            raise ValueError(
                f"Unexpected region shape {region.shape} for {self._tiff_path}"
            )

        if region.dtype != np.uint8:
            mx = float(np.max(region)) if np.max(region) > 0 else 1.0
            region = (
                (region.astype(np.float32) / mx * 255.0).clip(0, 255).astype(np.uint8)
            )
        return region

    def _init_image_metadata(self):
        path = self._get_image_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found for id={self.image}: {path}")

        self.sz = 256 * self.reduce

        if self.category == "test":
            self._open_tiff_for_tiled_reads(path)
            self.shape = (
                int(self._tiff_pages.shape[0]),
                int(self._tiff_pages.shape[1]),
            )  # (H,W)
            self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
            self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
            self.n0max = (self.shape[0] + self.pad0) // self.sz
            self.n1max = (self.shape[1] + self.pad1) // self.sz
            return

        img = self._read_tiff_rgb_uint8_full(path)
        self._img_full = img
        self.shape = (img.shape[0], img.shape[1])  # (H,W)

        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        m = self._rasterize_polygons(self.shape[0], self.shape[1])
        self._mask_full = np.stack([m, m, m], axis=0)  # (3,H,W) bool

    def close(self):
        if self._tiff is not None:
            try:
                self._tiff.close()
            except Exception:
                pass
        self._tiff = None
        self._tiff_pages = None
        self._tiff_mm = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
        self._img_full = None
        self._mask_full = None

    def __len__(self):
        return int(self.n0max * self.n1max)

    def __getitem__(self, idx):
        n0, n1 = idx // self.n1max, idx % self.n1max

        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

        img = self._tile_rgb
        img.fill(0)

        if self.category == "test":
            region = self._read_tile_rgb_uint8(p00, p01, p10, p11)
            img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = region
        else:
            img_full = self._img_full
            img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = img_full[
                p00:p01, p10:p11
            ]

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        if self.category == "train":
            mask3 = self._tile_mask3
            mask3.fill(0)
            msrc = self._mask_full[:, p00:p01, p10:p11].astype(np.uint8, copy=False)
            mask3[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0), :] = np.moveaxis(
                msrc, 0, -1
            )

            if self.reduce != 1:
                mask = cv2.resize(
                    mask3,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                mask = mask3
        else:
            mask = torch.zeros(1)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]

        result_tensor = torch.from_numpy(
            ((img.astype(np.float32) / 255.0) - mean) / std
        ).float()
        result_tensor = result_tensor.permute(2, 0, 1).contiguous().float()

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

    plots[0][c].imshow(sample)
    plots[0][c].set_title(f'Sample #{c}')
    if isinstance(mask, np.ndarray):
        plots[1][c].imshow(mask[...,0])
    plots[1][c].set_title(f'Sample #{c}')
    c += 1
"""


## === cell 2
"""
# Visualization cell (kept commented)
"""


## === cell 3
import torch
from torch import nn, optim
from torch.utils import data
import torchvision
import torch.nn.functional as F


class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
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




## === cell 4
import os
import glob
import numpy
import torchvision
from torch.utils import data

OUTPUT_PATH = "/kaggle/working/models"

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def collate_fn(x):
    x = list(filter(lambda t: t[-1] != -1, x))
    if len(x) == 0:
        return []
    return data.dataloader.default_collate(x)


VALSET = ["e79de561c", "cb2d976f4"]

_TRAIN_DATASETS_CACHE = None
_VAL_DATASETS_CACHE = None


def get_tiff_images_list():
    global _TRAIN_DATASETS_CACHE
    if _TRAIN_DATASETS_CACHE is None:
        input_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
        ds_list = []
        for images in glob.glob(input_folder):
            bname = os.path.basename(images)
            name = os.path.splitext(bname)[0]
            if name not in VALSET:
                ds_list.append(
                    HuBMAPDatasetPreprocessing(
                        name, category="train", use_compressed=True
                    )
                )
        _TRAIN_DATASETS_CACHE = ds_list
    return _TRAIN_DATASETS_CACHE


def get_val_set_list():
    global _VAL_DATASETS_CACHE
    if _VAL_DATASETS_CACHE is None:
        _VAL_DATASETS_CACHE = [
            HuBMAPDatasetPreprocessing(image, category="train", use_compressed=True)
            for image in VALSET
        ]
    return _VAL_DATASETS_CACHE


def dice_loss(pred, target, device):
    numerator = 2 * torch.sum(pred.float() * target.float())
    denominator = torch.sum(pred.float() + target.float())
    return numerator / denominator


class MultiImageDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self._cum = np.cumsum([0] + [len(ds) for ds in self.datasets]).astype(np.int64)

    def __len__(self):
        return int(self._cum[-1])

    def __getitem__(self, idx):
        ds_i = int(np.searchsorted(self._cum, idx, side="right") - 1)
        local_idx = int(idx - self._cum[ds_i])
        return self.datasets[ds_i][local_idx]

    def close(self):
        for ds in self.datasets:
            try:
                ds.close()
            except Exception:
                pass


class Trainer:
    def __init__(self):
        self.is_cuda_available = torch.cuda.is_available()
        dev = "cuda:0" if self.is_cuda_available else "cpu"
        print("Device", dev)
        self.device = torch.device(dev)
        self.model = Model()
        self.model.to(self.device)
        self.epochs = 20
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self.optim = optim.Adam(self.model.parameters())

        if os.path.exists(self._model_path):
            self._data_dict = torch.load(self._model_path, map_location=self.device)
            self.start_positions = int(self._data_dict.get("epoch", 0)) + 1
            self.loss = self._data_dict["loss"]
            self.optim.load_state_dict(self._data_dict["optimizer_state_dict"])
            self.model.load_state_dict(self._data_dict["model_state_dict"])
        else:
            self.start_positions = 0
            self.loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor([5]).to(self.device)
            )

    def evaluate(self, epoch):
        val_mds = MultiImageDataset(get_val_set_list())
        num_workers = 2 if torch.cuda.is_available() else 0
        loader = data.DataLoader(
            val_mds,
            batch_size=16,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        numerator, denominator = 0, 0
        self.model.eval()
        with torch.no_grad():
            for sample in loader:
                if not sample:
                    continue
                target, mask, idx = sample
                target = target.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                mask = (mask[:, :, :, :1] > 0).permute(0, 3, 1, 2).float()

                res = self.model(target)
                result = (res > 0).float()

                numerator += 2.0 * torch.sum(mask * result).item()
                denominator += torch.sum(mask + result).item()

                del mask, target, result, res

        print(
            f"epoch {epoch} evaluation",
            (numerator / denominator) if denominator > 0 else 0.0,
        )
        self.model.train()
        val_mds.close()
        del loader, val_mds

    def train(self):
        print(f"start position {self.start_positions}")
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        train_mds = MultiImageDataset(get_tiff_images_list())
        num_workers = 2 if torch.cuda.is_available() else 0
        loader = data.DataLoader(
            train_mds,
            batch_size=16,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")
            self.model.train()

            for sample in loader:
                if not sample:
                    continue
                self.optim.zero_grad(set_to_none=True)

                target, mask, idx = sample
                mask = mask[:, :, :, :1] > 0

                target = target.to(self.device, non_blocking=True)
                result = self.model.forward(target).to(self.device)

                mask = (
                    mask.to(self.device, non_blocking=True).float().permute(0, 3, 1, 2)
                )
                loss_result = self.loss(result, mask)
                loss_result.backward()
                self.optim.step()

                del mask, target, result, loss_result

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

        train_mds.close()
        del loader, train_mds


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
from torch.utils import data


def collate_fn(x):
    x = list(filter(lambda t: t[-1] != -1, x))
    if len(x) == 0:
        return []
    return data.dataloader.default_collate(x)


def get_test_dataset():
    submission_file = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    sub = pd.read_csv(submission_file)
    for image_id in sub["id"].astype(str).tolist():
        print(f"openning {image_id}")
        dataset = HuBMAPDatasetPreprocessing(
            image_id, category="test", use_compressed=True
        )
        yield dataset


def rle_encode_less_memory(img):
    if img.ndim != 2:
        img = np.squeeze(img)
    img = img.astype(np.uint8, copy=False)
    if img.max() == 0:
        return ""
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def make_submission(model):
    names, preds = [], []
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    model.eval()

    with torch.no_grad():
        for ds in get_test_dataset():
            print(ds.image, len(ds))

            num_workers = 0
            loader = data.DataLoader(
                ds,
                batch_size=32,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers,
                persistent_workers=False,
            )

            tile_sz_full = ds.sz
            tile_sz_model = tile_sz_full // ds.reduce

            full_h_m = ds.n0max * tile_sz_model
            full_w_m = ds.n1max * tile_sz_model
            mask_full_m = torch.zeros((full_h_m, full_w_m), dtype=torch.uint8)

            try:
                for batch_num, batch in enumerate(loader):
                    if not batch:
                        continue
                    image, _, idx = batch
                    image = image.to(dev, non_blocking=True)

                    logits = model(image)  # (B,1,tile_sz_model,tile_sz_model)
                    prob = logits.squeeze(1)  # (B,tile_sz_model,tile_sz_model)

                    idx_cpu = idx.to(torch.int64).cpu()
                    tile_mask = (prob > 0).to(torch.uint8).cpu()  # (B,ts,ts)

                    n0 = torch.div(idx_cpu, ds.n1max, rounding_mode="floor")
                    n1 = idx_cpu - n0 * ds.n1max

                    for r in torch.unique(n0):
                        r = int(r.item())
                        cols = n1[n0 == r].to(torch.int64)
                        tiles = tile_mask[n0 == r]  # (K,ts,ts)
                        if tiles.numel() == 0:
                            continue
                        order = torch.argsort(cols)
                        cols = cols[order]
                        tiles = tiles[order]

                        cols_np = cols.numpy()
                        breaks = np.where(np.diff(cols_np) != 1)[0] + 1
                        groups = np.split(np.arange(len(cols_np)), breaks)
                        row_base = r * tile_sz_model
                        for g in groups:
                            c_start = int(cols_np[g[0]])
                            c_end = int(cols_np[g[-1]])
                            block = tiles[g].reshape(
                                -1, tile_sz_model
                            )  # (len(g)*ts, ts) flattened vertically
                            block2 = (
                                tiles[g].permute(1, 0, 2).reshape(tile_sz_model, -1)
                            )
                            col_base = c_start * tile_sz_model
                            mask_full_m[
                                row_base : row_base + tile_sz_model,
                                col_base : col_base + block2.shape[1],
                            ] = block2

                    del logits, prob, tile_mask, image, idx, idx_cpu, n0, n1

                pad0_m = ds.pad0 // ds.reduce
                pad1_m = ds.pad1 // ds.reduce

                r0c = pad0_m // 2
                r1c = full_h_m - (pad0_m - pad0_m // 2)
                c0c = pad1_m // 2
                c1c = full_w_m - (pad1_m - pad1_m // 2)

                mask_m = mask_full_m[r0c:r1c, c0c:c1c]

                mask_fullres = torch.nn.functional.interpolate(
                    mask_m[None, None].float(),
                    scale_factor=ds.reduce,
                    mode="nearest",
                )[0, 0].to(torch.uint8)

                rle = rle_encode_less_memory(mask_fullres.numpy())
            except Exception as e:
                print(
                    f"WARNING: failed on {ds.image} with error: {repr(e)}; using empty prediction."
                )
                rle = ""

            names.append(ds.image)
            preds.append(rle)

            ds.close()
            del mask_full_m, ds, loader
            gc.collect()

    df = pd.DataFrame({"id": names, "predicted": preds})
    df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", df.shape)


model = Model()

candidate_paths = [
    "/kaggle/working/models/model.pkl",
    "/kaggle/input/hubmapmodel/model (3).pkl",
]

ckpt_path = None
for p in candidate_paths:
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path is not None:
    data_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(data_dict, dict) and "model_state_dict" in data_dict:
        model.load_state_dict(data_dict["model_state_dict"])
        print("Loaded checkpoint:", ckpt_path)
    else:
        model.load_state_dict(data_dict)
        print("Loaded raw state_dict:", ckpt_path)
else:
    print(
        "No checkpoint found; using untrained model (submission will be low-scoring but valid)."
    )

make_submission(model)




## === cell 7
"""
# Debug/visualization cell kept commented
"""
