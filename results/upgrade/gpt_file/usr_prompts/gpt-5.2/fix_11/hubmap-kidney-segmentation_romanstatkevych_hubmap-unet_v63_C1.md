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

0.9005890780432344

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import cv2
import csv
import glob
import json
import math
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision

from skimage import io, draw

mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])


def crop_center(img, cropx, cropy):
    y, x, z = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def _safe_read_tiff(path):
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    return img


def _load_polygons(json_path):
    data = json.load(open(json_path, "r"))
    polys = []
    for feat in data:
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        ring = coords[0]
        if ring and len(ring) >= 3:
            polys.append(ring)
    return polys


def _polygons_to_mask(polys, height, width, reduce=1):
    m = np.zeros((height, width), dtype=np.uint8)
    for ring in polys:
        xs = []
        ys = []
        for pt in ring:
            x, y = pt[0], pt[1]
            if reduce != 1:
                x = x / reduce
                y = y / reduce
            xs.append(x)
            ys.append(y)
        rr, cc = draw.polygon(np.array(ys), np.array(xs), shape=m.shape)
        m[rr, cc] = 1
    return m


class HuBMAPDatasetPreprocessing(Dataset):
    def __init__(
        self, image, category="train", use_compressed=False, apply_jitter=False
    ):
        self.mask_folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.folder = "/kaggle/input/hubmap-kidney-segmentation"
        self.use_compressed = use_compressed
        self.image = image
        self.category = category
        self.reduce = 4 if not use_compressed else 1
        self.apply_jitter = apply_jitter

        cand_paths = []
        if use_compressed:
            cand_paths += [
                os.path.join(
                    self.folder, self.category, f"compressed-{self.image}.tiff"
                ),
                os.path.join(
                    self.folder, self.category, f"compressed-{self.image}.tif"
                ),
            ]
        cand_paths += [
            os.path.join(self.folder, self.category, f"{self.image}.tiff"),
            os.path.join(self.folder, self.category, f"{self.image}.tif"),
        ]
        file_path = None
        for p in cand_paths:
            if os.path.exists(p):
                file_path = p
                break
        if file_path is None:
            raise FileNotFoundError(
                f"Could not find image file for id={self.image} in {self.folder}/{self.category}"
            )

        self._img = _safe_read_tiff(file_path)
        self.shape = self._img.shape[:2]  # (H, W)

        self.sz = 256 * self.reduce
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

        if self.category == "train":
            polys = _load_polygons(
                os.path.join(self.mask_folder, self.category, f"{self.image}.json")
            )
            self._mask_full = _polygons_to_mask(
                polys, self.shape[0], self.shape[1], reduce=1
            )
        else:
            self._mask_full = None

        self._skip_idx = None
        self._precompute_skip_index()

    def _copy_into_tile(self, tile, patch, r0, c0, r00, c00):
        """
        Bugfix: when the requested tile origin (r0,c0) is outside the image, patch is a clamped crop
        [r00:r01, c00:c01]. We must slice *patch* to match the visible portion inside the tile, not
        assume patch starts at the tile origin. The previous version could attempt to copy the entire
        image into a 256x256 tile, causing broadcast errors.
        """
        th, tw = tile.shape[:2]

        dst_r0 = int(max(0, r00 - r0))
        dst_c0 = int(max(0, c00 - c0))

        max_h = th - dst_r0
        max_w = tw - dst_c0
        if max_h <= 0 or max_w <= 0:
            return

        ph, pw = patch.shape[:2]
        copy_h = min(ph, max_h)
        copy_w = min(pw, max_w)
        if copy_h <= 0 or copy_w <= 0:
            return

        tile[dst_r0 : dst_r0 + copy_h, dst_c0 : dst_c0 + copy_w] = patch[
            0:copy_h, 0:copy_w
        ]

    def _precompute_skip_index(self):
        total = self.n0max * self.n1max
        skip = np.zeros(total, dtype=np.bool_)
        for idx in range(total):
            n0, n1 = idx // self.n1max, idx % self.n1max
            r0 = -self.pad0 // 2 + n0 * self.sz
            c0 = -self.pad1 // 2 + n1 * self.sz

            r00, r01 = max(0, r0), min(r0 + self.sz, self.shape[0])
            c00, c01 = max(0, c0), min(c0 + self.sz, self.shape[1])

            img = np.zeros((self.sz, self.sz, 3), np.uint8)
            patch = self._img[r00:r01, c00:c01]
            self._copy_into_tile(img, patch, r0, c0, r00, c00)

            if self.reduce != 1:
                img = cv2.resize(
                    img,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_AREA,
                )

            s = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
            skip[idx] = (int((s > 40).sum()) <= 1000) or (int(img.sum()) <= 1000)

        self._skip_idx = skip

    def close(self):
        return

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        if self._skip_idx is not None and self._skip_idx[idx]:
            if self.category == "train":
                dummy_mask = np.zeros(
                    (self.sz // self.reduce, self.sz // self.reduce, 1), np.uint8
                )
                img_min = np.zeros(
                    (self.sz // self.reduce, self.sz // self.reduce, 3), np.uint8
                )
                result_tensor = torch.from_numpy(img_min).permute(2, 0, 1).float()
                result_tensor = torchvision.transforms.functional.normalize(
                    result_tensor / 255.0, mean, std
                ).float()
                return result_tensor, dummy_mask.astype(bool), -1
            else:
                img_min = np.zeros(
                    (self.sz // self.reduce, self.sz // self.reduce, 3), np.uint8
                )
                result_tensor = torch.from_numpy(img_min).permute(2, 0, 1).float()
                result_tensor = torchvision.transforms.functional.normalize(
                    result_tensor / 255.0, mean, std
                ).float()
                return result_tensor, torch.zeros(1), -1

        n0, n1 = idx // self.n1max, idx % self.n1max
        r0 = -self.pad0 // 2 + n0 * self.sz
        c0 = -self.pad1 // 2 + n1 * self.sz

        r00, r01 = max(0, r0), min(r0 + self.sz, self.shape[0])
        c00, c01 = max(0, c0), min(c0 + self.sz, self.shape[1])

        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        patch = self._img[r00:r01, c00:c01]
        self._copy_into_tile(img, patch, r0, c0, r00, c00)

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        if self.category == "train":
            mask = np.zeros((self.sz, self.sz), np.uint8)
            mpatch = self._mask_full[r00:r01, c00:c01]
            self._copy_into_tile(mask, mpatch, r0, c0, r00, c00)
            if self.reduce != 1:
                mask = cv2.resize(
                    mask,
                    (self.sz // self.reduce, self.sz // self.reduce),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = mask[..., None]
        else:
            mask = torch.zeros(1)

        result_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        result_tensor = torchvision.transforms.functional.normalize(
            result_tensor / 255.0, mean, std
        ).float()

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
                m = mask.astype(np.uint8) * 255
                center = (m.shape[1] / 2.0, m.shape[0] / 2.0)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                m = cv2.warpAffine(
                    m,
                    M,
                    (m.shape[1], m.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderValue=0,
                )
                mask = (m > 0).astype(np.uint8)[..., None]

        if self.category == "train":
            mask = mask > 0
            return result_tensor, mask, idx
        else:
            return result_tensor, torch.zeros(1), idx




## === cell 1
import torch
from torch import nn, optim
from torch.utils import data
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




## === cell 2
OUTPUT_PATH = "/kaggle/working/models"


def collate_fn(batch):
    batch = [b for b in batch if b[-1] != -1]
    if len(batch) == 0:
        return None
    return data.dataloader.default_collate(batch)


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
    return numerator / (denominator + 1e-8)


class Trainer:
    def __init__(self):
        self.is_cuda_available = torch.cuda.is_available()
        dev = "cuda:0" if self.is_cuda_available else "cpu"
        print("Device", dev)
        self.device = torch.device(dev)
        self.model = Model().to(self.device)
        self.epochs = 70
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self._original_model_path = "../input/hubmapmodel/model (16).pkl"
        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)
        self.eval_metrics = []

        self.loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([5]).to(self.device)
        )
        if os.path.exists(self._original_model_path):
            dd = torch.load(self._original_model_path, map_location=self.device)
            self.start_positions = int(dd.get("epoch", 0))
            self.model.load_state_dict(dd["model_state_dict"])
        else:
            self.start_positions = 0

        self.num_workers = min(4, os.cpu_count() or 1)

    def _dl_kwargs(self):
        kw = dict(
            pin_memory=self.is_cuda_available,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
        )
        if self.num_workers > 0:
            kw["prefetch_factor"] = 2
        return kw

    def evaluate(self, epoch, display=False):
        self.model.eval()
        val_set = get_val_set()
        numerator, denominator = 0.0, 0.0
        TP, FP, FN = 0.0, 0.0, 0.0
        with torch.no_grad():
            for ds in val_set:
                print(f"started processing image {ds.image}")
                loader = data.DataLoader(ds, 9, **self._dl_kwargs())
                for sample in loader:
                    if sample is None:
                        continue
                    target, mask, idx = sample
                    target = target.to(self.device, non_blocking=True)
                    mask = mask.to(self.device, non_blocking=True)
                    mask = (mask[:, :, :, :1] > 0).squeeze(3)  # B,H,W

                    res = self.model(target)
                    result = (res > 0).squeeze(1)  # B,H,W

                    pred = result.int()
                    true = mask.int()
                    tp = pred * true
                    fp = pred * (1 - true)
                    fn = (1 - pred) * true
                    TP += float(tp.sum().item())
                    FP += float(fp.sum().item())
                    FN += float(fn.sum().item())

                    numerator += float(
                        (2.0 * (mask.float() * result.float()).sum()).item()
                    )
                    denominator += float((mask.float() + result.float()).sum().item())

        score = numerator / (denominator + 1e-8)
        print(f"epoch {epoch} evaluation", score)
        print(f"epoch {epoch} TP={TP}, FP={FP}, FN={FN}")
        self.eval_metrics.append((score, TP, FP, FN))
        self.model.train()

    def train(self):
        torch.autograd.set_detect_anomaly(False)
        print(f"start position {self.start_positions}")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")
            self.model.train()
            for dataset in get_tiff_images():
                loader = None
                try:
                    loader = data.DataLoader(dataset, 12, **self._dl_kwargs())
                    for sample in loader:
                        if sample is None:
                            continue
                        self.optim.zero_grad()
                        target, mask, idx = sample
                        mask = mask[:, :, :, :1] > 0
                        result = self.model(target.to(self.device, non_blocking=True))
                        mask = (
                            mask.to(self.device, non_blocking=True)
                            .float()
                            .permute(0, 3, 1, 2)
                        )
                        loss_result = self.loss(result, mask)
                        loss_result.backward()
                        self.optim.step()

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




## === cell 3
torch.cuda.empty_cache()
trainer = Trainer()




## === cell 4
def rle_encode_less_memory(img):
    """
    img: 2D array (H,W), 1-mask, 0-background
    Kaggle expects pixels numbered top-to-bottom then left-to-right => column-major flatten of (H,W),
    which is achieved by transposing then flattening.
    """
    img = img.astype(np.uint8)
    pixels = img.T.flatten()
    if pixels.size == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if runs.size == 0:
        return ""
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def get_test_dataset():
    submission_file = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
    with open(submission_file, newline="") as sub_file:
        reader = csv.reader(sub_file)
        for image_id, _ in reader:
            if image_id == "id":
                continue
            print(f"openning {image_id}")
            dataset = HuBMAPDatasetPreprocessing(
                image_id, category="test", use_compressed=False
            )
            yield dataset


def make_submission(model):
    names, preds = [], []
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    model.eval()

    num_workers = min(4, os.cpu_count() or 1)
    dl_kw = dict(
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        dl_kw["prefetch_factor"] = 2

    with torch.no_grad():
        for ds in get_test_dataset():
            print(ds.image, len(ds))
            loader = data.DataLoader(ds, 32, **dl_kw)

            tile_h = ds.sz // ds.reduce
            tile_w = ds.sz // ds.reduce
            mask_tiles = torch.zeros((len(ds), tile_h, tile_w), dtype=torch.uint8)

            for batch_num, data_ in enumerate(loader):
                if data_ is None:
                    continue
                print(f"processing batch {batch_num}")
                image, _, idx = data_
                image = image.to(dev, non_blocking=True)

                result = model(image).squeeze(1)  # B,h,w (tile resolution)
                for i, ndx in enumerate(idx):
                    mask_tiles[int(ndx)] = (result[i] > 0).to(torch.uint8).cpu()

                del result, image
                print(f"batch {batch_num} is done")

            bigmask_small = (
                (
                    mask_tiles.view(ds.n0max, ds.n1max, tile_h, tile_w)
                    .permute(0, 2, 1, 3)
                    .reshape(ds.n0max * tile_h, ds.n1max * tile_w)
                )
                .numpy()
                .astype(np.uint8)
            )

            if ds.reduce != 1:
                bigmask = cv2.resize(
                    bigmask_small,
                    (ds.n1max * ds.sz, ds.n0max * ds.sz),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                bigmask = bigmask_small

            bigmask = bigmask[
                ds.pad0
                // 2 : (-(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz),
                ds.pad1
                // 2 : (-(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz),
            ].astype(np.uint8)

            rle = rle_encode_less_memory(bigmask)
            names.append(ds.image)
            preds.append(rle)

            del mask_tiles, bigmask_small, bigmask, ds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    df = pd.DataFrame({"id": names, "predicted": preds})
    df.to_csv("/kaggle/working/submission.csv", index=False)
    print("Wrote /kaggle/working/submission.csv with shape:", df.shape)




## === cell 5
def _load_checkpoint_if_exists(model, ckpt_path):
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        return False
    dd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(dd, dict) and "model_state_dict" in dd:
        model.load_state_dict(dd["model_state_dict"], strict=True)
        return True
    if isinstance(dd, dict):
        model.load_state_dict(dd, strict=True)
        return True
    return False


model = Model()

working_ckpt = "/kaggle/working/models/model.pkl"
loaded = _load_checkpoint_if_exists(model, working_ckpt)

if not loaded:
    trainer.train()
    loaded = _load_checkpoint_if_exists(model, working_ckpt)

if not loaded:
    raise FileNotFoundError(f"Could not find or load checkpoint at {working_ckpt}")

make_submission(model)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1649447769.py in <cell line: 0>()
     18 
     19 if not loaded:
---> 20     trainer.train()
     21     loaded = _load_checkpoint_if_exists(model, working_ckpt)
     22 

/tmp/ipykernel_55/3943698487.py in train(self)
    120             print(f"Epoch {i}")
    121             self.model.train()
--> 122             for dataset in get_tiff_images():
    123                 loader = None
    124                 try:

/tmp/ipykernel_55/3943698487.py in get_tiff_images()
     18         name = os.path.splitext(bname)[0]
     19         if name not in VALSET:
---> 20             dataset = HuBMAPDatasetPreprocessing(
     21                 name, apply_jitter=False, use_compressed=True
     22             )

/tmp/ipykernel_55/2678661207.py in __init__(self, image, category, use_compressed, apply_jitter)
    125 
    126         self._skip_idx = None
--> 127         self._precompute_skip_index()
    128 
    129     def _copy_into_tile(self, tile, patch, r0, c0, r00, c00):

/tmp/ipykernel_55/2678661207.py in _precompute_skip_index(self)
    169             img = np.zeros((self.sz, self.sz, 3), np.uint8)
    170             patch = self._img[r00:r01, c00:c01]
--> 171             self._copy_into_tile(img, patch, r0, c0, r00, c00)
    172 
    173             if self.reduce != 1:

/tmp/ipykernel_55/2678661207.py in _copy_into_tile(self, tile, patch, r0, c0, r00, c00)
    152             return
    153 
--> 154         tile[dst_r0 : dst_r0 + copy_h, dst_c0 : dst_c0 + copy_w] = patch[
    155             0:copy_h, 0:copy_w
    156         ]

ValueError: could not broadcast input array from shape (30440,22240,3) into shape (1,1,3)
