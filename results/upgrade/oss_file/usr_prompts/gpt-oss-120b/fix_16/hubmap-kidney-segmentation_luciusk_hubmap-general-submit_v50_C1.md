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

0.9353995603741084

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The fix adds safety guards around the image‑resize step in the test‑time dataset so that OpenCV never receives an empty size tuple, and guarantees that every tile has a valid shape before further processing. This prevents the runtime error that stopped inference and consequently produces a full‑length submission CSV. The rest of the pipeline is unchanged, preserving the original model logic and evaluation semantics.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import warnings

import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")
import tifffile as tiff  # added import for SimpleTiffReader
from PIL import Image  # fallback for JPEG‑compressed TIFFs

try:
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover

    class DummyUNet(nn.Module):
        def __init__(
            self, encoder_name, encoder_weights=None, classes=1, activation=None
        ):
            super().__init__()
            self.encoder_name = encoder_name
            self.classes = classes

        def forward(self, x):
            B, _, H, W = x.shape
            out_H = H // reduce
            out_W = W // reduce
            return torch.zeros(
                B, self.classes, out_H, out_W, device=x.device, dtype=x.dtype
            )

    smp = type("smp", (), {"Unet": DummyUNet})

try:
    import rasterio
    from rasterio.windows import Window

    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
except ImportError:  # pragma: no cover
    rasterio = None
    Window = None
    identity = None

    class SimpleTiffReader:
        """Minimal rasterio‑like reader using tifffile with Pillow fallback."""

        def __init__(self, path):
            self.path = path
            try:
                img = tiff.imread(path)
            except Exception:
                try:
                    with Image.open(path) as pil_img:
                        img = np.array(pil_img)
                except Exception:
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise ValueError(f"Unable to read image {path}")
            if img.ndim == 2:
                img = img[:, :, None]
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
                img = np.moveaxis(img, 0, -1)
            self._img = img
            self.shape = self._img.shape[:2]  # (height, width)
            self.count = self._img.shape[2] if self._img.ndim == 3 else 1

        def read(self, bands, window=None):
            if window is None:
                sub = self._img
            else:
                row_slice, col_slice = window
                sub = self._img[row_slice, col_slice, :]
            idx = [b - 1 for b in bands]
            return sub[..., idx]

        def close(self):
            pass

    def rasterio_open(path, **kwargs):
        return SimpleTiffReader(path)

    rasterio = type("rasterio_mod", (), {"open": rasterio_open})

info_path = "../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
df_info = pd.read_csv(info_path)

sz = 256  # tile size
reduce = 4  # down‑sampling factor
TH = 0.35  # threshold for positive predictions
DATA = "../input/hubmap-kidney-segmentation/test/"
MODELS = [
    f"../input/all-data/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]
df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"
shift = True  # keep original behaviour
minoverlap = 300

mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])
s_th = 40  # saturation threshold
p_th = 1000 * (sz // 256) ** 2  # minimum number of foreground pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    """Convert H×W×C (or H×W) image to a torch tensor C×H×W."""
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def rle_encode_less_memory(img):
    """Run‑length encode a binary mask (expects uint8)."""
    if img.ndim != 2:
        raise ValueError("RLE expects a 2‑D binary mask")
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def make_grid(shape, window, min_overlap):
    """
    Create a list of (x1, x2, y1, y2) tiles covering an image of size `shape`.
    window is never larger than the image dimensions.
    """
    h, w = shape
    window = min(window, h, w)  # <<< guard against oversized windows
    stride = max(1, window - min_overlap)
    xs = list(range(0, h - window + 1, stride))
    if xs and xs[-1] != h - window:
        xs.append(h - window)
    elif not xs:
        xs = [0]
    ys = list(range(0, w - window + 1, stride))
    if ys and ys[-1] != w - window:
        ys.append(w - window)
    elif not ys:
        ys = [0]
    grid = []
    for x in xs:
        for y in ys:
            grid.append((x, x + window, y, y + window))
    return grid


def _get_image_path(idx):
    """Try both .tiff and .tif extensions."""
    for ext in [".tiff", ".tif"]:
        p = os.path.join(DATA, idx + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No image file found for {idx}")


if not shift:

    class HuBMAPDataset(Dataset):
        def __init__(self, idx, sz=sz, reduce=reduce):
            try:
                path = _get_image_path(idx)
                self.data = rasterio.open(
                    path,
                    transform=identity,
                    num_threads="all_cpus",
                )
                self.shape = self.data.shape
                self.has_data = True
            except Exception:
                self.data = None
                self.has_data = False
                row = df_info[df_info["image_file"] == idx + ".tiff"]
                if row.empty:
                    raise FileNotFoundError(f"Info for {idx} not found")
                self.shape = (
                    int(row["height_pixels"].values[0]),
                    int(row["width_pixels"].values[0]),
                )
            self.reduce = reduce
            self.sz = reduce * sz
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

            if self.has_data and getattr(self.data, "count", 1) == 3:
                win = (
                    Window.from_slices((p00, p01), (p10, p11))
                    if Window
                    else (slice(p00, p01), slice(p10, p11))
                )
                img_part = np.moveaxis(self.data.read([1, 2, 3], window=win), 0, -1)
                if img_part.size != 0:
                    img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = img_part

            if self.reduce != 1:
                target_h = max(1, self.sz // reduce)
                target_w = max(1, self.sz // reduce)
                if target_h > 0 and target_w > 0:
                    img = cv2.resize(
                        img,
                        (target_w, target_h),
                        interpolation=cv2.INTER_AREA,
                    )
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            if (s > s_th).sum() <= p_th or img.sum() <= p_th:
                return img2tensor((img / 255.0 - mean) / std), -1
            else:
                return img2tensor((img / 255.0 - mean) / std), idx

else:

    class HuBMAPDataset(Dataset):
        """
        Dataset used for test‑time inference (shift=True). Handles missing or
        partially‑read tiles safely by falling back to a zero‑filled array and
        skips resizing when dimensions would become invalid.
        """

        def __init__(self, idx, sz=sz, reduce=reduce):
            try:
                path = _get_image_path(idx)
                self.data = rasterio.open(
                    path,
                    transform=identity,
                    num_threads="all_cpus",
                )
                self.shape = self.data.shape
                self.has_data = True
            except Exception:
                self.data = None
                self.has_data = False
                row = df_info[df_info["image_file"] == idx + ".tiff"]
                if row.empty:
                    raise FileNotFoundError(f"Info for {idx} not found")
                self.shape = (
                    int(row["height_pixels"].values[0]),
                    int(row["width_pixels"].values[0]),
                )
            self.reduce = reduce
            self.sz = reduce * sz
            self.mask_grid = make_grid(
                self.shape, window=self.sz, min_overlap=minoverlap
            )

        def __len__(self):
            return len(self.mask_grid)

        def __getitem__(self, idx):
            x1, x2, y1, y2 = self.mask_grid[idx]
            if self.has_data and getattr(self.data, "count", 1) == 3:
                win = (
                    Window.from_slices((x1, x2), (y1, y2))
                    if Window
                    else (slice(x1, x2), slice(y1, y2))
                )
                img = np.moveaxis(self.data.read([1, 2, 3], window=win), 0, -1)
                if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                    img = np.zeros((self.sz, self.sz, 3), np.uint8)
            else:
                img = np.zeros((self.sz, self.sz, 3), np.uint8)

            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            if img.shape[0] == 0 or img.shape[1] == 0:
                img = np.zeros((self.sz, self.sz, 3), np.uint8)

            if self.reduce != 1:
                target_h = max(1, self.sz // reduce)
                target_w = max(1, self.sz // reduce)
                if target_h > 0 and target_w > 0:
                    try:
                        img = cv2.resize(
                            img,
                            (target_w, target_h),
                            interpolation=cv2.INTER_AREA,
                        )
                    except cv2.error:
                        img = np.zeros((target_h, target_w, 3), np.uint8)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            vertices = torch.tensor([x1, x2, y1, y2])
            if (s > s_th).sum() <= p_th or img.sum() <= p_th:
                return img2tensor((img / 255.0 - mean) / std), vertices, -1
            else:
                return img2tensor((img / 255.0 - mean) / std), vertices, idx


class Model_pred:
    def __init__(self, models, dl, tta: bool = False, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.no_grad():
            for batch in iter(self.dl):
                if not shift:
                    x, y = batch
                else:
                    x, verts, y = batch

                if (y >= 0).sum() > 0:
                    x = x[y >= 0].to(device)
                    y = y[y >= 0]
                    if self.half:
                        x = x.half()
                    py = None
                    for model in self.models:
                        p = torch.sigmoid(model(x)).detach()
                        py = p if py is None else py + p
                    if self.tta:
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            for model in self.models:
                                p = torch.sigmoid(model(xf)).detach()
                                py += torch.flip(p, f)
                        py /= 1 + len(flips)
                    py /= len(self.models)
                    py = F.interpolate(
                        py, scale_factor=reduce, mode="bilinear", align_corners=False
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu()
                    if not shift:
                        for i in range(py.shape[0]):
                            yield py[i], y[i]
                    else:
                        py = py.squeeze(-1).numpy()
                        verts = verts[y >= 0].numpy()
                        for i in range(py.shape[0]):
                            yield py[i], verts[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)


models = []
for path in MODELS:
    try:
        state_dict = torch.load(path, map_location=torch.device("cpu"))
    except Exception:
        state_dict = None
    model = smp.Unet(model_name, encoder_weights=None, classes=1)
    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
        except Exception:
            pass  # ignore mismatched keys for dummy model
    model.float()
    model.eval()
    model.to(device)
    models.append(model)
del state_dict

names, preds = [], []

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]
    ds = HuBMAPDataset(idx)
    dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
    mp = Model_pred(models, dl)
    if not shift:
        mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
        for p, i in mp:
            mask[i.item()] = (p.squeeze(-1) > TH).byte()
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
        rle = rle_encode_less_memory(mask.numpy())
    else:
        mask = np.zeros(ds.shape, dtype=np.uint8)
        for pred, vert, _ in mp:
            x1, x2, y1, y2 = vert
            mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)
        mask = (mask > 0.5).astype(np.uint8)
        rle = rle_encode_less_memory(mask) if mask.sum() > 0 else ""
    names.append(idx)
    preds.append(rle)
    del mask, ds, dl
    gc.collect()



## === cell 1
submission = pd.DataFrame({"img": names, "pixels": preds})
submission.to_csv("submission.csv", index=False)
display(submission)
