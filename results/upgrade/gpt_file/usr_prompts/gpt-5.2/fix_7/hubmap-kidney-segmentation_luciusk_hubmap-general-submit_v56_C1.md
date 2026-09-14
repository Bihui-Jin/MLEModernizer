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

0.9368490486475284

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import/runtime blockers so the notebook runs in this Kaggle environment without the unavailable `segmentation_models_pytorch` package and without missing model weight files. To keep the core inference logic (tiling, normalization, thresholding, and RLE submission) intact, I add a tiny fallback UNet-like stub that outputs low-probability masks when the real pretrained models can’t be loaded, allowing end-to-end execution and a valid `submission.csv` to be produced. I also fix the submission column name mismatch (`sample_submission` uses `predicted`, not `pixels`) and make the input paths robust to the provided directory layout. These changes are score-neutral in intent (they mainly restore execution); without the missing weights, a competitive score isn’t achievable, but the pipeline run and submit correctly.'
- What this solution (achieved 0.0) has done: 'I fix the crash caused by `tifffile.memmap` not being supported for these TIFFs by adding a safe fallback reader that uses `tifffile.imread` when memmap fails, while keeping the same tiling/inference/RLE logic. I also ensure the runtime environment differences don’t break execution by keeping paths and submission columns aligned with `sample_submission.csv` (`id`, `predicted`). These changes are purely to restore end-to-end inference and should move the score up from 0.0 (which currently comes from not completing predictions), without changing the model/inference semantics beyond the necessary TIFF reading fallback. The script always write a valid `submission.csv` in the working directory.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF reading crash by switching the loader to OpenCV for these JPEG-compressed TIFFs, since `tifffile` requires the unavailable `imagecodecs` package to decode them. This keeps the core tiling/inference/RLE pipeline intact while unblocking end-to-end execution and producing a valid `submission.csv`. I also make the dataset path resolution robust to the provided directory layout so it finds the actual `.tiff` files. These changes are primarily correctness/runtime fixes and should raise the score from 0.0 (currently caused by failure before submission generation) toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the OpenCV TIFF loading crash caused by its max image pixel safety limit by switching to a safe, windowed TIFF reader that can extract tiles without decoding the full-slide image at once. This keeps the existing tiling/inference/RLE logic the same, but changes the dataset to read each tile directly from disk via `tifffile.TiffFile(...).pages[0].asarray(key=...)`, which works without loading the whole image and avoids the OpenCV assertion. I also ensure width/height are derived from TIFF metadata (so `ds.shape` is correct) and that the submission is written with the required `id,predicted` columns to `submission.csv`. These changes should move the score up from 0.0 (previously crashing before submission) while preserving the model pipeline semantics.'
- What this solution (achieved 0.0) has done: 'I fix the immediate runtime error by updating the TIFF window reader to use the correct `tifffile` slicing API (recent versions don’t accept `key=` in `asarray`). To keep the core tiling/inference/RLE logic unchanged while ensuring end-to-end execution, I implement a robust window read that first tries `page.asarray(region=...)` and falls back to full-page read + slicing only if needed. I also add a small safeguard so the RLE encoder works even if a mask is entirely empty (it output an empty string instead of crashing). These changes are execution-focused and should move your score up from 0.0 by producing a valid submission with non-empty predictions when the model outputs positives.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

try:
    import segmentation_models_pytorch as smp  # type: ignore

    HAS_SMP = True
except Exception:
    HAS_SMP = False

    import torch.nn as nn

    class _FallbackUnet(nn.Module):
        """
        Minimal, lightweight UNet-like stub to keep the core inference pipeline runnable
        when segmentation_models_pytorch and pretrained weights are unavailable.
        It outputs logits with shape (B,1,H,W) as expected by the rest of the code.
        """

        def __init__(self, classes: int = 1):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, classes, kernel_size=1, bias=True),
            )

        def forward(self, x):
            return self.conv(x)

    class smp:  # noqa: N801
        Unet = staticmethod(
            lambda encoder_name, encoder_weights=None, classes=1: _FallbackUnet(
                classes=classes
            )
        )


print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("segmentation_models_pytorch available:", HAS_SMP)



## === cell 1
sz = 256  # size of tiles at model input resolution
reduce = 4  # reduce original images by 4x for model input
TH = 0.35  # threshold for positive predictions

_CAND_TEST_DIRS = [
    "../input/hubmap-kidney-segmentation/test/",
    "/kaggle/input/hubmap-kidney-segmentation/test/",
    "/kaggle/data/hubmap-kidney-segmentation/test/",
    "../input/test/",
    "/kaggle/input/hubmap-kidney-segmentation/test_images/",
    "/kaggle/input/hubmap-kidney-segmentation/test/",
]
DATA = next((p for p in _CAND_TEST_DIRS if os.path.isdir(p)), _CAND_TEST_DIRS[0])

MODELS = [
    f"../input/alldatad488others/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]

_SAMPLE_CANDS = [
    "../input/hubmap-kidney-segmentation/sample_submission.csv",
    "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv",
    "/kaggle/data/hubmap-kidney-segmentation/sample_submission.csv",
]
sample_path = next((p for p in _SAMPLE_CANDS if os.path.isfile(p)), _SAMPLE_CANDS[0])
df_sample = pd.read_csv(sample_path)

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "efficientnet-b4"
shift = True
minoverlap = 300

print("Using DATA:", DATA)
print("Sample submission columns:", df_sample.columns.tolist())
print("Num test ids:", len(df_sample))




## === cell 2
def enc2mask(encs, shape):
    """
    Decode RLE list to mask (not used in inference here, but keep functional).
    Fix: np.float is removed in recent NumPy; use np.floating.
    """
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
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
    """
    Kaggle HuBMAP expects pixels numbered top-to-bottom then left-to-right.
    This encoding uses transpose+flatten consistent with common solutions.
    Fix: handle fully-empty masks safely (return empty string).
    """
    pixels = img.T.flatten()
    if pixels.size == 0 or pixels.max() == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

s_th = 40  # saturation blanking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return array (N,4) where each row is x1,x2,y1,y2 in original image coordinates.
    shape is (H, W).
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




## === cell 4
def _find_tiff_path(idx: str) -> str:
    """
    Try a small set of known locations and pick the first existing .tiff.
    """
    cands = [
        os.path.join(DATA, idx + ".tiff"),
        os.path.join(os.path.dirname(DATA.rstrip("/")), "test", idx + ".tiff"),
        os.path.join("/kaggle/input/hubmap-kidney-segmentation/test", idx + ".tiff"),
        os.path.join("/kaggle/data/hubmap-kidney-segmentation/test", idx + ".tiff"),
        os.path.join("../input/test", idx + ".tiff"),
        os.path.join("/kaggle/input/test", idx + ".tiff"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return cands[0]


class _TiffWindowReader:
    """
    Fix: tifffile's window-reading API varies by version; recent versions do not accept
    `key=` for TiffPage.asarray. Use region=... when available, else fall back safely.
    """

    def __init__(self, path: str):
        self.path = path
        self.tf = tiff.TiffFile(path)
        self.page = self.tf.pages[0]
        shp = self.page.shape  # could be (H,W,3) or (3,H,W) depending on file
        if len(shp) == 3 and shp[-1] in (3, 4):
            self.layout = "HWC"
            self.h, self.w = int(shp[0]), int(shp[1])
            self.c = int(shp[2])
        elif len(shp) == 3 and shp[0] in (3, 4):
            self.layout = "CHW"
            self.c = int(shp[0])
            self.h, self.w = int(shp[1]), int(shp[2])
        else:
            raise ValueError(f"Unexpected TIFF shape for {path}: {shp}")

    def close(self):
        try:
            self.tf.close()
        except Exception:
            pass

    def _as_uint8(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        return arr

    def read_window_hwc(self, x1: int, x2: int, y1: int, y2: int) -> np.ndarray:
        """
        Return HWC uint8 window [x1:x2, y1:y2, :3].
        """
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

        try:
            if self.layout == "HWC":
                arr = self.page.asarray(region=(x1, y1, x2 - x1, y2 - y1))
                arr = self._as_uint8(arr)
                if arr.ndim == 2:
                    arr = np.repeat(arr[:, :, None], 3, axis=2)
                return arr[:, :, :3]
            else:
                arr = self.page.asarray(region=(x1, y1, x2 - x1, y2 - y1))
                arr = self._as_uint8(arr)
                if arr.ndim == 3 and arr.shape[0] in (3, 4):
                    arr = np.moveaxis(arr, 0, -1)
                if arr.ndim == 2:
                    arr = np.repeat(arr[:, :, None], 3, axis=2)
                return arr[:, :, :3]
        except Exception:
            full = self.page.asarray()
            full = self._as_uint8(full)
            if self.layout == "HWC":
                tile = full[x1:x2, y1:y2, :3]
                if tile.ndim == 2:
                    tile = np.repeat(tile[:, :, None], 3, axis=2)
                return tile
            else:
                tile = full[:3, x1:x2, y1:y2]
                tile = np.moveaxis(tile, 0, -1)
                if tile.ndim == 2:
                    tile = np.repeat(tile[:, :, None], 3, axis=2)
                return tile[:, :, :3]


def _read_tile_from_reader(reader: _TiffWindowReader, x1, x2, y1, y2, out_sz):
    """
    Read a tile and pad to (out_sz,out_sz,3) with zeros if needed.
    """
    h = x2 - x1
    w = y2 - y1
    img = np.zeros((out_sz, out_sz, 3), dtype=np.uint8)
    tile = reader.read_window_hwc(int(x1), int(x2), int(y1), int(y2))
    img[:h, :w, : tile.shape[2]] = tile[:, :, :3]
    return img


if shift:

    class HuBMAPDataset(Dataset):
        def __init__(self, idx, sz=sz, reduce=reduce):
            self.path = _find_tiff_path(idx)
            if not os.path.isfile(self.path):
                raise FileNotFoundError(f"Missing TIFF for id={idx}: {self.path}")

            self.reader = _TiffWindowReader(self.path)
            self.shape = (self.reader.h, self.reader.w)

            self.reduce = reduce
            self.sz = reduce * sz  # tile size in original resolution
            self.mask_grid = make_grid(
                self.shape, window=self.sz, min_overlap=minoverlap
            )

        def __len__(self):
            return len(self.mask_grid)

        def __getitem__(self, idx):
            x1, x2, y1, y2 = self.mask_grid[idx]
            img = _read_tile_from_reader(
                self.reader, int(x1), int(x2), int(y1), int(y2), self.sz
            )

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

    class Model_pred:
        def __init__(self, models, dl, tta: bool = False, half: bool = False):
            self.models = models
            self.dl = dl
            self.tta = tta
            self.half = half

        def __iter__(self):
            with torch.no_grad():
                for x, z, y in iter(self.dl):
                    if (y >= 0).sum() > 0:
                        keep = y >= 0
                        x = x[keep].to(device)
                        z = z[keep]
                        y = y[keep]

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
                            py,
                            scale_factor=reduce,
                            mode="bilinear",
                            align_corners=False,
                        )
                        py = py.permute(0, 2, 3, 1).float().cpu()

                        py = py.squeeze(-1).numpy()
                        z = z.numpy()

                        for i in range(len(py)):
                            yield py[i], z[i], y[i]

        def __len__(self):
            return len(self.dl.dataset)

    models = []
    existing = [p for p in MODELS if os.path.isfile(p)]
    if len(existing) == 0:
        model = smp.Unet(model_name, encoder_weights=None, classes=1)
        model.float()
        model.eval()
        model.to(device)
        models = [model]
        print(
            "WARNING: No model weight files found. Using fallback/random-initialized model."
        )
    else:
        for path in existing:
            state_dict = torch.load(path, map_location=torch.device("cpu"))
            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            model.load_state_dict(state_dict)
            model.float()
            model.eval()
            model.to(device)
            models.append(model)
        print(f"Loaded {len(models)} model(s) from disk.")

    names, preds = [], []
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        idx = row["id"]
        ds = HuBMAPDataset(idx)
        dl = DataLoader(
            ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
        )

        mp = Model_pred(models, dl)

        mask = np.zeros(ds.shape, dtype=np.uint8)
        for pred, vert, _i in iter(mp):
            x1, x2, y1, y2 = vert
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            tile_h = x2 - x1
            tile_w = y2 - y1
            mask[x1:x2, y1:y2] += (pred[:tile_h, :tile_w] > TH).astype(np.uint8)

        mask = (mask > 0.5).astype(np.uint8)
        rle = rle_encode_less_memory(mask)

        names.append(idx)
        preds.append(rle)

        try:
            ds.reader.close()
        except Exception:
            pass

        del mask, ds, dl
        gc.collect()

else:
    raise RuntimeError(
        "This script is configured with shift=True; shift=False branch requires rasterio in the original code."
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/550351525.py in read_window_hwc(self, x1, x2, y1, y2)
     60             if self.layout == "HWC":
---> 61                 arr = self.page.asarray(region=(x1, y1, x2 - x1, y2 - y1))
     62                 arr = self._as_uint8(arr)

TypeError: TiffPage.asarray() got an unexpected keyword argument 'region'

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/550351525.py in <cell line: 0>()
    233 
    234         mask = np.zeros(ds.shape, dtype=np.uint8)
--> 235         for pred, vert, _i in iter(mp):
    236             x1, x2, y1, y2 = vert
    237             x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

/tmp/ipykernel_55/550351525.py in __iter__(self)
    155         def __iter__(self):
    156             with torch.no_grad():
--> 157                 for x, z, y in iter(self.dl):
    158                     if (y >= 0).sum() > 0:
    159                         keep = y >= 0

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/550351525.py in __getitem__(self, idx)
    125         def __getitem__(self, idx):
    126             x1, x2, y1, y2 = self.mask_grid[idx]
--> 127             img = _read_tile_from_reader(
    128                 self.reader, int(x1), int(x2), int(y1), int(y2), self.sz
    129             )

/tmp/ipykernel_55/550351525.py in _read_tile_from_reader(reader, x1, x2, y1, y2, out_sz)
     98     w = y2 - y1
     99     img = np.zeros((out_sz, out_sz, 3), dtype=np.uint8)
--> 100     tile = reader.read_window_hwc(int(x1), int(x2), int(y1), int(y2))
    101     img[:h, :w, : tile.shape[2]] = tile[:, :, :3]
    102     return img

/tmp/ipykernel_55/550351525.py in read_window_hwc(self, x1, x2, y1, y2)
     75         except Exception:
     76             # Fallback: full read then slice (may be heavier but ensures correctness)
---> 77             full = self.page.asarray()
     78             full = self._as_uint8(full)
     79             if self.layout == "HWC":

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

## === cell 5
df = pd.DataFrame({"id": names, "predicted": preds})
df = df.set_index("id").reindex(df_sample["id"]).reset_index()

out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(df.head())
print("Wrote:", out_path, "rows:", len(df))
print("Any empty predictions:", int((df["predicted"].fillna("") == "").sum()))
