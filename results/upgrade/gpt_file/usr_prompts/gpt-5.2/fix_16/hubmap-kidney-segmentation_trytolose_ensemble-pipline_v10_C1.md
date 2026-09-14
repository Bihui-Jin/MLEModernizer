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

albumentations==2.0.8
geopandas==0.14.4
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

0.9378791552372312

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (it isn’t installed in this environment) by providing a tiny fallback `smp.FPN` wrapper that uses `torchvision`’s `fcn_resnet50`, which preserves the same inference API (`model(image) -> logits`) so the rest of your pipeline stays the same. I also fix the notebook cell numbering/order so imports run before later cells, which resolves the cascading `NameError`s (for `pd`, `Dataset`, `dataclass`, `my_models`, `df_sub`). Finally, I make the weight loading robust to common checkpoint key formats (`state_dict` and `module.` prefixes) and ensure the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (it isn’t installed here) by switching TIFF reading/windowing to OpenCV, while keeping your sliding-window inference, transforms, model API, and RLE encoding unchanged. I also fix the cell numbering so imports run first (your current tracebacks show later cells execute without `torch/Dataset/Window` defined due to the failing first cell). To avoid another hard failure when external weights aren’t attached, I keep the same weight-loading logic but fall back to a safe “empty mask” submission if weights are missing, ensuring a valid `submission.csv` is always produced. This should move the score up from 0.0 (invalid/empty run) to a meaningful score when weights are present, and at minimum stop runtime errors.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from producing an effectively empty submission because inference is currently skipped (`SKIP_COMMIT=True` makes `count_thr=5` while the sample submission has only 3 rows), so no predictions ever get written. I make the minimal change to always run inference for all rows in `df_sub` (no gating by row count), keeping your model/inference core logic unchanged. I also make `_check_background` robust to Albumentations’ `ToTensorV2` collating by ensuring `not_background` is a Python `bool` before checking it, preventing accidental skipping. These changes should move the score upward toward your target without altering the model architecture or training approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from the fallback path: your weights aren’t being found/loaded, so you submit empty masks for every image. The minimal fix is to make weight discovery robust (search common Kaggle input locations and auto-extract any `.tar` found) while keeping your same model/inference logic unchanged. I also switch `SKIP_COMMIT` to `False` and make `SKIP_BACKGROUND` effectively “safe” by never skipping a tile unless we’re sure it’s background (this avoids accidentally predicting all-empty due to a background check edge case). These changes should move you substantially upward toward the target without changing the architecture, loss, or inference approach.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with always taking the “no weights available → empty masks” path, so the smallest meaningful improvement is to reliably find and load the provided `.pth` weights after extraction. I fix the `extracted_any` check (it currently always evaluates truthy due to using generator objects) and add a minimal fallback search that collects all extracted `.pth` files so `weights_available` becomes `True` when weights exist. I also fix `_check_background`/`SKIP_BACKGROUND` logic: the function currently returns `True` for “background”, but inference treats `True` as “not background”, which can skip real tissue tiles; this is a semantic bug that can severely hurt Dice. These are small, targeted fixes that keep your model/inference/RLE core logic unchanged and should move the score upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with writing an “all-empty mask” submission because the script never finds/loads the real trained weights, so it always falls back to `predicted=""`. I make weight discovery/load robust without changing the model/inference core logic: search inside the competition dataset itself for any `.pth` files and auto-use them if your expected paths don’t exist. I also add a minimal safety fix to use OpenCV’s TIFF reader flags that better handle multi-page/large TIFFs, and ensure we don’t accidentally skip valid tiles due to background-check type issues (keeping the same skip semantics). These are the smallest changes that should move your Dice score up toward the target when weights are present, while still always producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import gc
import json
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import tifffile  # type: ignore

    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False

try:
    sys.path.append("../input/segmentation-models-pytorch-install")
    import segmentation_models_pytorch as smp  # type: ignore

    _HAS_SMP = True
except Exception:
    _HAS_SMP = False
    import torchvision

    class _FallbackFPN(torch.nn.Module):
        """
        Minimal drop-in replacement for smp.FPN(encoder_name, encoder_weights=None)
        returning 1-channel logits for binary segmentation.

        Change (score-relevant): use a torchvision model with stronger dense prediction head
        (DeepLabV3) than FCN to reduce incompatibility risk and produce non-trivial masks when
        SMP isn't available. API remains model(x)->logits (N,1,H,W).
        """

        def __init__(self, encoder_name: str, encoder_weights=None):
            super().__init__()
            self.net = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=None, weights_backbone=None, num_classes=1
            )

        def forward(self, x):
            out = self.net(x)
            return out["out"]

    class smp:  # noqa: N801
        FPN = _FallbackFPN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

print("CUDA available:", torch.cuda.is_available())
print("segmentation_models_pytorch available:", _HAS_SMP)
print("tifffile available:", _HAS_TIFFFILE)




## === cell 1
def _find_existing_dir(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


weights_dir = _find_existing_dir(
    [
        "../input/hubmap-folds-2",
        "/kaggle/input/hubmap-folds-2",
        "../input/hubmap-folds",
        "/kaggle/input/hubmap-folds",
        "../input/hubmap-kidney-segmentation",
        "/kaggle/input/hubmap-kidney-segmentation",
        "../input",
        "/kaggle/input",
    ]
)

print("Weights dir:", None if weights_dir is None else str(weights_dir.resolve()))

if weights_dir is not None:
    print("Contents (first 50):")
    for i, p in enumerate(sorted(weights_dir.iterdir())):
        if i >= 50:
            print(" - ...")
            break
        print(" -", p.name)
else:
    print(
        "WARNING: No weights directory found in common locations. "
        "Will try to discover weights inside the competition dataset; otherwise will produce empty masks."
    )




## === cell 2
def _safe_extract_tar(tar_path: Path, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cmd = f"tar -xvf {str(tar_path)} -C {out_dir}"
    print("Running:", cmd)
    return os.system(cmd)


extracted_dirs = []
if weights_dir is not None:
    os.makedirs("model_1", exist_ok=True)
    os.makedirs("model_2", exist_ok=True)
    os.makedirs("model_3", exist_ok=True)

    preferred = [
        ("1024_avg_last.tar", "model_1"),
        ("1024_512_pseudo_v1_avg.tar", "model_2"),
        ("4096_1365_avg.tar", "model_3"),
    ]
    for fname, out_dir in preferred:
        tar_path = weights_dir / fname
        if tar_path.exists():
            _safe_extract_tar(tar_path, out_dir)
            extracted_dirs.append(out_dir)

    extracted_any = any(
        len(list(Path(d).rglob("*.pth"))) > 0 for d in ["model_1", "model_2", "model_3"]
    )

    if not extracted_any:
        tars = sorted(list(weights_dir.rglob("*.tar")))
        if len(tars) > 0:
            print(
                "No .pth found after preferred extraction; extracting all discovered .tar files as fallback."
            )
            for i, tar_path in enumerate(tars):
                out_dir = f"model_extra_{i}"
                _safe_extract_tar(tar_path, out_dir)
                extracted_dirs.append(out_dir)
        else:
            print("No .tar files found under weights_dir.")
else:
    print(
        "WARNING: weights_dir is missing. Will try to discover weights inside the competition dataset; "
        "otherwise will produce empty masks."
    )



## === cell 3
BATCH_SIZE = 1
NUM_WORKERS = 0
CROP_SIZE = 1024 * 4
STEP = 1024 * 2
THR = 0.4
VOTE = 0

SKIP_COMMIT = False
SKIP_BACKGROUND = True

df_sub = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

if "img" in df_sub.columns and "pixels" in df_sub.columns:
    pass
elif "id" in df_sub.columns and "predicted" in df_sub.columns:
    df_sub = df_sub.rename(columns={"id": "img", "predicted": "pixels"})
elif "id" in df_sub.columns and "pixels" in df_sub.columns:
    df_sub = df_sub.rename(columns={"id": "img"})
elif "img" in df_sub.columns and "predicted" in df_sub.columns:
    df_sub = df_sub.rename(columns={"predicted": "pixels"})
else:
    raise ValueError(f"Unexpected sample_submission columns: {df_sub.columns.tolist()}")

assert {"img", "pixels"}.issubset(
    df_sub.columns
), f"Unexpected submission columns after normalize: {df_sub.columns.tolist()}"

print("Sample submission rows:", len(df_sub))
print(df_sub.head())




## === cell 4
@dataclass
class ModelConfig:
    encoder: str
    weights_path: List[str]
    img_size: int
    weight_blend: float


my_models = [
    ModelConfig(
        "timm-efficientnet-b1",
        [
            "./model_3/fold0_avg_0.9448.pth",
            "./model_3/fold1_avg_0.9312.pth",
            "./model_3/fold2_avg_0.9198.pth",
            "./model_3/fold3_avg_0.9456.pth",
        ],
        1344,
        1.0,
    ),
]

TRAIN_IMG_SIZE = max([cfg.img_size for cfg in my_models])
print("TRAIN_IMG_SIZE:", TRAIN_IMG_SIZE)




## === cell 5
def rle_encode_less_memory(img: np.ndarray) -> str:
    """
    Correct RLE encoding with proper sentinel padding so runs at the start/end aren't dropped.
    """
    if img is None:
        return ""
    m = (img > 0).astype(np.uint8)
    pixels = m.T.flatten()  # column-major per competition
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = changes.copy()
    runs[1::2] = runs[1::2] - runs[::2]
    if runs.size == 0:
        return ""
    return " ".join(str(x) for x in runs)


def valid_transform(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )




## === cell 6
def _check_background(img, crop_size) -> bool:
    s_th = 40  # saturation blanking threshold
    p_th = 1000 * (crop_size // 256) ** 2
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ss = hsv[:, :, 1]
        sat_cnt = int((ss > s_th).sum())
        img_sum = int(img.sum())
        is_background = (sat_cnt <= p_th) or (img_sum <= p_th)
        return bool(not is_background)
    except Exception:
        return True


def _read_tiff_bgr(path: str) -> np.ndarray:
    if _HAS_TIFFFILE:
        img = tifffile.imread(path)
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[2] not in (3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 2:
            img = cv2.cvtColor(img.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported TIFF shape: {img.shape} for {path}")
        return img
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read TIFF: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


class SingleTiffDataset(Dataset):
    """
    Keep the same sliding-window logic but read the full TIFF and slice windows in-memory.
    """

    def __init__(
        self,
        tiff_path: str,
        all_img_sizes,
        crop_size=1024,
        step=512,
        expected_wh: Optional[Tuple[int, int]] = None,
    ):
        self.crop_size = crop_size
        self.all_img_sizes = all_img_sizes
        self.step = step
        self.tiff_path = str(tiff_path)

        img = _read_tiff_bgr(self.tiff_path)

        if expected_wh is not None:
            ew, eh = expected_wh
            if (img.shape[1], img.shape[0]) != (ew, eh):
                print(
                    f"WARNING: metadata (w,h)=({ew},{eh}) != read image (w,h)=({img.shape[1]},{img.shape[0]}) for {Path(self.tiff_path).name}"
                )

        self.img = img
        self.h, self.w = img.shape[:2]

        self.row_count = 1 + math.ceil((self.h - self.crop_size) / self.step)
        self.col_count = 1 + math.ceil((self.w - self.crop_size) / self.step)

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = self.w - self.crop_size
        if y + self.crop_size > self.h:
            y = self.h - self.crop_size

        img_crop = self.img[y : y + self.crop_size, x : x + self.crop_size]
        if img_crop.shape[0] != self.crop_size or img_crop.shape[1] != self.crop_size:
            pad_y = self.crop_size - img_crop.shape[0]
            pad_x = self.crop_size - img_crop.shape[1]
            img_crop = cv2.copyMakeBorder(
                img_crop,
                0,
                max(0, pad_y),
                0,
                max(0, pad_x),
                borderType=cv2.BORDER_REFLECT,
            )

        transformed_imgs = {
            img_size: valid_transform(img_size)(image=img_crop)["image"]
            for img_size in self.all_img_sizes
        }
        transformed_imgs["crop_names"] = f"{x}_{y}"
        transformed_imgs["not_background"] = _check_background(img_crop, self.crop_size)
        return transformed_imgs




## === cell 7
def _load_state_dict_flexible(
    model: torch.nn.Module, ckpt_path: str
) -> Tuple[List[str], List[str]]:
    """
    handle checkpoints saved as:
      - raw state_dict
      - {"state_dict": ...}
      - DataParallel keys prefixed with "module."
    Returns (missing_keys, unexpected_keys).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if len(unexpected) > 0:
        print(
            f"WARNING: unexpected keys in {ckpt_path}: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
        )
    if len(missing) > 0:
        print(
            f"WARNING: missing keys in {ckpt_path}: {missing[:5]}{'...' if len(missing)>5 else ''}"
        )
    return missing, unexpected


def _discover_pth_files(search_dirs):
    pths = []
    for d in search_dirs:
        dp = Path(d)
        if dp.exists():
            pths.extend(sorted([p for p in dp.rglob("*.pth") if p.is_file()]))
    return pths


def _choose_best_matching_weights(
    expected_paths: List[str], discovered: List[Path]
) -> List[str]:
    """
    Change (score-relevant): instead of mapping arbitrary discovered weights to expected slots,
    pick the closest filename match per expected path (e.g., fold0_avg_*.pth -> fold0_avg_*.pth).
    This greatly increases chance that correct ensemble weights are loaded, producing non-empty
    predictions and better Dice, while keeping architecture/inference identical.
    """
    remaining = discovered.copy()
    chosen = []
    for exp in expected_paths:
        exp_name = Path(exp).name.lower()
        fold_token = None
        for t in ["fold0", "fold1", "fold2", "fold3", "fold4"]:
            if t in exp_name:
                fold_token = t
                break

        best_i = None
        best_score = -(10**9)
        for i, p in enumerate(remaining):
            n = p.name.lower()
            score = 0
            if fold_token is not None and fold_token in n:
                score += 1000
            for tok in exp_name.replace(".", "_").split("_"):
                if tok and tok in n:
                    score += 5
            if "avg" in exp_name and "avg" in n:
                score += 50
            if "pseudo" in exp_name and "pseudo" in n:
                score += 20
            score -= abs(len(exp_name) - len(n))
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            chosen.append(exp)  # leave as-is; will be handled by existence check
        else:
            chosen.append(str(remaining.pop(best_i)))
    return chosen


def _looks_like_hubmap_weight(p: Path) -> bool:
    n = p.name.lower()
    return ("fold" in n) or ("avg" in n) or ("pseudo" in n)


expected_missing = []
for cfg in my_models:
    for wp in cfg.weights_path:
        if not Path(wp).exists():
            expected_missing.append(wp)

if len(expected_missing) > 0:
    discovered_all = _discover_pth_files(
        ["model_1", "model_2", "model_3"]
        + [str(p) for p in Path(".").glob("model_extra_*")]
        + [
            "../input/hubmap-kidney-segmentation",
            "/kaggle/input/hubmap-kidney-segmentation",
            "../input",
            "/kaggle/input",
        ]
    )
    discovered = [p for p in discovered_all if _looks_like_hubmap_weight(Path(p))]
    discovered = sorted([Path(p) for p in discovered])
    print(
        f"Expected weights missing ({len(expected_missing)}). Discovered compatible-looking .pth files:",
        len(discovered),
    )

    if len(discovered) > 0:
        new_models = []
        for cfg in my_models:
            new_paths = _choose_best_matching_weights(cfg.weights_path, discovered)
            new_models.append(
                ModelConfig(
                    encoder=cfg.encoder,
                    weights_path=new_paths,
                    img_size=cfg.img_size,
                    weight_blend=cfg.weight_blend,
                )
            )
        my_models = new_models
        print(
            "Remapped my_models weights_path to best-matching discovered files where possible."
        )

all_img_sizes = set([cfg.img_size for cfg in my_models])
models_with_config = []
weights_available = True

MAX_MISSING_FRAC_FOR_ACCEPT = 0.35

for cfg in my_models:
    models_group = []
    for w_path in cfg.weights_path:
        if not Path(w_path).exists():
            print(f"WARNING: Weight file not found: {w_path}")
            weights_available = False
            break
        model = smp.FPN(cfg.encoder, encoder_weights=None).to(device)
        missing, unexpected = _load_state_dict_flexible(model, w_path)
        total_keys = max(1, len(model.state_dict()))
        missing_frac = len(missing) / total_keys
        if (not _HAS_SMP) and (missing_frac > MAX_MISSING_FRAC_FOR_ACCEPT):
            print(
                f"WARNING: Checkpoint {w_path} appears incompatible with torchvision fallback "
                f"(missing {len(missing)}/{total_keys} keys). Falling back to empty-mask submission."
            )
            weights_available = False
            break
        model.eval()
        models_group.append(model)
    if not weights_available:
        break
    models_with_config.append((cfg, models_group))

print("Weights available:", weights_available)
if weights_available:
    print("Loaded model groups:", len(models_with_config))
    print("All image sizes:", sorted(list(all_img_sizes)))




## === cell 8
def inference(data_loader, models_with_config, crop_size):
    img_size = (data_loader.dataset.h, data_loader.dataset.w)

    mask_pred = np.zeros(img_size, dtype=np.float32)
    mask_cnt = np.zeros(img_size, dtype=np.float32)

    for batch in tqdm(data_loader, ncols=70, leave=True):
        nb = batch["not_background"]
        if torch.is_tensor(nb):
            nb = bool(nb.item())
        else:
            nb = bool(nb)

        if SKIP_BACKGROUND is True and nb is False:
            continue

        with torch.no_grad():
            pred_total = None
            for cfg, model_group in models_with_config:
                image = batch[cfg.img_size].to(device, non_blocking=True)
                pred_group = None

                for model in model_group:
                    pred = model(image)
                    pred = pred.sigmoid()
                    if cfg.img_size != TRAIN_IMG_SIZE:
                        pred = torch.nn.functional.interpolate(
                            pred,
                            size=TRAIN_IMG_SIZE,
                            mode="bilinear",
                            align_corners=False,
                        )
                    pred_group = pred if pred_group is None else (pred_group + pred)

                pred_group = pred_group.squeeze()
                if len(pred_group.shape) == 2:
                    pred_group = pred_group.unsqueeze(0)

                pred_group = cfg.weight_blend * (pred_group / len(model_group))
                pred_total = (
                    pred_group if pred_total is None else (pred_total + pred_group)
                )

            pred_total = pred_total.detach().cpu().numpy().astype(np.float32)

            crop_names = batch["crop_names"]
            if isinstance(crop_names, str):
                crop_names = [crop_names]

            for predict_single, crop_name in zip(pred_total, crop_names):
                x = int(crop_name.split("_")[-2])
                y = int(crop_name.split("_")[-1])

                ps = predict_single.astype(np.float32)
                if ps.ndim == 3:
                    ps = ps[0]
                if crop_size != TRAIN_IMG_SIZE:
                    ps = cv2.resize(
                        ps,
                        (crop_size, crop_size),
                        interpolation=cv2.INTER_LINEAR,
                    )

                y2 = min(y + crop_size, mask_pred.shape[0])
                x2 = min(x + crop_size, mask_pred.shape[1])
                ps = ps[: (y2 - y), : (x2 - x)]

                mask_pred[y:y2, x:x2] += ps
                mask_cnt[y:y2, x:x2] += 1.0

    mask_cnt = np.maximum(mask_cnt, 1.0)
    mask_pred = mask_pred / mask_cnt

    mask_bin = (mask_pred > THR).astype(np.uint8)
    mask_rle = rle_encode_less_memory(mask_bin)
    del mask_pred, mask_bin, mask_cnt
    gc.collect()
    return mask_rle




## === cell 9
INFO_CSV = "../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
if not Path(INFO_CSV).exists():
    INFO_CSV = "../input/HuBMAP-20-dataset_information.csv"
info_df = pd.read_csv(INFO_CSV)

id_to_wh = {}
for _, r in info_df.iterrows():
    f = str(r.get("image_file", ""))
    if f.endswith(".tiff"):
        _id = Path(f).stem
        w = int(r["width_pixels"])
        h = int(r["height_pixels"])
        id_to_wh[_id] = (w, h)


def _find_test_tiff(img_id: str) -> str:
    candidates = [
        f"../input/hubmap-kidney-segmentation/test/{img_id}.tiff",
        f"../input/test/{img_id}.tiff",
        f"/kaggle/input/hubmap-kidney-segmentation/test/{img_id}.tiff",
        f"/kaggle/input/test/{img_id}.tiff",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    roots = [
        Path("../input/hubmap-kidney-segmentation/test"),
        Path("../input/test"),
        Path("/kaggle/input/hubmap-kidney-segmentation/test"),
        Path("/kaggle/input/test"),
    ]
    for r in roots:
        if r.exists():
            hits = list(r.rglob(f"{img_id}.tiff"))
            if len(hits) > 0:
                return str(hits[0])
    raise FileNotFoundError(f"Missing test TIFF for id={img_id}")


if not weights_available:
    print(
        "No compatible weights available -> writing empty RLE masks (valid submission, no leakage)."
    )
    df_sub["pixels"] = ""
else:
    for idx, row in df_sub.iterrows():
        img_id = str(row["img"])
        tiff_path = _find_test_tiff(img_id)
        expected_wh = id_to_wh.get(img_id, None)

        test_ds = SingleTiffDataset(
            tiff_path=tiff_path,
            all_img_sizes=all_img_sizes,
            crop_size=CROP_SIZE,
            step=STEP,
            expected_wh=expected_wh,
        )

        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        rle = inference(test_loader, models_with_config, CROP_SIZE)
        df_sub.loc[idx, "pixels"] = rle



## === cell 10
os.system("rm -rf model_1")
os.system("rm -rf model_2")
os.system("rm -rf model_3")
for p in Path(".").glob("model_extra_*"):
    os.system(f"rm -rf {str(p)}")

df_sub[["img", "pixels"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
print("Non-empty pixels rows:", int((df_sub["pixels"].astype(str).str.len() > 0).sum()))
