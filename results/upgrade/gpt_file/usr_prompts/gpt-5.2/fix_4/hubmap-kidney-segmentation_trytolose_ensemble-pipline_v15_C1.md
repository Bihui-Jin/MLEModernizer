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

0.9498136762796549

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by switching the TIFF reading and window cropping to OpenCV while keeping the same sliding-window inference logic and RLE encoding. I also fix the cascading `NameError`s by ensuring the global path/config variables are defined in the first cell and by renumbering cells starting from 1. Since the provided environment appears to include only JSONs under `test/`, I add a minimal fallback to rasterize the provided anatomical-structure polygons into a 3‑channel image when a `.tiff` is missing, so the pipeline can still run end-to-end and write `submission.csv`. Finally, I make the submission column naming robust (`predicted` vs `pixels`) by honoring the competition’s `sample_submission.csv` format in your input folder.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on external weight archives that are not present in this Kaggle environment by making weight loading optional and falling back to a deterministic, non-crashing baseline predictor when weights are missing. This fixes the current `FileNotFoundError` chain (tar extraction and missing `.pth/.pt` files) so the notebook runs end-to-end and always writes a valid `submission.csv`. To move the score up from 0.0 toward the target, the fallback generate a simple binary mask from the available anatomical-structure polygons (when TIFFs are absent) rather than outputting empty masks. The core sliding-window inference + RLE encoding logic is preserved; only the model/weights bootstrap is made robust.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import gc
import tarfile
import json
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_ROOT = Path("../input")
COMP_ROOT = INPUT_ROOT / "hubmap-kidney-segmentation"
if not COMP_ROOT.exists():
    alt = Path("/kaggle/input/hubmap-kidney-segmentation")
    if alt.exists():
        COMP_ROOT = alt
    else:
        raise FileNotFoundError(f"Competition root not found at {COMP_ROOT} or {alt}")

ss_path = COMP_ROOT / "sample_submission.csv"
assert ss_path.exists(), f"Missing sample_submission.csv at {ss_path}"




## === cell 1
try:
    import segmentation_models_pytorch as smp  # type: ignore

    HAS_SMP = True
except Exception:
    HAS_SMP = False
    import torchvision

    class _FallbackUnet(torch.nn.Module):
        """
        Fallback model with similar call semantics (returns a single-channel logit map).
        Uses torchvision's fcn_resnet50 and maps its output to 1 channel.
        """

        def __init__(self, encoder_name: str, encoder_weights=None):
            super().__init__()
            self.net = torchvision.models.segmentation.fcn_resnet50(
                weights=None, weights_backbone=None, num_classes=1
            )

        def forward(self, x):
            out = self.net(x)["out"]  # [B,1,H,W]
            return out

    class smp:  # noqa: N801
        Unet = _FallbackUnet




## === cell 2
def _safe_extract_tar(tar_path: Path, dst_dir: Path) -> bool:
    """
    Bug fix: in this environment, external weight tars may not exist.
    Make extraction optional; return False when missing instead of crashing.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not tar_path.exists():
        print(f"[warn] Weights tar not found (skipping): {tar_path}")
        return False
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=dst_dir)
    return True


FOLDS_DIR = INPUT_ROOT / "hubmap-folds-2"
MODEL_1_DIR = Path("./model_1")
MODEL_2_DIR = Path("./model_2")

MODEL_1_DIR.mkdir(parents=True, exist_ok=True)
MODEL_2_DIR.mkdir(parents=True, exist_ok=True)

if not any(MODEL_1_DIR.glob("*.pth")):
    _safe_extract_tar(FOLDS_DIR / "1024_avg_last.tar", MODEL_1_DIR)
if not any(MODEL_2_DIR.glob("*.pth")):
    _safe_extract_tar(FOLDS_DIR / "1024_512_pseudo_v1_avg.tar", MODEL_2_DIR)




## === cell 3
BATCH_SIZE = 1
NUM_WORKERS = 0
CROP_SIZE = 1024 * 4
STEP = 1024 * 2
THR = 0.5
VOTE = 0
SKIP_BACKGROUND = True
SKIP_COMMIT = True
RAMKA = 256

df_sub = pd.read_csv(COMP_ROOT / "sample_submission.csv")
assert (
    "id" in df_sub.columns
), f"Unexpected submission columns: {df_sub.columns.tolist()}"
PRED_COL = (
    "predicted"
    if "predicted" in df_sub.columns
    else ("pixels" if "pixels" in df_sub.columns else None)
)
if PRED_COL is None:
    raise ValueError(
        f"sample_submission.csv must contain 'predicted' or 'pixels', got: {df_sub.columns.tolist()}"
    )




## === cell 4
@dataclass
class ModelConfig:
    encoder: str
    weights_path: List[str]
    img_size: int
    weight_blend: float


my_models = [
    ModelConfig(
        "resnet34",
        [
            str(MODEL_1_DIR / "fold0_avg_0.9238.pth"),
            str(MODEL_1_DIR / "fold1_avg_0.9162.pth"),
            str(MODEL_1_DIR / "fold2_avg_0.9303.pth"),
            str(MODEL_1_DIR / "fold3_avg_0.9336.pth"),
            str(MODEL_1_DIR / "fold4_avg_0.9407.pth"),
        ],
        4096,
        0.4,
    ),
    ModelConfig(
        "timm-efficientnet-b3",
        [
            str(INPUT_ROOT / "hubmap5" / "exp_25_fold_0.pt"),
            str(INPUT_ROOT / "hubmap5" / "exp_24_fold_1.pt"),
            str(INPUT_ROOT / "hubmap5" / "exp_23_fold_2.pt"),
            str(INPUT_ROOT / "hubmap5" / "exp_22_fold_3.pt"),
            str(INPUT_ROOT / "hubmap5" / "exp_21_fold_4.pt"),
        ],
        1024,
        0.2,
    ),
    ModelConfig(
        "se_resnext50_32x4d",
        [
            str(MODEL_2_DIR / "fold0_avg_0.9457.pth"),
            str(MODEL_2_DIR / "fold1_avg_0.9566.pth"),
            str(MODEL_2_DIR / "fold2_avg_0.9379.pth"),
            str(MODEL_2_DIR / "fold3_avg_0.9095.pth"),
            str(MODEL_2_DIR / "fold4_avg_0.9338.pth"),
        ],
        2048,
        0.4,
    ),
]

TRAIN_IMG_SIZE = max([cfg.img_size for cfg in my_models])
TRAIN_IMG_SIZE




## === cell 5
def rle_encode_less_memory(img: np.ndarray) -> str:
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def valid_transform(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def read_window_from_bgr_image(
    img_bgr: np.ndarray, x: int, y: int, crop_size: int
) -> np.ndarray:
    crop = img_bgr[y : y + crop_size, x : x + crop_size]
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        pad_y = crop_size - crop.shape[0]
        pad_x = crop_size - crop.shape[1]
        crop = cv2.copyMakeBorder(
            crop, 0, max(0, pad_y), 0, max(0, pad_x), cv2.BORDER_REFLECT_101
        )
        crop = crop[:crop_size, :crop_size]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop


def _check_background(img_rgb, crop_size) -> bool:
    s_th = 40
    p_th = 1000 * (crop_size // 256) ** 2
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    _, ss, _ = cv2.split(hsv)
    background = False if (ss > s_th).sum() <= p_th or img_rgb.sum() <= p_th else True
    return background


def _infer_hw_from_dataset_info(image_id: str) -> Optional[Tuple[int, int]]:
    info_paths = [
        COMP_ROOT / "HuBMAP-20-dataset_information.csv",
        INPUT_ROOT / "HuBMAP-20-dataset_information.csv",
    ]
    for p in info_paths:
        if p.exists():
            info = pd.read_csv(p)
            if (
                "image_file" in info.columns
                and "width_pixels" in info.columns
                and "height_pixels" in info.columns
            ):
                stems = (
                    info["image_file"].astype(str).str.replace(".tiff", "", regex=False)
                )
                m = info.loc[stems == image_id]
                if len(m) == 1:
                    w = int(m["width_pixels"].iloc[0])
                    h = int(m["height_pixels"].iloc[0])
                    return h, w
    return None


def _render_anatomy_json_to_image(json_path: Path, h: int, w: int) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    with open(json_path, "r") as f:
        data = json.load(f)
    for feat in data:
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        ring = coords[0]
        pts = np.array(ring, dtype=np.int32).reshape(-1, 2)
        props = feat.get("properties", {}).get("classification", {})
        colorRGB = int(props.get("colorRGB", 0))
        r = (colorRGB >> 16) & 255
        g = (colorRGB >> 8) & 255
        b = (colorRGB) & 255
        cv2.fillPoly(canvas, [pts], color=(b, g, r))  # canvas is BGR for OpenCV
    return canvas


def load_test_image_or_fallback(comp_root: Path, image_id: str) -> np.ndarray:
    tiff_path = comp_root / "test" / f"{image_id}.tiff"
    if tiff_path.exists():
        img = cv2.imread(str(tiff_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read TIFF with OpenCV: {tiff_path}")
        return img

    json_path = comp_root / "test" / f"{image_id}-anatomical-structure.json"
    if not json_path.exists():
        json_path = comp_root / "test" / f"{image_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Neither TIFF nor JSON found for id={image_id} under {comp_root/'test'}"
        )

    hw = _infer_hw_from_dataset_info(image_id)
    if hw is None:
        h, w = 4096, 4096
    else:
        h, w = hw
    return _render_anatomy_json_to_image(json_path, h=h, w=w)




## === cell 6
class SingleTiffDataset(Dataset):
    """
    Rasterio-free replacement: preloads an image with OpenCV (or JSON rasterization fallback),
    then serves sliding-window crops.
    """

    def __init__(self, tiff_path, all_img_sizes, crop_size=1024, step=512):
        self.crop_size = crop_size
        self.all_img_sizes = all_img_sizes
        self.step = step

        tiff_path = str(tiff_path)
        if tiff_path.startswith("ID:"):
            image_id = tiff_path.split("ID:", 1)[1]
            self.img_bgr = load_test_image_or_fallback(COMP_ROOT, image_id)
        else:
            img = cv2.imread(tiff_path, cv2.IMREAD_COLOR)
            if img is None:
                image_id = Path(tiff_path).stem
                self.img_bgr = load_test_image_or_fallback(COMP_ROOT, image_id)
            else:
                self.img_bgr = img

        self.h, self.w = self.img_bgr.shape[:2]
        self.row_count = (
            1 + math.ceil((self.h - self.crop_size) / self.step)
            if self.h > self.crop_size
            else 1
        )
        self.col_count = (
            1 + math.ceil((self.w - self.crop_size) / self.step)
            if self.w > self.crop_size
            else 1
        )

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = max(0, self.w - self.crop_size)
        if y + self.crop_size > self.h:
            y = max(0, self.h - self.crop_size)

        img_rgb = read_window_from_bgr_image(
            self.img_bgr, x=x, y=y, crop_size=self.crop_size
        )

        transormed_imgs = {
            img_size: valid_transform(img_size)(image=img_rgb)["image"]
            for img_size in self.all_img_sizes
        }
        transormed_imgs["crop_names"] = f"{x}_{y}"
        transormed_imgs["not_background"] = _check_background(img_rgb, self.crop_size)
        return transormed_imgs




## === cell 7
def _load_state_dict_flexible(model: torch.nn.Module, weight_path: str):
    if not Path(weight_path).exists():
        raise FileNotFoundError(f"Missing weight file: {weight_path}")
    ckpt = torch.load(weight_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type at {weight_path}: {type(ckpt)}")

    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if len(unexpected) > 0:
        print(f"[warn] Unexpected keys in {Path(weight_path).name}: {len(unexpected)}")
    if len(missing) > 0:
        print(f"[warn] Missing keys in {Path(weight_path).name}: {len(missing)}")




## === cell 8
def inference(data_loader, models_with_config, crop_size):
    img_size = (data_loader.dataset.h, data_loader.dataset.w)
    mask_pred = np.zeros(img_size, dtype=np.uint8)

    for batch in tqdm(data_loader, ncols=70, leave=True):
        if SKIP_BACKGROUND is True and batch["not_background"].item() is False:
            continue

        with torch.no_grad():
            pred_total = None
            for cfg, model_group in models_with_config:
                image = batch[cfg.img_size].to(DEVICE, non_blocking=True)
                pred_group = None

                for model in model_group:
                    pred = model(image)
                    pred = pred.sigmoid()
                    if cfg.img_size != TRAIN_IMG_SIZE:
                        pred = torch.nn.functional.interpolate(
                            pred, size=TRAIN_IMG_SIZE, mode="bilinear"
                        )
                    if pred_group is None:
                        pred_group = pred
                    else:
                        pred_group += pred

                pred_group = pred_group.squeeze()
                if len(pred_group.shape) == 2:
                    pred_group = pred_group.unsqueeze(0)

                pred_group = cfg.weight_blend * (pred_group / len(model_group))
                if pred_total is None:
                    pred_total = pred_group
                else:
                    pred_total += pred_group

            pred_total = (pred_total.detach().cpu().numpy() > THR).astype(np.uint8)

            for predict_single, crop_name in zip(pred_total, batch["crop_names"]):
                x = int(crop_name.split("_")[-2])
                y = int(crop_name.split("_")[-1])

                if crop_size != TRAIN_IMG_SIZE:
                    predict_single = cv2.resize(predict_single, (crop_size, crop_size))

                predict_single[0:RAMKA] = 0
                predict_single[-RAMKA:] = 0
                predict_single[:, 0:RAMKA] = 0
                predict_single[:, -RAMKA:] = 0

                mask_pred[y : y + crop_size, x : x + crop_size] += predict_single

    mask_pred = mask_pred > VOTE
    mask_rle = rle_encode_less_memory(mask_pred.astype(np.uint8))
    del mask_pred
    gc.collect()
    return mask_rle




## === cell 9
def _all_weights_exist(cfgs: List[ModelConfig]) -> bool:
    for cfg in cfgs:
        for wp in cfg.weights_path:
            if not Path(wp).exists():
                return False
    return True


def baseline_rle_from_anatomy(image_id: str) -> str:
    """
    Minimal, deterministic baseline: use the rendered anatomical-structure polygons
    as the predicted binary mask (non-empty), then RLE encode.
    """
    img_bgr = load_test_image_or_fallback(COMP_ROOT, image_id)
    mask = (img_bgr.sum(axis=2) > 0).astype(np.uint8)
    return rle_encode_less_memory(mask)


all_img_sizes = set([cfg.img_size for cfg in my_models])
models_with_config = []

USE_MODELS = _all_weights_exist(my_models)

if USE_MODELS:
    for cfg in my_models:
        models_group = []
        for w_path in cfg.weights_path:
            model = smp.Unet(cfg.encoder, encoder_weights=None).to(DEVICE)
            _load_state_dict_flexible(model, w_path)
            model.eval()
            models_group.append(model)
        models_with_config.append((cfg, models_group))
    print("[info] Loaded models:", [(c.encoder, len(g)) for c, g in models_with_config])
else:
    missing = []
    for cfg in my_models:
        for wp in cfg.weights_path:
            if not Path(wp).exists():
                missing.append(wp)
    print(
        f"[warn] Missing weights detected ({len(missing)}). Falling back to anatomy-JSON baseline."
    )
    print("[warn] First few missing:\n" + "\n".join(missing[:10]))




## === cell 10
count_thr = 5 if SKIP_COMMIT is True else 4

test_dir = COMP_ROOT / "test"
if not test_dir.exists():
    raise FileNotFoundError(f"Missing test directory at {test_dir}")

tiff_exists_any = any(test_dir.glob("*.tiff"))

if len(df_sub) > count_thr:
    for idx, row in df_sub.iterrows():
        image_id = row["id"]

        if USE_MODELS:
            tiff_path = (
                str(test_dir / f"{image_id}.tiff")
                if tiff_exists_any
                else f"ID:{image_id}"
            )

            test_ds = SingleTiffDataset(
                tiff_path=tiff_path,
                all_img_sizes=all_img_sizes,
                crop_size=CROP_SIZE,
                step=STEP,
            )

            test_loader = DataLoader(
                dataset=test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=(DEVICE == "cuda"),
            )
            rle = inference(test_loader, models_with_config, CROP_SIZE)
        else:
            rle = baseline_rle_from_anatomy(image_id)

        df_sub.loc[idx, PRED_COL] = rle




## === cell 11
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
