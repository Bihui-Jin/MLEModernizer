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

0.9269368288363607

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not available in this Kaggle environment) and add a safe fallback that still produces a valid RLE submission. I also fix the missing model weight paths by checking existence before loading; if weights are absent, the code default to predicting an empty mask per image (valid but low score). Finally, I ensure `names`/`preds` are always defined and that a `submission.csv` with the exact required columns (`id`, `predicted`) is written even if inference fails.'
- What this solution (achieved 0.03117) has done: 'Your current 0.0 score is coming from always (or almost always) writing empty masks because `segmentation_models_pytorch` weights aren’t available in this environment, so the pipeline never produces meaningful segmentations. The smallest change that legitimately moves Dice upward is to keep your existing tiling/inference/RLE logic but replace the SMP Unet dependency with a tiny local PyTorch UNet-like model and train it briefly on the provided `train.csv` masks (downsampled) so predictions are non-empty and aligned to the metric. I’m also fixing the input paths to use the provided `/kaggle/input/hubmap-kidney-segmentation/...` layout, and keeping your same `TH` thresholding and RLE encoding semantics. This should reliably produce a valid `submission.csv` and move the score upward toward your target without changing the overall approach (tile -> model -> threshold -> merge -> RLE).'
- What this solution (achieved 0.03117) has done: 'I fix the training crash by making the downsampled resize dimensions always valid (OpenCV asserts when width/height become 0) and by using the official image sizes from the dataset information CSV when decoding the RLE so masks align correctly. I also correct a key column mismatch (`train.csv` uses `predicted`, not `encoding`) which currently breaks training label loading and silently harms learning. These are minimal, score-relevant fixes that keep your existing TinyUNet + full-image downsample training and the same tile→predict→threshold→merge→RLE submission logic. The script run end-to-end and write a valid `submission.csv` with columns `id,predicted`.'
- What this solution (achieved 0.03117) has done: 'I fix the training crash by ensuring the image tensor is always channel-first (C,H,W) before feeding the model; currently the train dataset returns HWC and PyTorch interprets it as 1-channel. I do this with a minimal change inside `img2tensor`, adding an HWC→CHW conversion when needed, which preserves your existing preprocessing and model. I also make the RLE encoder robust by working on a copy of the flattened array (to avoid modifying a view) and ensuring `mask` is `uint8`, which is correctness/stability-neutral but prevents subtle issues. These fixes keep your core TinyUNet + downsampled full-image training and tile→predict→threshold→merge→RLE pipeline intact, while enabling real training so the score can move upward toward the target.'
- What this solution (achieved 0.03117) has done: 'I fix the training crash by ensuring tensors are always channel-first (C,H,W) before batching; currently some images slip through as 1-channel due to inconsistent dimension handling and OpenCV resize behavior, causing the UNet to receive 1-channel inputs. I make `img2tensor` enforce 3-channel CHW output robustly, and I also make the training dataset explicitly output CHW float32 tensors to match the model’s expected input. These changes are minimal and score-relevant because they allow the model to actually train instead of failing, while keeping your TinyUNet architecture, loss, training loop, tiling inference, thresholding, and RLE encoding semantics unchanged. The rest of the pipeline remains the same and still write a valid `submission.csv` with columns `id,predicted`.'
- What this solution (achieved 0.03117) has done: 'I fix the training-time runtime error by ensuring the downsampled training images/masks are resized to a minimum spatial size compatible with the UNet’s two max-pool layers (at least 4×4 after `reduce`), which currently can produce 1×3 tensors and crash pooling. I do this with a minimal, score-relevant change inside the training dataset (compute `new_h/new_w` with a safe lower bound) while preserving your model, loss, and training loop. I also make the inference dataset resizing use the dataset’s actual tile dimensions (not a fixed `self.sz//reduce`) to avoid subtle distortions at borders, which is a correctness improvement and should slightly help Dice. The rest of the tile→predict→threshold→merge→RLE submission pipeline remains unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 0.03117) has done: 'I fix the OpenCV resize crash by ensuring the downsampled width/height passed to `cv2.resize` are always valid positive integers (the current code can compute 0 due to edge cases / metadata mismatches). I also make the train dataset more robust by enforcing a 3-channel HWC uint8 image before resizing and by safeguarding against any unexpected TIFF layout, without changing the TinyUNet, loss, epochs, thresholding, tiling, or RLE semantics. These changes are execution-critical and should let the model actually train rather than failing, which should improve Dice from the current very low score. Finally, I keep submission writing unchanged but ensure the pipeline always completes and produces `submission.csv`.'
- What this solution (achieved 0.03117) has done: 'I fix the training crash by making the TIFF loader robust to multi-page/extra-dimension TIFFs (your failing file loads as ndim=5), collapsing it into a standard H×W×C array before preprocessing. This keeps your existing TinyUNet, loss, training loop, and inference/RLE logic unchanged, but allows the model to actually train on the provided masks instead of erroring out. I also reuse the same robust TIFF-to-HWC conversion in both the training dataset and the inference `TiffReader` to prevent similar shape issues at test time. With training now running, predictions should become meaningfully non-empty, moving Dice upward toward your target while preserving evaluation semantics.'
- What this solution (achieved 0.03117) has done: 'I fix the UNet skip-connection size mismatch that crashes training by adding minimal center-cropping/padding inside the model forward pass (this preserves the same architecture and training loop but makes it robust to odd image sizes after downsampling). I also make the training dataset resize to dimensions divisible by 4 (UNet has two pooling layers), which prevents future off-by-one issues and is consistent with your existing downsample approach. Finally, I keep inference/RLE logic unchanged but ensure the pipeline always reaches submission writing and produces a valid `submission.csv` with `id,predicted`. These changes should both unblock training (improving score from the current very low baseline) and keep evaluation semantics intact.'
- What this solution (achieved 0.0) has done: 'I fix the GPU out-of-memory during training by keeping your exact TinyUNet + BCEWithLogits training loop, but switching training to CPU only (the dataset images are still large after downsampling, and the forward pass allocates huge skip tensors on GPU). I also ensure we explicitly free big tensors each iteration to prevent memory accumulation. Inference still use GPU if available (fast), but safely fall back to CPU if CUDA runs out of memory. These changes are minimal, execution-critical, and should allow the model to actually train (instead of crashing), which should improve Dice from the current very low score toward the target.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import warnings
import json

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

try:
    from IPython.display import display
except Exception:
    display = print

warnings.filterwarnings("ignore")

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
sz = 256  # tile size at model input resolution
reduce = 4  # downsample factor from original image tiles
TH = 0.3  # threshold for positive predictions (kept)
DATA_TEST = "/kaggle/input/hubmap-kidney-segmentation/test/"
DATA_TRAIN = "/kaggle/input/hubmap-kidney-segmentation/train/"
INFO_CSV = "/kaggle/input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
TRAIN_CSV = "/kaggle/input/hubmap-kidney-segmentation/train.csv"
SAMPLE_SUB = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"

df_sample = pd.read_csv(SAMPLE_SUB)
df_train = pd.read_csv(TRAIN_CSV)
df_info = pd.read_csv(INFO_CSV)

if "encoding" not in df_train.columns:
    if "predicted" in df_train.columns:
        df_train = df_train.rename(columns={"predicted": "encoding"})
    else:
        raise KeyError(f"train.csv columns unexpected: {df_train.columns.tolist()}")

df_info = df_info.copy()
df_info["id"] = df_info["image_file"].astype(str).str.replace(".tiff", "", regex=False)
id2wh = {
    r["id"]: (int(r["width_pixels"]), int(r["height_pixels"]))
    for _, r in df_info.iterrows()
    if pd.notna(r["id"])
}

bs = 32
model_name = "efficientnet-b2"  # kept variable for compatibility (not used now)
shift = True
minoverlap = 300

names, preds = [], []




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if enc is None or (isinstance(enc, float) and np.isnan(enc)):
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
    img = (img > 0).astype(np.uint8, copy=False)
    pixels = img.T.flatten().copy()
    if pixels.size == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

s_th = 40  # saturation blanking threshold
p_th = 1000 * (sz // 256) ** 2  # minimum pixels threshold


def img2tensor(img, dtype: np.dtype = np.float32):
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[..., None]  # HWC, C=1
    if img.ndim != 3:
        raise ValueError(f"img2tensor expects 2D or 3D array, got shape {img.shape}")

    if img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
        img = np.moveaxis(img, 0, -1)  # to HWC

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    x, y = shape  # x=H, y=W
    step = max(1, window - min_overlap)

    nx = x // step + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = max(0, x - window)
    x2 = (x1 + window).clip(0, x)

    ny = y // step + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = max(0, y - window)
    y2 = (y1 + window).clip(0, y)

    slices = np.zeros((nx, ny, 4), dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


def tiff_to_hwc3_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)  # HWC
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(
            f"Unexpected tiff array ndim={arr.ndim} after squeeze/collapse, shape={arr.shape}"
        )

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        mx = float(a.max()) if a.size else 1.0
        if mx > 0:
            a = a / mx * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)

    return arr


class TiffReader:
    """
    Minimal replacement for rasterio usage.
    Provides:
      - .shape as (H,W)
      - .count as channel count
      - .read(window=(x1,x2,y1,y2)) -> uint8 array (h,w,3)
    """

    def __init__(self, path):
        self.path = path
        self._arr = tiff_to_hwc3_uint8(tiff.imread(path))
        self.shape = (int(self._arr.shape[0]), int(self._arr.shape[1]))
        self.count = int(self._arr.shape[2])

    def read(self, window):
        x1, x2, y1, y2 = window
        tile = self._arr[x1:x2, y1:y2]
        h = int(x2 - x1)
        w = int(y2 - y1)
        if tile.shape[0] != h or tile.shape[1] != w:
            out = np.zeros((h, w, 3), dtype=np.uint8)
            out[: tile.shape[0], : tile.shape[1]] = tile[:, :, :3]
            tile = out
        else:
            tile = tile[:, :, :3]
        return tile




## === cell 4
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def _center_crop_or_pad(x: torch.Tensor, target_hw):
    th, tw = int(target_hw[0]), int(target_hw[1])
    h, w = int(x.shape[-2]), int(x.shape[-1])

    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        h, w = int(x.shape[-2]), int(x.shape[-1])

    if h > th:
        dh = h - th
        top = dh // 2
        x = x[..., top : top + th, :]
    if w > tw:
        dw = w - tw
        left = dw // 2
        x = x[..., :, left : left + tw]

    return x


class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))

        u2 = self.up2(x3)
        x2m = _center_crop_or_pad(x2, u2.shape[-2:])
        u2 = torch.cat([u2, x2m], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        x1m = _center_crop_or_pad(x1, u1.shape[-2:])
        u1 = torch.cat([u1, x1m], dim=1)
        u1 = self.conv1(u1)

        return self.out(u1)




## === cell 5
class HuBMAPTrainFullDataset(Dataset):
    def __init__(self, df, img_dir, reduce=4, id2wh=None, max_items=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.reduce = int(reduce)
        self.id2wh = id2wh or {}
        if max_items is not None:
            self.df = self.df.iloc[:max_items].reset_index(drop=True)

        self.min_hw_after_reduce = 4

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        idx = self.df.loc[i, "id"]
        rle = self.df.loc[i, "encoding"]
        path = os.path.join(self.img_dir, idx + ".tiff")

        img = tiff_to_hwc3_uint8(tiff.imread(path))
        H_img, W_img = int(img.shape[0]), int(img.shape[1])

        if idx in self.id2wh:
            W_rle, H_rle = self.id2wh[idx]
        else:
            W_rle, H_rle = W_img, H_img

        mask = enc2mask([rle], (int(W_rle), int(H_rle))).astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)

        if mask.shape[0] != H_img or mask.shape[1] != W_img:
            mask = cv2.resize(mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

        if self.reduce != 1:
            new_w = int(max(self.min_hw_after_reduce, W_img // self.reduce))
            new_h = int(max(self.min_hw_after_reduce, H_img // self.reduce))

            new_w = max(self.min_hw_after_reduce, (new_w // 4) * 4)
            new_h = max(self.min_hw_after_reduce, (new_h // 4) * 4)

            new_w = max(1, new_w)
            new_h = max(1, new_h)

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        x = (img.astype(np.float32) / 255.0 - mean) / std
        x = img2tensor(x).float()
        y = torch.from_numpy(mask[None].astype(np.float32))
        return x, y


def train_one_epoch(model, loader, optim, loss_fn, train_device):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(train_device)
        y = y.to(train_device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()

        total += float(loss.detach().cpu())
        n += 1

        del x, y, logits, loss
        if train_device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return total / max(n, 1)




## === cell 6
torch.manual_seed(0)
np.random.seed(0)

model = TinyUNet(in_ch=3, out_ch=1, base=32)

train_device = torch.device("cpu")
model = model.to(train_device)

train_ds = HuBMAPTrainFullDataset(
    df_train, DATA_TRAIN, reduce=reduce, id2wh=id2wh, max_items=None
)
train_dl = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False
)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

for ep in range(3):
    l = train_one_epoch(model, train_dl, optim, loss_fn, train_device=train_device)
    print(f"epoch {ep+1}/3 - loss: {l:.4f}")

model.eval()
gc.collect()

try:
    model = model.to(device)
except Exception:
    device = torch.device("cpu")
    model = model.to(device)

if torch.cuda.is_available():
    torch.cuda.empty_cache()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2989852365.py in <cell line: 0>()
     19 
     20 for ep in range(3):
---> 21     l = train_one_epoch(model, train_dl, optim, loss_fn, train_device=train_device)
     22     print(f"epoch {ep+1}/3 - loss: {l:.4f}")
     23 

/tmp/ipykernel_55/2155508793.py in train_one_epoch(model, loader, optim, loss_fn, train_device)
     57     total = 0.0
     58     n = 0
---> 59     for x, y in loader:
     60         x = x.to(train_device)
     61         y = y.to(train_device)

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

/tmp/ipykernel_55/2155508793.py in __getitem__(self, i)
     18         path = os.path.join(self.img_dir, idx + ".tiff")
     19 
---> 20         img = tiff_to_hwc3_uint8(tiff.imread(path))
     21         H_img, W_img = int(img.shape[0]), int(img.shape[1])
     22 

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in imread(files, selection, aszarr, key, series, level, squeeze, maxworkers, buffersize, mode, name, offset, size, pattern, axesorder, categories, imread, imreadargs, sort, container, chunkshape, chunkdtype, axestiled, ioworkers, chunkmode, fillvalue, zattrs, multiscales, omexml, out, out_inplace, _multifile, _useframes, **kwargs)
   1236 
   1237                     return zarr_selection(store, selection, out=out)
-> 1238                 return tif.asarray(
   1239                     key=key,
   1240                     series=series,

/usr/local/lib/python3.11/dist-packages/tifffile/tifffile.py in asarray(self, key, series, level, squeeze, out, maxworkers, buffersize)
   4540             if page0 is None:
   4541                 raise ValueError('page is None')
-> 4542             result = page0.asarray(
   4543                 out=out, maxworkers=maxworkers, buffersize=buffersize
   4544             )

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

## === cell 7
if shift:

    class HuBMAPDataset(Dataset):
        def __init__(self, idx, sz=sz, reduce=reduce):
            self.data = TiffReader(os.path.join(DATA_TEST, idx + ".tiff"))
            self.shape = self.data.shape  # (H,W)
            self.reduce = reduce
            self.sz = reduce * sz  # tile size at full resolution
            self.mask_grid = make_grid(
                self.shape, window=self.sz, min_overlap=minoverlap
            )

        def __len__(self):
            return len(self.mask_grid)

        def __getitem__(self, idx):
            x1, x2, y1, y2 = self.mask_grid[idx]
            img = self.data.read((x1, x2, y1, y2))  # uint8 HxWx3

            if self.reduce != 1:
                h = max(1, (x2 - x1) // self.reduce)
                w = max(1, (y2 - y1) // self.reduce)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

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
                        x = x[y >= 0].to(device)
                        z = z[y >= 0]
                        y = y[y >= 0]
                        if self.half:
                            x = x.half()

                        py = None
                        for m in self.models:
                            p = m(x)
                            p = torch.sigmoid(p).detach()
                            py = p if py is None else (py + p)

                        if self.tta:
                            flips = [[-1], [-2], [-2, -1]]
                            for f in flips:
                                xf = torch.flip(x, f)
                                for m in self.models:
                                    p = m(xf)
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

                        batch_size = len(py)
                        for i in range(batch_size):
                            yield py[i], z[i], y[i]

        def __len__(self):
            return len(self.dl.dataset)

    try:
        from tqdm.notebook import tqdm as tqdm_nb

        _tqdm = tqdm_nb
    except Exception:
        from tqdm import tqdm as tqdm_std

        _tqdm = tqdm_std

    models = [model]

    for _, row in _tqdm(df_sample.iterrows(), total=len(df_sample)):
        idx = row["id"]
        try:
            ds = HuBMAPDataset(idx)
            dl = DataLoader(
                ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
            )
            mp = Model_pred(models, dl)

            mask = np.zeros(ds.shape, dtype=np.uint8)
            for pred, vert, i in iter(mp):
                x1, x2, y1, y2 = vert
                mask[int(x1) : int(x2), int(y1) : int(y2)] += (pred > TH).astype(
                    np.uint8
                )

            mask = (mask > 0.5).astype(np.uint8)
            rle = rle_encode_less_memory(mask)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _old_device = device
            device = torch.device("cpu")
            try:
                model_cpu = models[0].to(device)
                models = [model_cpu]
                ds = HuBMAPDataset(idx)
                dl = DataLoader(
                    ds,
                    batch_size=max(1, bs // 4),
                    pin_memory=False,
                    shuffle=False,
                    num_workers=0,
                )
                mp = Model_pred(models, dl)

                mask = np.zeros(ds.shape, dtype=np.uint8)
                for pred, vert, i in iter(mp):
                    x1, x2, y1, y2 = vert
                    mask[int(x1) : int(x2), int(y1) : int(y2)] += (pred > TH).astype(
                        np.uint8
                    )

                mask = (mask > 0.5).astype(np.uint8)
                rle = rle_encode_less_memory(mask)
            except Exception:
                rle = ""
            finally:
                device = _old_device
                try:
                    models = [model.to(device)]
                except Exception:
                    models = [model]
        except Exception:
            rle = ""
        finally:
            names.append(idx)
            preds.append(rle)
            try:
                del ds, dl, mask
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



## === cell 8
df = pd.DataFrame({"id": names, "predicted": preds})
df = df.set_index("id").reindex(df_sample["id"]).reset_index()
df["predicted"] = df["predicted"].fillna("")
df.to_csv("submission.csv", index=False)

display(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Submission columns:", df.columns.tolist())
print("Non-empty predictions:", int((df["predicted"].astype(str).str.len() > 0).sum()))
