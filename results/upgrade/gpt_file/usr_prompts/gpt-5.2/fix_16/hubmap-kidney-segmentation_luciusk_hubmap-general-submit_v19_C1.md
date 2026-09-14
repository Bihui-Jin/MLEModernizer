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

0.9274917681347176

# 6. Current score

0.03226

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by reading the TIFFs with `tifffile` and using NumPy slicing to extract windows, keeping the exact tiling/blank-tile filtering logic intact. I also fix runtime errors caused by deprecated `np.float`, missing imports due to the first-cell crash, and replace deprecated `F.upsample` with `F.interpolate` (same semantics) for stability. Finally, I ensure the submission is always created with exactly the sample submission’s IDs (3 rows) and correct `id,predicted` columns, even if an image read fails, so Kaggle accepts the CSV.'
- What this solution (achieved 0.03226) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not installed) by adding a lightweight fallback `Unet` implementation that keeps the same call pattern (`HuBMAP().forward -> logits`), so inference can run end-to-end. I also fix the missing model-weight paths by conditionally loading weights only when they exist; if none exist, the code still generate a valid `submission.csv` (but this likely not meet the target score because it’s effectively untrained). Finally, I add small safety fixes for device/half usage and make sure the submission uses exactly the sample submission IDs and required columns.'
- What this solution (achieved 0.03226) has done: 'Your score is low because the code is effectively running with random/untrained weights (the referenced checkpoint paths don’t exist in this environment), so the generated masks are mostly noise. To move toward the target with minimal semantic change, I keep the same tiling/inference/RLE pipeline but (1) load the available pretrained model weights shipped with the competition dataset (the common `resnet34` checkpoint), (2) build the exact same UNet architecture those weights expect (no fallback unless the file is missing), and (3) use mild test-time augmentation already supported by your wrapper (horizontal/vertical flips) to lift Dice without changing the core approach. These changes should substantially increase the score and stay within Kaggle constraints while still producing a valid `submission.csv`.'
- What this solution (achieved 0.03226) has done: 'Your current score is low because the intended pretrained checkpoint path doesn’t exist, so the model runs with random weights; the smallest change that moves score toward your target is to (1) automatically discover and load any `.pth` checkpoints shipped under the dataset’s `ckpts/` folder, and (2) ensure the architecture matches those checkpoints by requiring `segmentation_models_pytorch`’s ResNet34 UNet when a checkpoint is found (fallback only if no checkpoint exists). I’m also keeping your exact tiling, blank-tile filtering, upsampling, TTA logic, thresholding, and RLE pipeline unchanged, so evaluation semantics stay the same. Finally, I keep the submission IDs aligned to `sample_submission.csv` exactly as before so the CSV is always valid.'
- What this solution (achieved 0.03226) has done: 'Your pipeline is producing a valid CSV, but your Kaggle score is “not yielded” because the notebook likely fails to run in the Kaggle internet-off environment when trying to `pip install segmentation-models-pytorch`, so no submission is actually generated. I make the smallest change to remove the runtime dependency on online installs: use `torchvision`’s built-in `resnet34` encoder to build a SMP-compatible UNet only when checkpoints are found, otherwise keep your existing fallback UNet. I also make checkpoint loading tolerant to common key mismatches (while still preferring strict loading) so that “found checkpoint but won’t load” doesn’t crash and prevent `submission.csv` creation. These changes keep your tiling, preprocessing, TTA, thresholding, and RLE exactly the same, but should move your score upward whenever usable checkpoints exist locally.'
- What this solution (achieved 0.03226) has done: 'Your score is low because the current inference likely still uses randomly initialized weights (no compatible pretrained checkpoint is being found/loaded), so predictions are near-random. The smallest meaningful change toward your target is to (1) correctly discover the *competition-provided* fold checkpoints that live under `../input/hubmap-kidney-segmentation/models/` (common in this dataset) in addition to `ckpts/`, and (2) make loading more robust to common wrapper key prefixes while keeping the exact same tiling, TTA, thresholding, and RLE pipeline. I also keep the architecture selection the same, but when SMP is unavailable I ensure we only attempt to load checkpoints that match our in-notebook ResNet34-UNet keys (otherwise they “load” mostly mismatched and behave untrained). These changes should materially increase Dice while preserving core logic and producing the same valid `submission.csv`.'
- What this solution (achieved 0.03226) has done: 'Your score gap is large (0.03226 → 0.92749), which strongly suggests inference is effectively untrained or the checkpoint load is not actually matching the model’s parameter names, so predictions are near-random. I keep your tiling/TTA/threshold/RLE pipeline unchanged, but make checkpoint loading much more compatible by (1) trying both “raw keys” and “cnn_model.*” prefixed keys, and (2) preferring checkpoints that look like real HuBMAP UNet fold/best weights while skipping obviously incompatible ones. I also automatically enable the SMP Unet only when the checkpoint appears to have SMP-style keys (so we don’t silently load most weights non-strictly into the wrong architecture). These are minimal, directly score-relevant changes that should move Dice sharply upward while still producing a valid `submission.csv`.'
- What this solution (achieved 0.03226) has done: 'Your Dice is extremely low because the notebook is still effectively producing near-empty or near-random masks: either no compatible checkpoints are being loaded, or the loaded checkpoint doesn’t match the exact architecture (SMP vs custom ResNet34-UNet) so most weights are missing and the model behaves almost untrained. I keep your tiling/TTA/threshold/RLE pipeline unchanged, but make checkpoint loading *architecture-aware*: if a checkpoint has SMP-style keys we build SMP Unet and load strictly (or near-strict), otherwise we build the custom ResNet34-UNet and require a very high key match ratio (so we don’t “accept” mostly-mismatched loads). Finally, I ensure we ensemble only the best few truly-compatible checkpoints (to move Dice up reliably toward your target without changing inference semantics), while still always writing a valid `submission.csv`.'
- What this solution (achieved 0.03226) has done: 'Your Dice is extremely low because inference is still effectively untrained: in this environment `segmentation_models_pytorch` is not available, so SMP checkpoints (which are the common competition-provided weights) can’t be loaded into your custom ResNet34-UNet, and your loader may silently accept a mostly-mismatched non-strict load. I keep your tiling, blank-tile filtering, TTA, thresholding, and RLE exactly the same, but add a tiny SMP-compatible UNet wrapper (same forward semantics) so SMP-style checkpoints can actually be used without installing extra packages. I also make checkpoint loading more deterministic/compatible by (1) scoring checkpoints by “key match ratio” against both architectures and (2) only ensembling truly compatible checkpoints, which should move Dice sharply upward toward the target. The script still always write a valid `submission.csv` with the sample submission’s IDs and required column names.'
- What this solution (achieved 0.03226) has done: 'Your Dice is extremely low because—even though the pipeline runs and writes a valid CSV—the model is still effectively behaving close to untrained due to incompatible checkpoint loading (too-strict/near-strict filtering + key-shape mismatches between SMP-style checkpoints and the current “compat” model). I keep your exact tiling, blank-tile filtering, TTA, thresholding, and RLE logic unchanged, but make checkpoint selection/loading more architecture-aware and robust: we score each checkpoint against both candidate architectures using both key-match and tensor-shape match, then pick the best and load it. This is a minimal, directly score-relevant change that should let you actually use the competition-provided pretrained weights and move Dice sharply upward toward your target. The submission format/ID alignment stays exactly as your sample submission.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import warnings

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)


def _try_import_smp_only():
    try:
        import segmentation_models_pytorch as smp  # noqa: F401

        return True
    except Exception:
        return False


HAS_SMP = _try_import_smp_only()
if HAS_SMP:
    import segmentation_models_pytorch as smp  # type: ignore
else:
    smp = None



## === cell 1
sz = 256  # the size of tiles (at reduced resolution)
reduce = 4  # reduce the original images by 4 times
TH = 0.35  # threshold for positive predictions

DATA_CANDIDATES = [
    "../input/hubmap-kidney-segmentation",  # expected location for .tiff files
    "../input/hubmap-kidney-segmentation/test",  # sometimes contains .tiff in other mirrors
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/test",
    "../input",
]

CKPT_DIRS = [
    "../input/hubmap-kidney-segmentation/ckpts",
    "../input/hubmap-kidney-segmentation/models",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/ckpts",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation/models",
    "../input/ckpts",
    "../input/models",
]

df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
if "img" in df_sample.columns:
    ID_COL = "img"
elif "id" in df_sample.columns:
    ID_COL = "id"
else:
    raise ValueError(
        f"Cannot infer ID column from sample_submission columns: {df_sample.columns.tolist()}"
    )

if "pixels" in df_sample.columns:
    PRED_COL = "pixels"
elif "predicted" in df_sample.columns:
    PRED_COL = "predicted"
else:
    raise ValueError(
        f"Cannot infer prediction column from sample_submission columns: {df_sample.columns.tolist()}"
    )

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, float) and np.isnan(enc):
            continue
        s = enc.split()
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
    pixels = img.T.flatten()
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


def _find_tiff_path(idx: str):
    fns = [idx + ".tiff", idx + ".tif"]
    rels = [
        "",  # base/{id}.tiff
        "test",  # base/test/{id}.tiff
        os.path.join("test", "test"),  # base/test/test/{id}.tiff (mirrored datasets)
        "hubmap-kidney-segmentation",
        os.path.join("hubmap-kidney-segmentation", "test"),
        os.path.join("hubmap-kidney-segmentation", "test", "test"),
    ]
    for base in DATA_CANDIDATES:
        for rel in rels:
            for fn in fns:
                fp = os.path.join(base, rel, fn) if rel else os.path.join(base, fn)
                if os.path.exists(fp):
                    return fp
    return None


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.reduce = reduce
        self.sz = reduce * sz  # tile size at full resolution

        fp = _find_tiff_path(idx)
        if fp is None:
            raise FileNotFoundError(
                f"Could not find TIFF for id={idx} in {DATA_CANDIDATES}"
            )

        img = tiff.imread(fp)

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            if img.shape[0] in (3, 4) and img.shape[1] > 32 and img.shape[2] > 32:
                img = np.moveaxis(img[:3], 0, -1)
            if img.shape[-1] > 3:
                img = img[..., :3]
        else:
            raise ValueError(f"Unexpected tiff shape for {idx}: {img.shape}")

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8, copy=False)

        self.full_img = img
        self.shape = self.full_img.shape[:2]  # (H,W)

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
        img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = self.full_img[
            p00:p01, p10:p11, :
        ]

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // reduce, self.sz // reduce),
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img2tensor((img / 255.0 - mean) / std), -1
        else:
            return img2tensor((img / 255.0 - mean) / std), idx




## === cell 4
class Model_pred:
    def __init__(self, models, dl, tta: bool = False, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.no_grad():
            for x, y in iter(self.dl):
                if (y >= 0).sum() > 0:  # exclude empty tiles
                    x = x[y >= 0].to(device)
                    y = y[y >= 0]
                    if self.half:
                        x = x.half()

                    total = None
                    denom = 0

                    for model in self.models:
                        p = torch.sigmoid(model(x)).detach()
                        total = p if total is None else (total + p)
                        denom += 1

                    if self.tta:
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            for model in self.models:
                                p = model(xf)
                                p = torch.flip(p, f)
                                p = torch.sigmoid(p).detach()
                                total += p
                                denom += 1

                    py = total / float(denom)

                    py = F.interpolate(
                        py, scale_factor=reduce, mode="bilinear", align_corners=False
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    for i in range(len(py)):
                        yield py[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 5
from torchvision.models import resnet34


class _DoubleConv(nn.Module):
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


class _ResNet34Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet34(weights=None)
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))  # 64, /2
        x1 = self.layer1(self.maxpool(x0))  # 64, /4
        x2 = self.layer2(x1)  # 128, /8
        x3 = self.layer3(x2)  # 256, /16
        x4 = self.layer4(x3)  # 512, /32
        return x0, x1, x2, x3, x4


class _UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffX != 0 or diffY != 0:
            x = F.pad(
                x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class _ResNet34UNet(nn.Module):
    def __init__(self, classes=1):
        super().__init__()
        self.encoder = _ResNet34Encoder()
        self.center = _DoubleConv(512, 512)
        self.up4 = _UpBlock(512, 256, 256)
        self.up3 = _UpBlock(256, 128, 128)
        self.up2 = _UpBlock(128, 64, 64)
        self.up1 = _UpBlock(64, 64, 32)
        self.final_up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.outc = nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        x = self.center(x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        x = self.final_up(x)
        return self.outc(x)


class _SMPResNet34UNetCompat(nn.Module):
    """
    Minimal structure + naming to match common SMP-Unet checkpoints:
      - encoder.*
      - decoder.blocks.{i}.*
      - segmentation_head.*
    """

    def __init__(self, classes=1):
        super().__init__()
        self.encoder = _ResNet34Encoder()

        self.decoder = nn.Module()
        self.decoder.center = _DoubleConv(512, 512)
        self.decoder.blocks = nn.ModuleList(
            [
                _UpBlock(512, 256, 256),
                _UpBlock(256, 128, 128),
                _UpBlock(128, 64, 64),
                _UpBlock(64, 64, 32),
            ]
        )

        self.segmentation_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, classes, kernel_size=1),
        )

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        x = self.decoder.center(x4)
        x = self.decoder.blocks[0](x, x3)
        x = self.decoder.blocks[1](x, x2)
        x = self.decoder.blocks[2](x, x1)
        x = self.decoder.blocks[3](x, x0)
        return self.segmentation_head(x)


class HuBMAP(nn.Module):
    def __init__(self, arch: str = "custom"):
        super(HuBMAP, self).__init__()
        if arch == "smp" and HAS_SMP:
            self.cnn_model = smp.Unet("resnet34", encoder_weights=None, classes=1)
        elif arch == "smp_compat":
            self.cnn_model = _SMPResNet34UNetCompat(classes=1)
        else:
            self.cnn_model = _ResNet34UNet(classes=1)

    def forward(self, imgs):
        return self.cnn_model(imgs)




## === cell 6
def _discover_ckpts(ckpt_dirs, extra_roots=None):
    exts = (".pth", ".pt", ".bin")
    paths = []

    for ckpt_dir in ckpt_dirs:
        if not os.path.isdir(ckpt_dir):
            continue
        for root, _, files in os.walk(ckpt_dir):
            for fn in files:
                if fn.lower().endswith(exts):
                    paths.append(os.path.join(root, fn))

    if extra_roots is not None:
        for base in extra_roots:
            if not os.path.isdir(base):
                continue
            for root, _, files in os.walk(base):
                for fn in files:
                    if fn.lower().endswith(exts):
                        paths.append(os.path.join(root, fn))

    return sorted(list(dict.fromkeys(paths)))


def _extract_state_dict(obj):
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unexpected checkpoint type: {type(obj)}")


def _strip_known_prefixes(sd):
    out = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model.", "net.", "seg_model.", "unet."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


def _maybe_add_prefix(sd, prefix: str):
    return {prefix + k: v for k, v in sd.items()}


def _looks_like_smp(sd_keys):
    ks = list(sd_keys)
    has_decoder = any(k.startswith("decoder.") for k in ks)
    has_seghead = any(k.startswith("segmentation_head.") for k in ks)
    has_encoder = any(k.startswith("encoder.") for k in ks)
    return has_decoder or has_seghead or has_encoder


def _shape_match_ratio_for_model(model: nn.Module, sd_try: dict) -> float:
    msd = model.state_dict()
    inter = 0
    shape_ok = 0
    for k, tv in sd_try.items():
        if k in msd:
            inter += 1
            try:
                if tuple(msd[k].shape) == tuple(tv.shape):
                    shape_ok += 1
            except Exception:
                pass
    denom = max(1, len(msd))
    return (shape_ok / float(denom)) + 0.25 * (inter / float(denom))


EXTRA_CKPT_ROOTS = [
    "../input/hubmap-kidney-segmentation",
    "../input/hubmap-kidney-segmentation/hubmap-kidney-segmentation",
    "../input",
]

MODELS = _discover_ckpts(CKPT_DIRS, extra_roots=EXTRA_CKPT_ROOTS)
existing_paths = [p for p in MODELS if os.path.exists(p)]

MAX_MODELS_TO_USE = 3

MIN_SHAPE_MATCH_RATIO = 0.75


def _ckpt_priority(p: str) -> int:
    pl = os.path.basename(p).lower()
    score = 1000
    if "best" in pl:
        score -= 200
    if "fold" in pl:
        score -= 150
    if "hubmap" in pl or "kidney" in pl:
        score -= 50
    if "resnet34" in pl:
        score -= 30
    if "unet" in pl:
        score -= 20
    if "fp16" in pl or "half" in pl:
        score += 10
    return score


existing_paths = sorted(existing_paths, key=lambda p: (_ckpt_priority(p), p))

models = []
if len(existing_paths) == 0:
    warnings.warn(
        "No pretrained checkpoint found; inference will run with randomly initialized weights, "
        "which will likely yield a very low Dice score."
    )
    model = HuBMAP(arch="custom").to(device)
    model.float().eval()
    models = [model]
else:
    for path in existing_paths:
        ckpt = torch.load(path, map_location=torch.device("cpu"))
        raw_sd = _extract_state_dict(ckpt)
        sd0 = _strip_known_prefixes(raw_sd)

        cand_sds = []
        cand_sds.append(sd0)

        sd1 = {}
        for k, v in sd0.items():
            nk = k
            if nk.startswith("cnn_model."):
                nk = nk[len("cnn_model.") :]
            sd1[nk] = v
        cand_sds.append(sd1)
        cand_sds.append(_maybe_add_prefix(sd1, "cnn_model."))

        looks_smp = _looks_like_smp(sd0.keys())

        if looks_smp and HAS_SMP:
            archs = ["smp"]
        elif looks_smp and (not HAS_SMP):
            archs = ["smp_compat"]
        else:
            archs = ["custom"]

        best_arch = archs[0]
        best_sd = cand_sds[0]
        best_score = -1.0
        for arch in archs:
            tmp_model = HuBMAP(arch=arch)
            for sd_try in cand_sds:
                score = _shape_match_ratio_for_model(tmp_model, sd_try)
                if score > best_score:
                    best_score = score
                    best_arch = arch
                    best_sd = sd_try
            del tmp_model

        if best_score < MIN_SHAPE_MATCH_RATIO:
            del ckpt, raw_sd, cand_sds
            gc.collect()
            continue

        model = HuBMAP(arch=best_arch)

        loaded = False
        try:
            model.load_state_dict(best_sd, strict=True)
            loaded = True
        except Exception:
            _ = model.load_state_dict(best_sd, strict=False)
            loaded = True

        if not loaded:
            del ckpt, raw_sd, cand_sds, model
            gc.collect()
            continue

        model.float().eval().to(device)
        models.append(model)

        del ckpt, raw_sd, cand_sds
        gc.collect()

        if len(models) >= MAX_MODELS_TO_USE:
            break

if len(models) == 0:
    warnings.warn(
        "Checkpoints were found but none were compatible with the current architecture; "
        "falling back to random weights."
    )
    model = HuBMAP(arch="custom").to(device)
    model.float().eval()
    models = [model]

gc.collect()



## === cell 7
names, preds = [], []

use_tta = True

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row[ID_COL]
    rle = ""
    try:
        ds = HuBMAPDataset(idx)
        dl = DataLoader(
            ds,
            batch_size=bs,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            num_workers=0,
        )
        mp = Model_pred(models, dl, tta=use_tta)

        mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
        for p, i in iter(mp):
            mask[i.item()] = p.squeeze(-1) > TH

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

        rle = rle_encode_less_memory(mask.numpy().astype(np.uint8))
    except Exception:
        rle = ""

    names.append(idx)
    preds.append(rle)

    if "ds" in locals():
        del ds
    if "dl" in locals():
        del dl
    if "mask" in locals():
        del mask
    gc.collect()



## === cell 8
sub = pd.DataFrame({ID_COL: names, PRED_COL: preds})

sub = df_sample[[ID_COL]].merge(sub, on=ID_COL, how="left")
sub[PRED_COL] = sub[PRED_COL].fillna("")

sub.to_csv("submission.csv", index=False)

print(sub.head())
print(f"Saved submission.csv with {len(sub)} rows")
print(f"ID_COL={ID_COL}, PRED_COL={PRED_COL}")
print(f"segmentation_models_pytorch available: {HAS_SMP}")
print(f"Discovered checkpoints in {CKPT_DIRS} (+extra roots): {len(MODELS)}")
print(f"Loaded models: {len(models)} (MAX_MODELS_TO_USE={MAX_MODELS_TO_USE})")
print(f"Using TTA: {use_tta}")
if len(existing_paths) > 0:
    print("Top checkpoint candidates (first 10):", existing_paths[:10])
