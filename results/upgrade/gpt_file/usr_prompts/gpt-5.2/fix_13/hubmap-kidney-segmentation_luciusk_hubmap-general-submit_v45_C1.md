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

0.9323729005535468

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by replacing TIFF reading and window extraction with `tifffile.memmap` and numpy slicing, keeping the same tiling/prediction logic. I also fix `np.float` deprecation, ensure `torch` is imported before use, and keep the same model loading/inference/thresholding so evaluation semantics remain unchanged. Finally, I make sure we always create `submission.csv` with the exact `id,predicted` columns from `sample_submission.csv`, even if an image ends up with an empty mask.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not installed here) by adding a tiny fallback U-Net implementation that can still load the provided `.pth` weights when they match, and otherwise safely continue with randomly initialized weights (so the pipeline always runs and creates a valid submission). I also fix the `tifffile.memmap` failure by switching to a robust TIFF reader that uses `tifffile.TiffFile(...).asarray(out='memmap')` when possible and falls back to normal array loading when not memory-mappable, while keeping the same window-slicing logic. Finally, I guard model-path discovery (search under `../input/**`) so missing paths don’t crash, and ensure RLE encoding returns an empty string for an all-zero mask to avoid invalid runs.'
- What this solution (achieved 0.0) has done: 'The runtime error comes from `tifffile` needing the optional `imagecodecs` package to decode JPEG-compressed TIFFs, which isn’t installed in this environment. To keep the core tiling/inference logic unchanged but make the pipeline run end-to-end, I replace the TIFF reading with an OpenCV-based TIFF reader fallback (OpenCV can decode these TIFFs here) and only if that fails we try `tifffile`. I also keep the array axis handling identical (ensure HWC with 3 channels) so downstream resizing, HSV filtering, and model input remain the same. This move the score from 0.0 (no submission) toward the target by enabling actual inference and a valid `submission.csv` to be produced.'
- What this solution (achieved 0.0) has done: 'We need to stop `tifffile` from being called at all for JPEG-compressed TIFFs (it hard-fails without `imagecodecs`), because your current OpenCV read attempt returns `None` in this environment. The minimal fix is to add a JPEG-TIFF-capable reader fallback using Pillow (PIL is installed) before trying `tifffile`, and only use `tifffile` for non-JPEG compressions or when it can actually decode. This is score-improving from 0.0 toward the target because it unblocks real inference and ensures `submission.csv` is always created. I also add a clear file-existence check with a helpful error message to avoid silent `None` reads and to keep alignment with sample submission ids.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF reader so it never falls through to `tifffile` for JPEG-compressed TIFFs (which hard-fails without `imagecodecs`), by detecting JPEG compression via TIFF tags and forcing a PIL/OpenCV decode path instead. I also add a final safety fallback using `cv2.imdecode` from raw bytes, which often succeeds even when `cv2.imread` returns `None`, while keeping the same output array shape/axis handling. These changes are directly targeted at the current runtime error that prevents any submission from being produced (hence score 0.0), and should move the score toward the target by enabling actual model inference. The rest of the tiling, model inference, thresholding, and RLE encoding logic is left unchanged.'
- What this solution (achieved 0.0) has done: 'We fix the runtime failure in the TIFF reader by adding a robust, `imagecodecs`-free decode path that works for JPEG-compressed TIFFs: use `tifffile` only to extract the embedded JPEG tile/strip bytes, then decode those bytes with OpenCV (`cv2.imdecode`). This is a minimal change focused on unblocking end-to-end inference and producing a valid `submission.csv`, which should move the score up from 0.0 toward the target because predictions finally be generated for all test ids. We also make the dataset path resolution slightly more defensive (try both `.tiff` and `.tif`) without changing any model/inference logic, thresholds, or tiling semantics.'
- What this solution (achieved 0.0) has done: 'I fix the crash by making the TIFF reader robust for JPEG-compressed TIFFs without `imagecodecs`: instead of trying to extract/concatenate embedded JPEG strips/tiles (which often fails), we decode via Pillow first (PIL can read these TIFFs here) and only use `tifffile` for non-JPEG TIFFs or metadata checks. This change is minimal and only touches image I/O, keeping tiling, model inference, thresholding, and RLE encoding identical. I also ensure the reader normalizes channel order correctly for PIL (RGB→BGR) so the existing OpenCV HSV blank-tile logic remains consistent. With this unblocked, the notebook run end-to-end and write a valid `submission.csv`, moving the score up from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF decoding crash that prevents any submission from being generated by adding a robust, `imagecodecs`-free reader path for JPEG-compressed TIFFs: first try PIL, then try OpenCV, and if both fail, fall back to `tifffile`’s low-level JPEG byte extraction (tiles/strips) followed by `cv2.imdecode`. This change is strictly limited to image I/O and keeps the tiling, model inference, thresholding, and RLE encoding logic unchanged. I also make the JPEG-compression detection more reliable by checking the standard TIFF Compression tag, so we only take the special path when necessary. With these fixes, the notebook should run end-to-end and write a valid `submission.csv`, moving the score up from 0.0 toward your target.'
- What this solution (achieved 0.04734) has done: 'I fix the end-to-end runtime failure by making the TIFF reader robust for the remaining JPEG-compressed TIFFs that PIL/OpenCV can’t decode directly, without changing any tiling, model, thresholding, or RLE logic. The minimal fix is to add a last-resort conversion step using the already-available `tifftools` command-line utility (present in the dataset directory) to convert the problematic TIFF to a non-JPEG/deflate TIFF, then read it with `tifffile`. I also add small defensive fallbacks so that if an image still can’t be decoded, we emit an empty mask for that id rather than crashing (this turns the current 0.0 “no submission” into a valid submission and should increase the score toward the target). All I/O paths and the submission format (`id,predicted`) remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should improve real segmentation quality without changing the core tiling/inference/RLE logic. The biggest issue here is that `build_model()` falls back to a tiny U-Net when `segmentation_models_pytorch` isn’t available, which makes the loaded EfficientNet-B2 weights effectively unusable (hence near-random predictions and very low Dice). The minimal, directly score-relevant fix is to remove the dependency on `segmentation_models_pytorch` by implementing an EfficientNet-B2 encoder + U-Net–style decoder model in pure `torchvision`, so the provided `.pth` weights can load correctly and inference semantics remain the same (sigmoid + threshold + tile stitching). I also make model-file discovery prefer fold checkpoints (instead of arbitrary first 5 `.pth` files) to avoid accidentally loading unrelated weights.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 indicates the submission is effectively empty/invalid or predictions are near-all background; to move toward the 0.932 target with minimal semantic change, the most direct fix is to ensure the provided EfficientNet-B2 U-Net weights actually load into a matching architecture. I keep your tiling, normalization, thresholding, stitching, and RLE exactly as-is, but adjust the `EfficientNetB2UNet` encoder tap points to align with the standard EfficientNet feature stages used by common Unet implementations (so skip-channel shapes match the checkpoint), and I make state-dict loading robust to checkpoints that store `model`/`state_dict` keys and `module.` prefixes. Finally, I prevent the “all tiles excluded” path from silently creating all-empty masks by ensuring the blank-tile filter only skips truly blank tiles (keeping the same thresholds), so the model is actually run on informative regions.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly suggests the submission masks are effectively empty/near-empty (or not aligned), so the smallest score-relevant change is to make sure the ensemble actually runs on informative tiles rather than being over-filtered as “blank.” I keep your exact model, tiling, inference, stitching, thresholding, and RLE logic, but adjust the blank-tile exclusion to only skip truly empty tiles by removing the overly strict `(img.sum() <= p_th)` clause that can wrongly exclude real tissue after downscaling. I also add a tiny safety clamp so the vote-aggregation mask can’t overflow `uint8` before thresholding (this preserves semantics but prevents wraparound on dense overlaps). These changes should move the score up from 0.0 toward the target without altering the core approach.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import glob
import warnings
import subprocess
import re

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

warnings.filterwarnings("ignore")

try:
    import segmentation_models_pytorch as smp  # type: ignore
except Exception:
    smp = None



## === cell 1
sz = 256  # tile size at network input resolution
reduce = 4  # downscale factor applied before model; later upsample back by this factor
TH = 0.35  # threshold for positive predictions

DATA = "../input/hubmap-kidney-segmentation/test/"

MODELS = [
    f"../input/b4-256-noshifttrain/efficientnet-b2-256-FOLD-{i}-model.pth"
    for i in range(5)
]

df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b2"
shift = True
minoverlap = 300




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
    if img is None or img.size == 0 or img.max() == 0:
        return ""
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385], dtype=np.float32)
std = np.array([0.15167958, 0.23584107, 0.13146145], dtype=np.float32)

s_th = 40  # saturation blanking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for minimum number of pixels


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return array of size (N,4), where N - number of tiles,
    axis represents: x1,x2,y1,y2
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


def _ensure_hwc_3ch(arr: np.ndarray) -> np.ndarray:
    """Normalize to HWC with exactly 3 channels."""
    mm = arr
    if mm.ndim == 2:
        mm = mm[..., None]
    if mm.ndim == 3 and mm.shape[0] in (3, 4) and mm.shape[-1] not in (3, 4):
        mm = np.moveaxis(mm, 0, -1)
    if mm.shape[-1] < 3:
        mm = np.repeat(mm, 3, axis=-1)
    elif mm.shape[-1] > 3:
        mm = mm[..., :3]
    return mm


def _tiff_is_jpeg_compressed(path: str) -> bool:
    try:
        with tiff.TiffFile(path) as tif:
            page = tif.pages[0]
            try:
                comp_tag = page.tags.get("Compression", None)
                if comp_tag is not None:
                    comp_val = comp_tag.value
                    if isinstance(comp_val, (tuple, list)):
                        comp_val = comp_val[0]
                    if isinstance(comp_val, (int, np.integer)):
                        return int(comp_val) == 7
                    if isinstance(comp_val, str):
                        return "JPEG" in comp_val.upper()
            except Exception:
                pass

            comp = getattr(page, "compression", None)
            if isinstance(comp, (int, np.integer)):
                return int(comp) == 7
            return "JPEG" in str(comp).upper()
    except Exception:
        return False


def _read_tiff_pil(path: str):
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            arr = np.array(im)
        if arr is None:
            return None
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = arr[..., :3]
            arr = arr[..., ::-1].copy()  # RGB->BGR for downstream cv2 usage
        return arr
    except Exception:
        return None


def _read_tiff_opencv(path: str):
    try:
        cv_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if cv_img is not None:
            return cv_img
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        cv_img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return cv_img
    except Exception:
        return None


def _decode_jpeg_tiff_via_tifffile_cv2(path: str):
    try:
        with tiff.TiffFile(path) as tif:
            page = tif.pages[0]

            if getattr(page, "is_tiled", False):
                try:
                    tj = page.tags.get("TileJPEGInterchangeFormat", None)
                    tl = page.tags.get("TileJPEGInterchangeFormatLength", None)
                    if tj is not None and tl is not None:
                        offsets = np.array(tj.value).ravel().tolist()
                        lengths = np.array(tl.value).ravel().tolist()
                        if len(offsets) > 0 and len(offsets) == len(lengths):
                            with open(path, "rb") as f:
                                f.seek(int(offsets[0]))
                                jpeg_bytes = f.read(int(lengths[0]))
                            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                            return img
                except Exception:
                    pass

            try:
                sj = page.tags.get("JPEGInterchangeFormat", None)
                sl = page.tags.get("JPEGInterchangeFormatLength", None)
                if sj is not None and sl is not None:
                    offset = int(sj.value)
                    length = int(sl.value)
                    with open(path, "rb") as f:
                        f.seek(offset)
                        jpeg_bytes = f.read(length)
                    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    return img
            except Exception:
                pass

    except Exception:
        return None
    return None


def _resolve_tiff_path(base_dir: str, idx: str) -> str:
    p1 = os.path.join(base_dir, idx + ".tiff")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(base_dir, idx + ".tif")
    if os.path.exists(p2):
        return p2
    return p1


def _tifftools_convert_to_uncompressed(
    src_path: str, out_dir: str = "/kaggle/working"
) -> str:
    """
    Runtime fallback: rewrite problematic JPEG-compressed TIFFs into a decodeable one.
    """
    os.makedirs(out_dir, exist_ok=True)
    dst_path = os.path.join(
        out_dir, os.path.basename(src_path).replace(".tif", ".tiff")
    )
    dst_path = dst_path.replace(".tiff", "") + "__converted.tiff"
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return dst_path

    tt_local = "../input/hubmap-kidney-segmentation/tifftools"
    tifftools_cmd = tt_local if os.path.exists(tt_local) else "tifftools"
    cmd = [tifftools_cmd, "tiff_set", src_path, dst_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            return dst_path
    except Exception:
        pass
    return src_path


def open_tiff_memmap(path):
    """
    Ensure we can read JPEG-compressed TIFFs without imagecodecs.
    Order: PIL -> OpenCV -> tifffile tag extract + cv2 -> tifftools rewrite -> tifffile.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"TIFF not found: {path}")

    is_jpeg = _tiff_is_jpeg_compressed(path)

    arr = None
    if is_jpeg:
        arr = _read_tiff_pil(path)
        if arr is None:
            arr = _read_tiff_opencv(path)
        if arr is None:
            arr = _decode_jpeg_tiff_via_tifffile_cv2(path)
        if arr is None:
            conv_path = _tifftools_convert_to_uncompressed(path)
            if conv_path != path:
                try:
                    with tiff.TiffFile(conv_path) as tif:
                        arr = tif.asarray()
                except Exception:
                    arr = None
        if arr is None:
            raise ValueError(
                f"Failed to decode JPEG-compressed TIFF without imagecodecs: {path}"
            )
    else:
        try:
            with tiff.TiffFile(path) as tif:
                try:
                    arr = tif.asarray(out="memmap")
                except Exception:
                    arr = tif.asarray()
        except Exception:
            arr = None

        if arr is None:
            arr = _read_tiff_pil(path)
        if arr is None:
            arr = _read_tiff_opencv(path)
        if arr is None:
            raise ValueError(f"Failed to decode TIFF: {path}")

    mm = _ensure_hwc_3ch(arr)
    if mm.dtype != np.uint8:
        mm = np.clip(mm, 0, 255).astype(np.uint8)
    return mm


def read_window(mm, x1, x2, y1, y2):
    w = mm[x1:x2, y1:y2]
    if w.dtype != np.uint8:
        w = np.clip(w, 0, 255).astype(np.uint8)
    return w




## === cell 4
class Model_pred_shift:
    def __init__(self, models, dl, tta: bool = False, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.no_grad():
            for x, z, y in iter(self.dl):
                if (y >= 0).sum() > 0:  # exclude empty tiles
                    x = x[y >= 0].to(device)
                    z = z[y >= 0]
                    y = y[y >= 0]
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

                    py /= max(1, len(self.models))

                    py = F.interpolate(
                        py, scale_factor=reduce, mode="bilinear", align_corners=False
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    py = py.squeeze(-1).numpy()
                    z = z.numpy()

                    batch_size = len(py)
                    for i in range(batch_size):
                        yield py[i], z[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)


class HuBMAPDatasetShift(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        path = _resolve_tiff_path(DATA, idx)
        self.mm = open_tiff_memmap(path)
        self.shape = self.mm.shape[:2]  # (H,W)
        self.reduce = reduce
        self.sz = reduce * sz
        self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=minoverlap)

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        img = read_window(self.mm, x1, x2, y1, y2)

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // reduce, self.sz // reduce),
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)
        vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)

        if (s > s_th).sum() <= p_th:
            return img2tensor((img / 255.0 - mean) / std), vertices, -1
        else:
            return img2tensor((img / 255.0 - mean) / std), vertices, idx




## === cell 5
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch, 3, 1, 1)
        self.conv2 = ConvBNReLU(out_ch, out_ch, 3, 1, 1)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class EfficientNetB2UNet(nn.Module):
    def __init__(self, classes=1):
        super().__init__()
        self.encoder = torchvision.models.efficientnet_b2(weights=None)
        self.features = self.encoder.features

        self._tap_indices = [1, 2, 3, 5, 8]

        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 256)
            skips = []
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self._tap_indices:
                    skips.append(x)
            enc_ch = [t.shape[1] for t in skips]

        c1, c2, c3, c4, c5 = enc_ch  # shallow->deep
        self.center = ConvBNReLU(c5, 256, 3, 1, 1)
        self.dec4 = DecoderBlock(256, c4, 256)
        self.dec3 = DecoderBlock(256, c3, 128)
        self.dec2 = DecoderBlock(128, c2, 64)
        self.dec1 = DecoderBlock(64, c1, 32)
        self.head = nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self._tap_indices:
                skips.append(x)
        s1, s2, s3, s4, s5 = skips  # shallow->deep
        x = self.center(s5)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)


class FallbackUNet(nn.Module):
    def __init__(self, in_channels=3, classes=1, base=32):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base * 4, base * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.outc = nn.Conv2d(base * 8, classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up(x4)
        return self.outc(x)


def build_model():
    if model_name == "efficientnet-b2":
        return EfficientNetB2UNet(classes=1)
    if smp is not None:
        return smp.Unet(model_name, encoder_weights=None, classes=1)
    return FallbackUNet(in_channels=3, classes=1, base=32)


def _find_model_files(requested_paths):
    existing = [p for p in requested_paths if os.path.exists(p)]
    if existing:
        return existing

    candidates = glob.glob("../input/**/*.pth", recursive=True)
    fold_like = []
    for p in candidates:
        b = os.path.basename(p).lower()
        if ("fold" in b) and ("efficientnet" in b or "eff" in b) and ("model" in b):
            fold_like.append(p)
    fold_like = sorted(fold_like)
    if fold_like:
        return fold_like[:5]

    candidates = sorted(candidates)
    return candidates[:5]


MODEL_FILES = _find_model_files(MODELS)
print("Found model files:", MODEL_FILES if MODEL_FILES else "NONE")


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_module_prefix(sd):
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd




## === cell 6
models = []
if MODEL_FILES:
    for path in MODEL_FILES:
        raw = torch.load(path, map_location=torch.device("cpu"))
        state_dict = _strip_module_prefix(_extract_state_dict(raw))

        model = build_model()
        loaded = False
        try:
            model.load_state_dict(state_dict, strict=True)
            loaded = True
        except Exception:
            try:
                model.load_state_dict(state_dict, strict=False)
                loaded = True
            except Exception:
                loaded = False

        model.float()
        model.eval()
        model.to(device)
        models.append(model)
        print(f"Loaded: {os.path.basename(path)} (state_dict_loaded={loaded})")

    del raw, state_dict
    gc.collect()
else:
    model = build_model().to(device).eval().float()
    models = [model]
    print(
        "No .pth weights found; using randomly initialized model (submission will be low-scoring)."
    )



## === cell 7
names, preds = [], []

for _, row in df_sample.iterrows():
    idx = row["id"]
    try:
        ds = HuBMAPDatasetShift(idx)
        dl = DataLoader(
            ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0
        )
        mp = Model_pred_shift(models, dl)

        mask = np.zeros(ds.shape, dtype=np.uint16)
        for pred, vert, i in iter(mp):
            x1, x2, y1, y2 = vert
            mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint16)

        mask = (mask > 0.5).astype(np.uint8)
        rle = rle_encode_less_memory(mask)
    except Exception as e:
        print(f"[WARN] Failed on id={idx}: {repr(e)}; writing empty mask.")
        rle = ""

    names.append(idx)
    preds.append(rle)

    try:
        del mask, ds, dl
    except Exception:
        pass
    gc.collect()



## === cell 8
sub = pd.DataFrame({"id": names, "predicted": preds})

sub = df_sample[["id"]].merge(sub, on="id", how="left")
sub["predicted"] = sub["predicted"].fillna("")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))
