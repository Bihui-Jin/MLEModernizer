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

0.9273109418623416

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by replacing the TIFF window reading with `tifffile.imread` and a pure-numpy crop/pad pipeline, while keeping the same tiling/reduce/threshold logic and model unchanged. I also fix the import/cell-order issues (your script starts at cell 0, causing `torch`, `nn`, and `rasterio` to be undefined when later cells run) by consolidating required imports at the top and renumbering cells from 1. Finally, I make the RLE helpers compatible with modern NumPy (replace deprecated `np.float`) and ensure the produced `submission.csv` has exactly the sample submission’s rows/ids so Kaggle accepts it.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not installed) and the missing external weight files by swapping in a lightweight U-Net defined in pure PyTorch and loading weights only if present; otherwise it still run and produce a valid submission. I also fix TIFF reading for JPEG-compressed tiles by using OpenCV’s TIFF decoder (which works without `imagecodecs`) as a fallback when `tifffile` raises the compression error. Finally, I keep your existing tiling/reduce/threshold/RLE submission logic intact, and ensure the output CSV matches `sample_submission.csv`’s `id,predicted` schema so Kaggle accepts it (score move from 0.0 to non-zero by producing non-empty predictions).'
- What this solution (achieved 0.0) has done: 'I fix the TIFF loading crash by avoiding full-image decoding in OpenCV (it hits the max-pixels safety check) and instead reading only the needed tiles directly from the TIFF using `tifffile.TiffFile(...).asarray(region=...)`, which does not require `imagecodecs` for these files and keeps your tiling logic the same. I keep your model, thresholding, and RLE encoding intact, only changing the dataset’s reading backend to be tile-based and memory-safe. I also make the submission writing robust to any unexpected missing predictions while preserving the sample submission’s `id,predicted` schema. This should turn the current 0.0 (runtime failure) into a valid non-zero score by producing real masks.'
- What this solution (achieved 0.0) has done: 'I fix the runtime error by replacing the unsupported `TiffPage.asarray(region=...)` call with a safe, tile-based reader that works in this Kaggle environment: we read the full TIFF once with `tifffile.imread()` (and fall back to OpenCV if needed), then perform the exact same crop/pad/resize tiling logic as before. This keeps the model, thresholding, and RLE encoding unchanged while unblocking end-to-end execution so a valid `submission.csv` is always produced. I also ensure channel ordering is consistent (RGB/BGR handling) so the existing HSV blank-tile filter continues to behave as intended. These changes should move the current score from 0.0 (runtime failure) to a valid non-zero score toward the target by actually generating predictions.'
- What this solution (achieved 0.0) has done: 'I fix the crash by removing the OpenCV full-image TIFF fallback (it hits OpenCV’s max-pixels guard) and instead implement a memory-safe tile reader using `tifffile.memmap`, which avoids decoding the entire image at once while preserving your exact tiling/crop/pad/reduce logic. I keep the model, thresholding, and RLE encoding unchanged, only swapping the image backend inside the dataset to read per-tile slices. This unblock end-to-end inference so `submission.csv` is always written and should move the score from 0.0 (no valid run) to a non-zero score toward the target. I also add small robustness for rare TIFF channel layouts without changing semantics.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF loading crash by replacing the current `tifffile.memmap/imread` fallback (which fails on JPEG-compressed TIFFs without `imagecodecs`) with a safe OpenCV-based tile reader that reads the image via `cv2.imreadmulti` and then slices tiles, keeping the same tiling/reduce/threshold/model inference/RLE pipeline. This change is strictly to unblock end-to-end execution and produce non-empty predictions (so the score moves up from 0.0 toward the target), without altering your model architecture or post-processing logic. I also keep channel handling consistent (BGR as in your current code) so the blank-tile HSV filter semantics remain unchanged. The script still write `submission.csv` matching `sample_submission.csv` (`id,predicted`).'
- What this solution (achieved 0.0) has done: 'I fix the OpenCV TIFF loading crash by removing full-image decoding (which triggers OpenCV’s `CV_IO_MAX_IMAGE_PIXELS` guard on these huge slides) and switching the dataset to a memory-safe, tile-based reader using `tifffile.TiffFile(...).pages[0].asarray(...)` with the page’s native `tile=(row,col)` access. This keeps your existing tiling, padding, reduce/resize, blank-tile filtering, model inference, thresholding, and RLE encoding logic unchanged, but unblocks end-to-end inference so you get a non-empty `submission.csv` and a non-zero Dice score. I also add a small fallback to read tiles via `page.asarray()` only if tiling isn’t available, and ensure the produced CSV matches `sample_submission.csv` (`id,predicted`) exactly.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF reading crash by removing the fallback path that decodes the full page via `page.asarray()` (which triggers the missing `imagecodecs` JPEG decompressor) and replacing it with a safe tile/strip region reader using `tifffile`’s built-in `aszarr()` + array slicing (this works without `imagecodecs` in this dataset and avoids full-image decode). This change is strictly to unblock end-to-end inference and produce a valid `submission.csv`, keeping your tiling, resizing, model, thresholding, and RLE encoding logic unchanged. I also renumber cells to start at 1 (Kaggle/Notebook compatibility) while preserving the original cell order/content. No score-tuning changes (threshold/model) are introduced beyond enabling the pipeline to actually generate non-empty predictions instead of crashing (current score 0.0).'
- What this solution (achieved 0.0) has done: 'I fix the crash in TIFF region reading by removing the `page.aszarr()` fallback (it requires the unavailable `zarr` package) and replacing it with a pure-`tifffile` fallback that reads the smallest possible region via `page.asarray(key=...)` (and, if needed, the full page as a last resort). This keeps your existing dataset tiling/padding/reduce logic, model inference, thresholding, and RLE encoding unchanged, but unblocks end-to-end inference so `submission.csv` is always produced. I also correct the type annotation that references a non-existent `tiff.tifffile` attribute to avoid potential runtime/name issues. With this, the pipeline should run and produce non-empty predictions, moving the score up from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF reading failure by ensuring we never fall back to `page.asarray()` on JPEG-compressed TIFFs (which requires the missing `imagecodecs` package). Instead, we read only the required tile/region using `tifffile`’s segment-level decoding (`page.segments`) and slice out the requested window, preserving your existing tile/pad/reduce/model/threshold/RLE logic. I also make the reader robust to either tiled or stripped TIFF storage while keeping BGR ordering consistent for the HSV blank-tile filter. This should unblock end-to-end inference so a valid `submission.csv` is produced and the score moves from 0.0 (crash) to a non-zero value toward the target.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with generating essentially empty/near-empty masks, which can happen because the inference tiles are never normalized with the training mean/std (even though `mean`/`std` are defined) and because the TIFF tile reader is channel-swapping twice (returning BGR but then reversing again), breaking the HSV blank-tile filter and starving the model of valid tiles. I make two minimal, directly score-relevant fixes: (1) correct the TIFF reader to return true BGR without an extra RGB↔BGR flip, and (2) apply the existing mean/std normalization in `__getitem__` exactly as intended (no architectural/training changes). These keep your tiling, model, thresholding, and RLE submission logic unchanged while making predictions non-trivial and moving the Dice up from 0.0 toward your target. The script still run end-to-end and write a valid `submission.csv` with `id,predicted`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with producing invalid/degenerate masks (e.g., masks that are always empty or always full) even though the notebook runs. To move the Dice score upward toward your target while preserving the exact model/tiling/inference core logic, I make two minimal, directly score-relevant fixes: (1) ensure the RLE encoder does not forcibly zero-out the first/last pixel (this can delete true positives and also breaks the standard Kaggle RLE convention), and (2) make the empty-mask case output an empty string (as expected by this competition) instead of an empty/incorrect run list. Everything else (model, thresholding, tile reading, normalization, reduction, stitching) stays the same, and the script still writes a valid `submission.csv` matching `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with producing essentially empty masks (all-empty RLEs), which can happen because the blank-tile filter is too aggressive and silently skips most/all tiles, and because the submission loop never fills skipped tiles (they remain zeros). To move the Dice upward toward the target while keeping your model/tiling/inference logic intact, I make two minimal, score-relevant adjustments: (1) relax the blank-tile gating slightly so fewer informative tiles are discarded, and (2) ensure the mask buffer is filled using the dataset-provided tile indices even when batching (no semantic change, but avoids any edge-case misalignment). Everything else (architecture, thresholding, resize/reduce, sigmoid+avg ensemble, RLE format) remains unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing (almost) all-empty masks despite running, which often happens here when the “blank tile” filter is too aggressive and silently drops most informative tiles, leaving large areas as zeros. To move the Dice upward toward your target while preserving your model, tiling, inference, thresholding, and RLE semantics, I make a minimal adjustment to the blank-tile gating so it only rejects truly blank/white tiles (rather than low-saturation but still informative tissue). I also ensure the stitched `mask` tensor is indexed with the dataset tile index robustly (avoiding any edge-case where `y` might not map 1:1 after filtering), without changing any prediction math. These changes should increase non-empty predictions and improve Dice from 0.0 toward the target, while keeping runtime and core logic essentially identical.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

SMP_AVAILABLE = False

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
sz = 256  # the size of tiles (model input size after reduce)
reduce = 4  # reduce the original images by 4 times
TH = 0.44  # threshold for positive predictions
DATA = "../input/hubmap-kidney-segmentation/test/"
MODELS = [f"../input/test-ac/result/FOLD-{i}-model.pth" for i in range(5)]
df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if enc is None:
            continue
        if isinstance(enc, float) and np.isnan(enc):
            continue
        if not isinstance(enc, str):
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
    pixels = img.T.flatten().astype(np.uint8, copy=False)
    if pixels.size == 0:
        return ""
    if pixels.sum() == 0:
        return ""  # empty mask should be empty string in this competition

    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

s_th = 32
p_th = 500 * (sz // 256) ** 2

gray_std_th = 8.0  # below this, tile is nearly uniform (paper/background)
gray_mean_th = 235  # very bright uniform background tends to be high mean
min_nonzero_th = p_th  # keep original safeguard against fully empty padded tiles


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def _normalize_tile_to_hwc3_uint8(tile: np.ndarray) -> np.ndarray:
    """
    Convert a sliced tile (could be HxW, HxWxC, CxHxW, various dtypes) to HxWx3 uint8.
    Robustness-only to ensure valid 3-channel input for the model/HSV filter.
    """
    if tile.ndim == 2:
        tile = tile[..., None]
    elif tile.ndim == 3:
        if tile.shape[0] in (3, 4) and tile.shape[2] not in (3, 4):
            tile = np.transpose(tile, (1, 2, 0))
    else:
        raise ValueError(f"Unexpected tile ndim={tile.ndim}")

    if tile.shape[2] == 1:
        tile = np.repeat(tile, 3, axis=2)
    elif tile.shape[2] > 3:
        tile = tile[..., :3]
    elif tile.shape[2] != 3:
        raise ValueError(f"Unexpected channel count {tile.shape[2]}")

    if tile.dtype != np.uint8:
        if np.issubdtype(tile.dtype, np.floating):
            tile = np.clip(tile * 255.0, 0, 255).astype(np.uint8)
        else:
            tile = np.clip(tile, 0, 255).astype(np.uint8)
    return tile


def _read_tiff_tile_bgr_tifffile(
    page, x0: int, x1: int, y0: int, y1: int
) -> np.ndarray:
    """
    Coordinates: x is row (0..H), y is col (0..W), consistent with existing code.
    Returns (x1-x0, y1-y0, 3) uint8 in BGR order.
    """
    H = int(page.imagelength)
    W = int(page.imagewidth)

    x0 = max(0, min(H, int(x0)))
    x1 = max(0, min(H, int(x1)))
    y0 = max(0, min(W, int(y0)))
    y1 = max(0, min(W, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    try:
        if getattr(page, "is_tiled", False) and page.is_tiled:
            tl, tw = int(page.tilelength), int(page.tilewidth)
            rr0, rr1 = x0 // tl, (x1 - 1) // tl
            cc0, cc1 = y0 // tw, (y1 - 1) // tw

            out_rgb = np.zeros((x1 - x0, y1 - y0, 3), dtype=np.uint8)

            for rr in range(rr0, rr1 + 1):
                for cc in range(cc0, cc1 + 1):
                    tile = page.tile(rr, cc)
                    tile = _normalize_tile_to_hwc3_uint8(tile)

                    tx0, ty0 = rr * tl, cc * tw
                    tx1, ty1 = tx0 + tile.shape[0], ty0 + tile.shape[1]

                    ix0, ix1 = max(x0, tx0), min(x1, tx1)
                    iy0, iy1 = max(y0, ty0), min(y1, ty1)

                    out_x0, out_x1 = ix0 - x0, ix1 - x0
                    out_y0, out_y1 = iy0 - y0, iy1 - y0

                    tile_x0, tile_x1 = ix0 - tx0, ix1 - tx0
                    tile_y0, tile_y1 = iy0 - ty0, iy1 - ty0

                    if out_x1 > out_x0 and out_y1 > out_y0:
                        out_rgb[out_x0:out_x1, out_y0:out_y1] = tile[
                            tile_x0:tile_x1, tile_y0:tile_y1
                        ]

            return out_rgb[..., ::-1]  # RGB->BGR
    except Exception:
        pass

    out_rgb = np.zeros((x1 - x0, y1 - y0, 3), dtype=np.uint8)

    try:
        if getattr(page, "is_tiled", False) and page.is_tiled:
            seg_h, seg_w = int(page.tilelength), int(page.tilewidth)
        else:
            seg_h = int(getattr(page, "rowsperstrip", H))
            if seg_h <= 0:
                seg_h = H
            seg_w = W
    except Exception:
        seg_h, seg_w = H, W

    rr0, rr1 = x0 // seg_h, (x1 - 1) // seg_h
    cc0, cc1 = y0 // seg_w, (y1 - 1) // seg_w

    try:
        for _, (sr, sc), seg in page.segments(lock=True):
            if sr is None or sc is None:
                continue
            sr = int(sr)
            sc = int(sc)

            rr = sr // seg_h
            cc = sc // seg_w
            if rr < rr0 or rr > rr1 or cc < cc0 or cc > cc1:
                continue

            seg = _normalize_tile_to_hwc3_uint8(seg)
            sx0, sy0 = sr, sc
            sx1, sy1 = sx0 + seg.shape[0], sy0 + seg.shape[1]

            ix0, ix1 = max(x0, sx0), min(x1, sx1)
            iy0, iy1 = max(y0, sy0), min(y1, sy1)

            if ix1 <= ix0 or iy1 <= iy0:
                continue

            out_x0, out_x1 = ix0 - x0, ix1 - x0
            out_y0, out_y1 = iy0 - y0, iy1 - y0

            seg_x0, seg_x1 = ix0 - sx0, ix1 - sx0
            seg_y0, seg_y1 = iy0 - sy0, iy1 - sy0

            out_rgb[out_x0:out_x1, out_y0:out_y1] = seg[seg_x0:seg_x1, seg_y0:seg_y1]
    except Exception:
        return np.zeros((x1 - x0, y1 - y0, 3), dtype=np.uint8)

    return out_rgb[..., ::-1]  # RGB->BGR


class HuBMAPDataset(Dataset):
    """
    TIFF is huge and JPEG-compressed; full decode requires imagecodecs (not installed).
    Use segment-based reads only; keep the rest of the tiling logic unchanged.
    """

    def __init__(self, idx, sz=sz, reduce=reduce):
        tiff_path = os.path.join(DATA, idx + ".tiff")
        if not os.path.exists(tiff_path):
            alt_path = os.path.join(DATA, idx + ".tif")
            if os.path.exists(alt_path):
                tiff_path = alt_path
            else:
                raise FileNotFoundError(f"Missing TIFF for id={idx}: {tiff_path}")

        self.tiff_path = tiff_path
        self.tf = tiff.TiffFile(tiff_path)
        self.page = self.tf.pages[0]

        H, W = int(self.page.imagelength), int(self.page.imagewidth)
        self.shape = (H, W)  # (H, W)
        self.reduce = reduce
        self.sz = reduce * sz  # tile size in original resolution

        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        x1, y1 = x0 + self.sz, y0 + self.sz

        p00, p01 = max(0, x0), min(x1, self.shape[0])
        p10, p11 = max(0, y0), min(y1, self.shape[1])

        tile = np.zeros((self.sz, self.sz, 3), np.uint8)
        if (p01 > p00) and (p11 > p10):
            region = _read_tiff_tile_bgr_tifffile(self.page, p00, p01, p10, p11)
            region = _normalize_tile_to_hwc3_uint8(region)
            tile[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = region

        if self.reduce != 1:
            tile = cv2.resize(
                tile,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )

        tile_bgr = tile  # BGR for HSV blanking logic

        hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)

        gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
        gstd = float(gray.std())
        gmean = float(gray.mean())

        tile_f = tile_bgr.astype(np.float32) / 255.0
        tile_f = (tile_f - mean) / std

        is_blank = (s > s_th).sum() <= p_th and tile_bgr.sum() <= min_nonzero_th
        is_uniform_bright = gstd < gray_std_th and gmean > gray_mean_th

        if is_blank or is_uniform_bright:
            return img2tensor(tile_f), -1
        else:
            return img2tensor(tile_f), idx

    def __del__(self):
        try:
            if hasattr(self, "tf") and self.tf is not None:
                self.tf.close()
        except Exception:
            pass




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
                if (y >= 0).sum() > 0:  # exclude empty images
                    x = x[y >= 0].to(device)
                    y = y[y >= 0]
                    if self.half:
                        x = x.half()

                    py = None
                    for model in self.models:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        py = p if py is None else (py + p)
                    py /= len(self.models)

                    py = F.interpolate(
                        py, scale_factor=reduce, mode="bilinear", align_corners=False
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    for i in range(len(py)):
                        yield py[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 5
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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffX != 0 or diffY != 0:
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.up1 = Up(base * 8 + base * 4, base * 4)
        self.up2 = Up(base * 4 + base * 2, base * 2)
        self.up3 = Up(base * 2 + base, base)
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


class HuBMAP(nn.Module):
    def __init__(self):
        super(HuBMAP, self).__init__()
        self.cnn_model = TinyUNet(in_ch=3, out_ch=1, base=32)

    def forward(self, imgs):
        img_segs = self.cnn_model(imgs)
        return img_segs




## === cell 6
models = []
available = [p for p in MODELS if os.path.exists(p)]
if len(available) == 0:
    model = HuBMAP().to(device)
    model.float()
    model.eval()
    models = [model]
else:
    for path in available:
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model = HuBMAP()
        if (
            isinstance(state_dict, dict)
            and "state_dict" in state_dict
            and isinstance(state_dict["state_dict"], dict)
        ):
            state_dict = state_dict["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            if isinstance(state_dict, dict):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
        model.load_state_dict(state_dict, strict=False)
        model.float()
        model.eval()
        model.to(device)
        models.append(model)
    del state_dict
gc.collect()



## === cell 7
names, preds = [], []

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]
    ds = HuBMAPDataset(idx)
    dl = DataLoader(
        ds,
        batch_size=bs,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        num_workers=0,
    )
    mp = Model_pred(models, dl)

    mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.uint8)

    for p, i in iter(mp):
        ti = int(i.detach().cpu().item()) if torch.is_tensor(i) else int(i)
        if 0 <= ti < len(ds):
            mask[ti] = (p.squeeze(-1) > TH).to(torch.uint8)

    mask = (
        mask.view(ds.n0max, ds.n1max, ds.sz, ds.sz)
        .permute(0, 2, 1, 3)
        .reshape(ds.n0max * ds.sz, ds.n1max * ds.sz)
    )

    mask = mask[
        ds.pad0 // 2 : -(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz,
        ds.pad1 // 2 : -(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz,
    ]

    rle = rle_encode_less_memory(mask.numpy())
    names.append(idx)
    preds.append(rle)

    del mask, dl, ds, mp
    gc.collect()



## === cell 8
sub = pd.DataFrame({"id": names, "predicted": preds})
sub = df_sample[["id"]].merge(sub, on="id", how="left")
sub["predicted"] = sub["predicted"].fillna("")  # if any missing, submit empty mask
sub.to_csv("submission.csv", index=False)

print(sub.head())
print(f"Saved submission.csv with {len(sub)} rows")
