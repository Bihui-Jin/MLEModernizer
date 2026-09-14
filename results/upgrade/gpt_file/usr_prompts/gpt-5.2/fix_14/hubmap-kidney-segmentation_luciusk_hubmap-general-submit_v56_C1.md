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
- What this solution (achieved 0.0) has done: 'I fix the TIFF reading failure that currently blocks inference and causes the 0.0 score by removing reliance on `tifffile` decoding (which needs the unavailable `imagecodecs` for JPEG-compressed TIFFs). The minimal change is to read tiles directly with OpenCV using `IMREAD_REDUCED_COLOR_4` (matching your `reduce=4`) and then keep the same tiling, normalization, thresholding, and RLE encoding logic. I also make the dataset shape come from the downsampled image so the mask assembly indices stay consistent, and I ensure the submission is written with the required `id,predicted` columns to `submission.csv`. This should move the score up from 0.0 by producing valid, non-empty predictions end-to-end.'
- What this solution (achieved 0.0) has done: 'I fix the OpenCV crash caused by its `CV_IO_MAX_IMAGE_PIXELS` safety limit by removing OpenCV from the TIFF-reading path and instead reading the already-downsampled whole-slide image using `tifffile.TiffFile(...).asarray(out='memmap')`, which avoids full decompression into RAM and does not require `imagecodecs` for these baseline test TIFFs. This keeps your existing tiling, normalization, model inference, thresholding, and RLE submission logic the same—only the image loader changes to be compatible with the Kaggle runtime. I also add a small fallback to handle grayscale pages safely and ensure the submission is always generated with `id,predicted` aligned to `sample_submission.csv`. These changes are necessary to run end-to-end and should raise the score above 0.0 by actually producing non-empty predictions (assuming the provided weights exist); if weights are missing, it still produce a valid submission.'
- What this solution (achieved 0.0) has done: 'I fix the crash that prevents any submission from being generated: `tifffile` cannot decode the JPEG-compressed TIFFs here because `imagecodecs` isn’t installed. To keep your core tiling/inference/RLE logic intact while unblocking execution, I switch the downsampled TIFF loading to OpenCV’s reduced-resolution TIFF reader (fast and avoids full-slide decode), and I add a safe fallback that still produces an image if reduced reading fails. I also ensure the script uses the correct sample submission columns (`id`, `predicted`) and always writes `submission.csv` in the working directory. These changes are runtime/correctness-focused and should move the score up from 0.0 by actually producing valid predictions end-to-end.'
- What this solution (achieved 0.0) has done: 'I fix the crash in TIFF loading that currently prevents inference and leads to a 0.0 score by avoiding both OpenCV full-image decode (blocked by `CV_IO_MAX_IMAGE_PIXELS`) and `tifffile` JPEG decode (blocked by missing `imagecodecs`). The minimal robust solution is to read only a *downsampled* image for tiling using `PIL.Image.open(...).draft(...)`, which can decode these TIFFs via Pillow without `imagecodecs`, and keeps your existing tiling/inference/thresholding/RLE logic unchanged. I also keep the same submission column names (`id`, `predicted`) and ensure the pipeline always writes `submission.csv` end-to-end. This should move the score up from 0.0 by successfully generating non-empty predictions (assuming the model weights exist; otherwise it still generate a valid submission).'
- What this solution (achieved 0.0) has done: 'I fix the crash that prevents inference by avoiding full-resolution TIFF decoding (blocked by Pillow’s decompression-bomb safeguard) and avoiding JPEG-compressed TIFF decoding via `tifffile` (blocked by missing `imagecodecs`). The minimal stable approach is to read a small RGB thumbnail directly from the TIFF container using Pillow’s `Image.thumbnail`, after disabling the pixel-limit check for this trusted Kaggle dataset; this preserves your existing pipeline (downsample → tiling → model → threshold → RLE). I also ensure the output mask/shape stays consistent with the downsampled image used for tiling so coordinates align, and keep the submission columns as `id,predicted`. These are execution/correctness fixes; they should move the score above 0.0 by producing a valid, non-crashing submission end-to-end.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by (1) repeatedly converting every tile to HSV and counting saturation pixels in Python, and (2) single-threaded data loading and per-model sequential inference. I keep the exact tiling/inference/thresholding logic, but speed it up by vectorizing the “blank tile” check (use a cheap RGB-based saturation proxy with identical semantics for skipping only obviously-empty tiles), batching and parallelizing data loading with multiple workers, and reducing per-tile overhead by precomputing constant tensors and avoiding repeated allocations. I also enable cuDNN benchmarking for fixed input sizes and use `inference_mode()` to cut autograd overhead while preserving outputs. All file paths, model architecture/weights loading, thresholds, and overall evaluation behavior remain unchanged.'

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

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = (
        True  # fixed (sz,sz) batches benefit; numerics unchanged
    )
    torch.backends.cudnn.deterministic = False




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


_MEAN_T = torch.tensor(mean.reshape(3, 1, 1), dtype=torch.float32)
_STD_T = torch.tensor(std.reshape(3, 1, 1), dtype=torch.float32)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
    Return array (N,4) where each row is x1,x2,y1,y2 in original image coordinates.
    shape is (H, W).

    Fix: guard against invalid (window - min_overlap) <= 0, which caused negative nx/ny
    and crashed np.linspace. Semantics preserved: stride is still (window-min_overlap),
    but clamped to at least 1 pixel so we always produce at least one tile.
    """
    x, y = int(shape[0]), int(shape[1])
    window = int(window)
    min_overlap = int(min_overlap)

    stride = max(1, window - min_overlap)
    nx = max(1, x // stride + 1)
    ny = max(1, y // stride + 1)

    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)

    x1[-1] = max(0, x - window)
    y1[-1] = max(0, y - window)

    x2 = (x1 + window).clip(0, x)
    y2 = (y1 + window).clip(0, y)

    X1, Y1 = np.meshgrid(x1, y1, indexing="ij")
    X2, Y2 = np.meshgrid(x2, y2, indexing="ij")
    slices = (
        np.stack([X1, X2, Y1, Y2], axis=-1).reshape(-1, 4).astype(np.int64, copy=False)
    )
    return slices




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


from PIL import Image, ImageFile  # pillow is installed

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def _pil_read_downsampled_rgb(path: str, reduce_factor: int) -> np.ndarray:
    """
    Read TIFF using Pillow and return an RGB uint8 image downsampled by reduce_factor.
    Uses thumbnail so decoding work stays bounded; this avoids full-res decode.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        if reduce_factor and reduce_factor != 1:
            target_size = (
                max(1, im.size[0] // reduce_factor),
                max(1, im.size[1] // reduce_factor),
            )
            im.thumbnail(target_size, resample=Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    return arr


def _tiff_read_downsampled_rgb(path: str, reduce_factor: int) -> np.ndarray:
    """
    Returns (H', W', 3) RGB uint8.
    Prefer Pillow downsampled load; fall back to tifffile if Pillow cannot open the file.
    Note: tifffile JPEG-compressed decode requires imagecodecs, which isn't installed here.
    """
    try:
        return _pil_read_downsampled_rgb(path, reduce_factor)
    except Exception as e:
        try:
            with tiff.TiffFile(path) as tf:
                page = tf.pages[0]
                img = page.asarray()
        except Exception as e2:
            raise RuntimeError(
                f"Failed to read TIFF (Pillow and tifffile). Pillow error: {type(e).__name__}: {e}. "
                f"tifffile error: {type(e2).__name__}: {e2}"
            )
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.integer):
                img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
            else:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        if reduce_factor != 1:
            new_w = max(1, int(img.shape[1] // reduce_factor))
            new_h = max(1, int(img.shape[0] // reduce_factor))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img


if shift:

    class HuBMAPDataset(Dataset):
        def __init__(self, idx, sz=sz, reduce=reduce):
            self.path = _find_tiff_path(idx)
            if not os.path.isfile(self.path):
                raise FileNotFoundError(f"Missing TIFF for id={idx}: {self.path}")

            self.img_small = _tiff_read_downsampled_rgb(self.path, reduce)
            self.shape = (int(self.img_small.shape[0]), int(self.img_small.shape[1]))

            self.reduce = reduce
            self.sz = int(sz)  # tile size at model input resolution (downsampled)

            _mo = int(minoverlap)
            _mo = min(_mo, self.sz - 1) if self.sz > 1 else 0

            self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=_mo)

            self._blank_tile = np.zeros((self.sz, self.sz, 3), dtype=np.uint8)

        def __len__(self):
            return len(self.mask_grid)

        def __getitem__(self, idx):
            x1, x2, y1, y2 = self.mask_grid[idx]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

            h = x2 - x1
            w = y2 - y1

            img = self._blank_tile.copy()
            tile = self.img_small[x1:x2, y1:y2, :]
            img[:h, :w, :3] = tile[:, :, :3]

            mx = img.max(axis=2)
            mn = img.min(axis=2)
            s = np.zeros_like(mx, dtype=np.uint8)
            nz = mx != 0
            s[nz] = (
                (mx[nz].astype(np.uint16) - mn[nz].astype(np.uint16))
                * 255
                // mx[nz].astype(np.uint16)
            ).astype(np.uint8)

            vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)

            if (s > s_th).sum() <= p_th or img.sum() <= p_th:
                x = img2tensor(img / 255.0).float()
                x = (x - _MEAN_T) / _STD_T
                return x, vertices, -1
            else:
                x = img2tensor(img / 255.0).float()
                x = (x - _MEAN_T) / _STD_T
                return x, vertices, idx

    class Model_pred:
        def __init__(self, models, dl, tta: bool = False, half: bool = False):
            self.models = models
            self.dl = dl
            self.tta = tta
            self.half = half

        def __iter__(self):
            with torch.inference_mode():
                for x, z, y in iter(self.dl):
                    if (y >= 0).sum() > 0:
                        keep = y >= 0
                        x = x[keep].to(device, non_blocking=True)
                        z = z[keep]
                        y = y[keep]

                        if self.half:
                            x = x.half()

                        py = None
                        for model in self.models:
                            p = model(x)
                            p = torch.sigmoid(p)
                            py = p if py is None else (py + p)

                        if self.tta:
                            flips = [[-1], [-2], [-2, -1]]
                            for f in flips:
                                xf = torch.flip(x, f)
                                for model in self.models:
                                    p = model(xf)
                                    p = torch.flip(p, f)
                                    py += torch.sigmoid(p)
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

    _num_workers = min(4, (os.cpu_count() or 2))
    names, preds = [], []
    for _, row in tqdm(df_sample.itertuples(index=False), total=len(df_sample)):
        idx = row  # because sample has columns (id, predicted) but itertuples gives both; handle below
        if isinstance(idx, tuple):
            idx = idx[0]

        ds = HuBMAPDataset(idx)
        dl = DataLoader(
            ds,
            batch_size=bs,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            num_workers=_num_workers,
            persistent_workers=(_num_workers > 0),
            prefetch_factor=2 if _num_workers > 0 else None,
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

        del mask, ds, dl
        gc.collect()

else:
    raise RuntimeError(
        "This script is configured with shift=True; shift=False branch requires rasterio in the original code."
    )




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1695286605.py in <cell line: 0>()
    228             idx = idx[0]
    229 
--> 230         ds = HuBMAPDataset(idx)
    231         dl = DataLoader(
    232             ds,

/tmp/ipykernel_55/1695286605.py in __init__(self, idx, sz, reduce)
     81     class HuBMAPDataset(Dataset):
     82         def __init__(self, idx, sz=sz, reduce=reduce):
---> 83             self.path = _find_tiff_path(idx)
     84             if not os.path.isfile(self.path):
     85                 raise FileNotFoundError(f"Missing TIFF for id={idx}: {self.path}")

/tmp/ipykernel_55/1695286605.py in _find_tiff_path(idx)
      4     """
      5     cands = [
----> 6         os.path.join(DATA, idx + ".tiff"),
      7         os.path.join(os.path.dirname(DATA.rstrip("/")), "test", idx + ".tiff"),
      8         os.path.join("/kaggle/input/hubmap-kidney-segmentation/test", idx + ".tiff"),

TypeError: unsupported operand type(s) for +: 'float' and 'str'

## === cell 5
df = pd.DataFrame({"id": names, "predicted": preds})
df = df.set_index("id").reindex(df_sample["id"]).reset_index()

out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(df.head())
print("Wrote:", out_path, "rows:", len(df))
print("Any empty predictions:", int((df["predicted"].fillna("") == "").sum()))
