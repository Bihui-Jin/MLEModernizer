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

0.9279253721993572

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by implementing the same tiled TIFF reading using `tifffile`/NumPy with equivalent window slicing, so the pipeline runs end-to-end. I also fix a few runtime issues (`np.float` deprecation, missing imports due to earlier cell failure, and `F.upsample` deprecation) without changing the model, weights, TTA logic, tiling scheme, or thresholding semantics. Finally, I ensure we iterate over all test IDs and always write a 3-row `submission.csv` with the exact required columns (`id`, `predicted`) so Kaggle accepts it.'
- What this solution (achieved 0.0) has done: 'I fix the two root runtime blockers: (1) `segmentation_models_pytorch` is not installed, so I provide a tiny Unet-like fallback that keeps the same call signature and output shape; and (2) the test TIFFs are JPEG-compressed and `tifffile` needs `imagecodecs`, so I switch the TIFF reading to OpenCV’s TIFF loader which supports JPEG-compressed TIFFs in this environment. I also make model-weight loading robust: if the external `.pth` files aren’t present, the code still run end-to-end by instantiating the model(s) with random weights (this won’t reach the target Dice but prevents the 0.0-from-no-submission situation). Finally, I ensure the submission is written with the required `id,predicted` columns and a `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'We need to fix the OpenCV TIFF read failure caused by OpenCV’s built-in pixel limit for very large images, which currently stops inference before any submission is written (leading to a 0.0). The smallest safe fix is to avoid loading the full TIFF at once and instead read only the needed tile window using `tifffile.memmap` (does not require `imagecodecs` for memmapping) and fall back to a full read only if memmap fails. This preserves the existing tiling, normalization, model, TTA, and thresholding semantics, but changes the I/O to be windowed so it runs within OpenCV limits. Finally, ensure the submission is always produced with the required `id,predicted` columns and `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'I fix the OpenCV “max image pixels” crash by removing the last-resort full-image `cv2.imread` path and replacing it with a safe, tile-wise TIFF reader that never loads the whole slide at once. The patch keeps your tiling, normalization, model forward/ensemble/TTA logic, and thresholding unchanged; only the tile I/O is made robust so inference completes and a valid `submission.csv` is always written. I implement a `_read_tiff_tile()` that tries `tifffile.memmap` first, and if that fails (e.g., compressed TIFF), uses `tifffile.TiffFile(...).pages[0].asarray(key=...)` to decode only the requested window. Finally, I ensure the submission uses the exact `id,predicted` columns and fills missing predictions with an empty string.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF tile reader so it never calls `TiffPage.asarray(key=...)` (not supported in this tifffile version) and also never triggers full-image JPEG decompression that requires `imagecodecs`. The minimal safe approach is: try `tifffile.memmap` slicing first (works for uncompressed/compatible files), and if that fails, decode only the requested window using OpenCV’s TIFF loader on a temporary cropped TIFF written via `tifffile.imwrite` from the memmapped region-free fallback (for JPEG-compressed files, OpenCV can decode without `imagecodecs`). This keeps your tiling/inference logic unchanged while unblocking end-to-end submission generation, which should move the score up from 0.0 (no valid inference) toward the target. Finally, I keep the submission formatting as `id,predicted` and always write `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the crash caused by OpenCV’s global pixel-limit by removing the full-image `cv2.imread` fallback and replacing it with a robust tile-window reader that never loads the whole TIFF. The new reader (1) try `tifffile.memmap` slicing when possible and (2) otherwise decode the entire image once via PIL (which avoids OpenCV’s pixel-limit) and then serve tiles from that cached array; this is a minimal I/O change that preserves the existing tiling, normalization, model forward/TTA logic, and thresholding. I also ensure the dataset always returns RGB uint8 tiles consistently (handling grayscale/alpha) and that the script always writes a valid `submission.csv` with `id,predicted`. This should move the score up from 0.0 by enabling end-to-end inference and a valid submission.'
- What this solution (achieved 0.0) has done: 'I fix the crash by removing the “read full TIFF with PIL” fallback (it triggers PIL’s decompression-bomb safeguard on these huge slides) and replacing it with a true tile-window reader that never loads the whole image. The new `_read_tiff_tile` try memory-mapped slicing first, and if the TIFF is not mappable, it decode only the requested window using `tifffile.TiffFile(...).pages[0].asarray(region=...)`, which works without loading the full slide. This change preserves your tiling, normalization, model/TTA logic, and thresholding semantics, but unblocks end-to-end inference so a valid `submission.csv` is produced (moving score up from 0.0). I also ensure we don’t rely on any unsupported tifffile APIs by handling both possible `region` signatures.'
- What this solution (achieved 0.0) has done: 'I fix the inference crash by removing the unsupported `page.asarray(region=...)` call in the TIFF tile reader and replacing it with a safe, always-available approach: attempt `tifffile.memmap` slicing first, and if that’s not possible, fall back to reading the full image via OpenCV with `IMREAD_REDUCED_COLOR_4` (matching your `reduce=4`) to avoid both `imagecodecs` and the huge-pixel limit. This keeps your tiling, normalization, model forward/TTA, and thresholding semantics intact while ensuring the pipeline completes and writes `submission.csv`. I also make the dataset robust to the reduced-read path by computing shapes from metadata and serving tiles from the cached reduced image when used. These changes should move the score up from 0.0 by enabling end-to-end prediction + valid submission creation.'
- What this solution (achieved 0.0) has done: 'The crash comes from OpenCV enforcing a maximum pixel limit even for `IMREAD_REDUCED_*`, which prevents dataset construction and stops submission creation (yielding a 0.0). I remove the OpenCV reduced-read fallback and replace it with a true tile-window TIFF reader that never loads the full image: it tries `tifffile.memmap` (fast for uncompressed/compatible TIFFs) and otherwise decodes only the requested window using `TiffPage.asarray(region=...)` with a robust signature fallback. This keeps your tiling, normalization, model forward/TTA behavior, and thresholding the same, but unblocks end-to-end inference so a valid `submission.csv` is always written. I also make the RLE encoder safe on empty masks (return empty string) to avoid malformed submissions.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF tile reader crash by removing the unsupported `page.asarray(region=...)` call and replacing it with a compatibility path that uses `tifffile.imread(..., aszarr=True)` to get a lazy, window-readable Zarr array and slice tiles from it. This preserves your existing tiling, normalization, model forward/TTA, and thresholding logic; only the tile I/O backend changes so inference can finish. I also make the blank-tile logic safe by ensuring the dataset returns the original tile index (so `mask[i]` assignment never goes out of range) and keep submission formatting exactly as `id,predicted` written to `submission.csv`. These changes should move the score up from 0.0 by enabling end-to-end prediction and a valid submission file.'

# 9. Code solution

## === cell 0
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import gc
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff

import warnings

warnings.filterwarnings("ignore")

try:
    import segmentation_models_pytorch as smp  # type: ignore

    _HAS_SMP = True
except Exception:
    smp = None
    _HAS_SMP = False



## === cell 1
sz = 256  # the size of tiles
reduce = 4  # reduce the original images by 4 times
TH = 0.44  # threshold for positive predictions

DATA = "../input/hubmap-kidney-segmentation/test/"
if not os.path.isdir(DATA):
    alt = "/kaggle/input/hubmap-kidney-segmentation/test/"
    if os.path.isdir(alt):
        DATA = alt

MODELS = [
    f"../input/normalize-seresnext/result/se_resnext50_32x4d-FOLD-{i}-model.pth"
    for i in range(5)
]

df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
if not set(["id", "predicted"]).issubset(df_sample.columns):
    df_sample.columns = ["id", "predicted"]

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, (float, np.floating)) and np.isnan(enc):
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
    if pixels.size == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if runs.size == 0:
        return ""
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


def _tiff_get_shape(path: str):
    """
    Avoid loading full huge TIFF. Use tifffile metadata/shape quickly.
    """
    with tiff.TiffFile(path) as tf:
        page = tf.pages[0]
        shape = page.shape  # can be (H,W) or (H,W,C) or (C,H,W)
        if len(shape) == 2:
            h, w = shape
            c = 1
        elif len(shape) == 3:
            if shape[-1] in (3, 4):
                h, w, c = shape
            elif shape[0] in (3, 4):
                c, h, w = shape
            else:
                h, w, c = shape
        else:
            raise ValueError(f"Unexpected TIFF shape {shape} for {path}")
        return int(h), int(w), int(c)


def _tile_to_rgb_uint8(tile: np.ndarray) -> np.ndarray:
    """Convert (H,W), (H,W,C), or (C,H,W) tile to HWC RGB uint8."""
    if tile.ndim == 2:
        tile = np.stack([tile, tile, tile], axis=-1)
    elif tile.ndim == 3:
        if tile.shape[-1] in (3, 4):  # HWC
            tile = tile[..., :3]
        elif tile.shape[0] in (3, 4):  # CHW
            tile = tile[:3, ...]
            tile = np.transpose(tile, (1, 2, 0))
        else:
            tile = tile[..., :3]
    else:
        raise ValueError(f"Unsupported tile ndim={tile.ndim}")

    if tile.dtype == np.uint16:
        tile = (tile / 257.0).astype(np.uint8)
    elif tile.dtype != np.uint8:
        tile = np.clip(tile, 0, 255).astype(np.uint8)
    return tile


class _TiffTileSource:
    """
    BUGFIX:
    tifffile.TiffPage.asarray(region=...) is not available in this environment's tifffile.
    Use tifffile.imread(..., aszarr=True) to get a lazy, sliceable Zarr store for window reads
    (no full-image decode, no OpenCV pixel limit).
    """

    def __init__(self, path: str):
        self.path = path
        self._arr = None  # memmap
        self._z = None  # zarr array

        try:
            self._arr = tiff.memmap(path)
        except Exception:
            self._arr = None

        try:
            store = tiff.imread(path, aszarr=True)
            import zarr  # zarr is a dependency of tifffile's aszarr path in Kaggle envs typically

            self._z = zarr.open(store, mode="r")
        except Exception:
            self._z = None

        if self._arr is None and self._z is None:
            raise RuntimeError(
                f"Could not initialize any tiled reader backend for TIFF: {path}. "
                f"memmap failed and aszarr failed."
            )

    def read(self, p00: int, p01: int, p10: int, p11: int) -> np.ndarray:
        if self._arr is not None:
            arr = self._arr
            if arr.ndim == 2:
                tile = np.asarray(arr[p00:p01, p10:p11])
            elif arr.ndim == 3:
                if arr.shape[-1] in (3, 4):  # HWC
                    tile = np.asarray(arr[p00:p01, p10:p11, :3])
                elif arr.shape[0] in (3, 4):  # CHW
                    tile = np.asarray(arr[:3, p00:p01, p10:p11])
                else:
                    tile = np.asarray(arr[p00:p01, p10:p11, :3])
            else:
                raise ValueError(f"Unsupported array ndim={arr.ndim}")
            return _tile_to_rgb_uint8(tile)

        z = self._z
        if z is None:
            raise RuntimeError("No available backend to read tiles.")

        if hasattr(z, "shape"):
            za = z
        else:
            keys = list(z.array_keys())
            if len(keys) == 0:
                raise RuntimeError("aszarr returned a group with no arrays.")
            za = z[keys[0]]

        if len(za.shape) == 2:
            tile = np.asarray(za[p00:p01, p10:p11])
        elif len(za.shape) == 3:
            if za.shape[-1] in (3, 4):  # HWC
                tile = np.asarray(za[p00:p01, p10:p11, :3])
            elif za.shape[0] in (3, 4):  # CHW
                tile = np.asarray(za[:3, p00:p01, p10:p11])
            else:
                tile = np.asarray(za[p00:p01, p10:p11, :3])
        else:
            raise ValueError(f"Unsupported zarr array shape={za.shape}")

        return _tile_to_rgb_uint8(tile)


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path = os.path.join(DATA, idx + ".tiff")

        self._src = _TiffTileSource(self.path)

        h, w, _ = _tiff_get_shape(self.path)
        self.shape = (h, w)

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

        tile = np.zeros((self.sz, self.sz, 3), np.uint8)

        patch = self._src.read(p00, p01, p10, p11)
        tile[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = patch

        if self.reduce != 1:
            tile = cv2.resize(
                tile,
                (self.sz // reduce, self.sz // reduce),
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        if (s > s_th).sum() <= p_th or tile.sum() <= p_th:
            return img2tensor((tile / 255.0 - mean) / std), -1
        else:
            return img2tensor((tile / 255.0 - mean) / std), idx




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
                        py, scale_factor=reduce, mode="bilinear", align_corners=False
                    )
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    batch_size = len(py)
                    for i in range(batch_size):
                        yield py[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 5
class _ConvBlock(nn.Module):
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


class _TinyUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.enc1 = _ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


class HuBMAP(nn.Module):
    def __init__(self):
        super(HuBMAP, self).__init__()
        if _HAS_SMP:
            self.cnn_model = smp.Unet(
                "se_resnext50_32x4d", encoder_weights=None, classes=1
            )
        else:
            self.cnn_model = _TinyUNet(in_ch=3, out_ch=1, base=32)

    def forward(self, imgs):
        img_segs = self.cnn_model(imgs)
        return img_segs




## === cell 6
models = []
missing = []
for path in MODELS:
    if os.path.isfile(path):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model = HuBMAP()
        model.load_state_dict(state_dict)
        model.float()
        model.eval()
        model.to(device)
        models.append(model)
        del state_dict
    else:
        missing.append(path)

if len(models) == 0:
    model = HuBMAP()
    model.float()
    model.eval()
    model.to(device)
    models = [model]

gc.collect()



## === cell 7
names, preds = [], []
for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]
    ds = HuBMAPDataset(idx)
    dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
    mp = Model_pred(models, dl)

    mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
    for p, i in iter(mp):
        mask[i.item()] = p.squeeze(-1) > TH

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

    del mask, ds, dl
    gc.collect()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2480616948.py in <cell line: 0>()
      2 for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
      3     idx = row["id"]
----> 4     ds = HuBMAPDataset(idx)
      5     dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
      6     mp = Model_pred(models, dl)

/tmp/ipykernel_55/3707686118.py in __init__(self, idx, sz, reduce)
    143         self.path = os.path.join(DATA, idx + ".tiff")
    144 
--> 145         self._src = _TiffTileSource(self.path)
    146 
    147         h, w, _ = _tiff_get_shape(self.path)

/tmp/ipykernel_55/3707686118.py in __init__(self, path)
     87 
     88         if self._arr is None and self._z is None:
---> 89             raise RuntimeError(
     90                 f"Could not initialize any tiled reader backend for TIFF: {path}. "
     91                 f"memmap failed and aszarr failed."

RuntimeError: Could not initialize any tiled reader backend for TIFF: ../input/hubmap-kidney-segmentation/test/8242609fa.tiff. memmap failed and aszarr failed.

## === cell 8
df = pd.DataFrame({"id": names, "predicted": preds})
df = df_sample[["id"]].merge(df, on="id", how="left")
df["predicted"] = df["predicted"].fillna("")  # empty mask allowed

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with rows:", len(df))
if len(missing) > 0:
    print(
        f"Warning: {len(missing)} model weight files were missing; ran with {len(models)} model(s) available."
    )
