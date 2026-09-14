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

0.9301632421017284

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not installed) by providing a small in-notebook TIFF reader that supports windowed reads and the minimal `Window.from_slices` API the dataset code expects, keeping the dataset tiling logic unchanged. I also fix runtime errors caused by deprecated `np.float`, missing imports due to the early crash, and replace deprecated `F.upsample` with `F.interpolate` (same semantics). Finally, I ensure the inference loop actually iterates over all test IDs and always writes a 3-row `submission.csv` with `id,predicted` columns matching `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the pipeline so it runs end-to-end by (1) removing the unavailable `segmentation_models_pytorch` dependency while keeping the same U-Net-style core idea via a small in-script fallback U-Net, (2) making `rasterio`-dependent dataset code work reliably by providing a minimal rasterio-compatible TIFF reader (including the missing global definition that caused `NameError`), and (3) preventing the missing external model weights from crashing by safely falling back to randomly initialized models. This produce a valid `submission.csv` with the required `id,predicted` columns and non-empty (but likely low-quality) predictions instead of a 0.0 due to runtime failure/missing submission. Because your current score is 0.0 (submission effectively invalid), these changes should strictly improve toward the target by enabling actual mask generation and correct RLE formatting without changing the tiling/inference semantics beyond the necessary dependency fallbacks.'
- What this solution (achieved 0.0) has done: 'I fix the crash caused by `tifffile` needing the missing `imagecodecs` package for JPEG-compressed TIFFs by switching the fake rasterio reader to use `cv2.imread` (which can decode these TIFFs in this environment) as the primary backend, with a safe fallback to `tifffile` when possible. I keep your dataset tiling/inference logic intact by returning data in the same channel-first format expected by the existing `read(..., window=...)` API. I also ensure the test file path resolution is robust by trying both `.tiff` and `.tif` extensions without changing the core approach. These changes are score-neutral in intent but should move you from a 0.0 (runtime failure) to a valid non-empty submission and thus toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the crash caused by OpenCV’s safety limit on maximum image pixels by changing the fake `rasterio.open` reader to avoid loading the full TIFF at once and instead use `tifffile` to read only requested windows (tiles), which preserves your existing tiling/inference logic. This removes the `cv2.imread` assertion failure while keeping the dataset’s `read(..., window=...)` API unchanged. I also make the windowed reader robust to TIFF channel order and ensure it returns channel-first arrays exactly as your code expects. These changes are execution/stability fixes and should move the score up from 0.0 by enabling actual inference and a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the immediate runtime error by updating the fake rasterio TIFF window reader to use the current `tifffile` API (slicing the array directly instead of `asarray(key=...)`, which is no longer supported). To keep memory bounded and avoid loading full huge TIFFs, the reader use `TiffPage.asarray(out='memmap')` (when possible) so window slicing happens on a memory-mapped array. I also make the channel-order handling robust across grayscale/HWC/CHW pages, while preserving your existing tiling/inference logic and RLE submission format. These changes are execution/stability fixes and should move your score up from 0.0 by enabling actual inference and a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash caused by `tifffile` requiring the missing `imagecodecs` package for JPEG-compressed TIFFs by replacing the TIFF backend with a minimal `pyvips`-based window reader that supports the same `rasterio.open(...).read(..., window=...)` API your dataset tiling code expects. This keeps your tiling/inference core logic unchanged while making windowed reads work for the competition TIFFs without loading full images into memory. I also fix the submission RLE encoding to avoid mutating the mask data (which can silently zero-out real predictions), which should move the score up from 0.0 toward your target while preserving evaluation semantics. Finally, the script still always write a valid `submission.csv` with `id,predicted` matching `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash by removing the hard dependency on `pyvips` (not available in this Kaggle image) and replacing it with a minimal rasterio-compatible TIFF window reader backed by `tifffile` using memory-mapped reads, so your existing tiling/dataloader/inference logic stays the same. I also make the reader robust to TIFFs stored as CHW or HWC and to grayscale vs RGB, returning channel-first arrays exactly as your dataset expects. Finally, I keep the rest of the pipeline unchanged but ensure we always write a valid `submission.csv` with `id,predicted` and non-null strings, so the score moves up from 0.0 (runtime failure) toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the crash caused by JPEG-compressed TIFF decoding needing `imagecodecs` (not installed) by changing the fake `rasterio.open` implementation to use `opencv` as the decoding backend, while still supporting the same windowed `read(..., window=...)` API your tiling code expects. To keep memory safe (OpenCV can’t load these giant TIFFs at once), the OpenCV reader decode the full image once at a reduced resolution (downsample factor = `reduce`) and then serve corresponding window slices so the downstream model still receives the same reduced tiles and the rest of the pipeline stays unchanged. I keep model/inference/RLE logic intact, but ensure grayscale/alpha/channel ordering is handled robustly so `cv2.cvtColor` always receives a 3-channel image. This should move the score up from 0.0 by enabling real predictions and a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash in the fake `rasterio.open` implementation: OpenCV cannot decode these giant TIFFs due to its max-pixels safety limit, so we must avoid `cv2.imread` entirely for full-image loading. I replace the CV2-based reduced reader with a `tifffile`-backed window reader that memory-maps the TIFF and only slices the requested window, which preserves your existing tiling/inference logic and stays within memory/time limits. I also add a small safety fallback so if a TIFF cannot be memory-mapped, it try a non-memmap page read (still window-sliced) rather than crashing. These are execution/stability fixes intended to move you off the 0.0 score (runtime failure) toward the target by producing a valid, non-empty `submission.csv`.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by per-tile CPU work in `HuBMAPDataset.__getitem__` (TIFF window reads + OpenCV HSV/sum blank-tile filtering) and by slow single-process dataloading; additionally, the inference loop does extra Python overhead when assigning each predicted tile into the big mask. I keep the exact model/inference logic but speed up the pipeline by (1) enabling safe multi-worker DataLoader with persistent workers and prefetching, (2) vectorizing/optimizing the blank-tile check using a cheaper early-exit path plus OpenCV optimizations without changing the criterion, (3) reducing Python overhead in mask assembly by accumulating indices and doing fewer scalar `.item()` calls, and (4) using cuDNN benchmarking for fixed input sizes. These changes are deterministic (fixed seeds) and preserve the same tiles, thresholds, model outputs, and final RLE semantics.'
- What this solution (achieved 0.0) has done: 'I fix the DataLoader crash by changing the TIFF window reader to avoid `tifffile` full decompression for JPEG-compressed tiles (which needs `imagecodecs`) and also avoid the failing `torchvision.decode_image` path (it can’t decode raw JPEG-in-TIFF tile streams). Instead, I implement true window reads via `tifffile.TiffPage.asarray(region=...)`, which decodes only the requested window and is supported by `tifffile` without `imagecodecs` for this dataset’s layout. I also make the DataLoader safer by setting `num_workers=0` (so file handles aren’t shared across worker processes) which is a correctness/stability fix; the model/inference/RLE logic is unchanged. These changes should move you off the 0.0 (crash / invalid submission) toward a meaningful score by producing a valid submission with real predicted masks.'
- What this solution (achieved 0.0) has done: 'I fix the crash by removing the unsupported `TiffPage.asarray(region=...)` call and replacing it with a safe, minimal window-read implementation that uses `tifffile.memmap()` when possible (true window slicing without full decode), and otherwise falls back to `page.asarray()` for small test images. This keeps your dataset tiling/inference/RLE logic unchanged while making TIFF reading work with the installed `tifffile` version. I also make the fallback robust to grayscale/CHW/HWC layouts so the returned array shape matches what your dataset expects. These changes are execution/stability focused and should move the score up from 0.0 by enabling a valid submission with real (non-empty) predictions.'
- What this solution (achieved 0.0) has done: 'I fix the TIFF window-reader so it never falls back to full-image decoding (which triggers the `imagecodecs` JPEG requirement) by using `tifffile.TiffFile(...).series[0].aszarr()` (zarr-backed lazy access) and slicing windows from that. I also make the reader robust to zarr layouts that are CHW/HWC (and to grayscale/alpha), returning channel-first data exactly as your existing dataset code expects. These changes are strictly runtime/stability fixes (no model/inference logic changes) and allow the notebook to generate a valid `submission.csv` so your score can move up from 0.0. The rest of the pipeline (tiling, thresholding, RLE encoding, submission format) is kept intact.'

# 9. Code solution

## === cell 0
import sys, os, gc, json, math, warnings
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

try:
    import segmentation_models_pytorch as smp  # type: ignore

    _HAS_SMP = True
except Exception:
    smp = None
    _HAS_SMP = False

try:
    import tifffile
    from tifffile import TiffFile

    _HAS_TIFFFILE = True
except Exception:
    tifffile = None
    TiffFile = None
    _HAS_TIFFFILE = False


class _FakeAffine:
    def __init__(self, *args, **kwargs):
        pass


class _FakeWindow:
    @staticmethod
    def from_slices(rows, cols):
        return (rows, cols)


class _TiffFileWindowReader:
    """
    Minimal rasterio-like TIFF reader supporting windowed reads without loading full image.

    Bug fix: JPEG-compressed TIFFs require `imagecodecs` if you try to decode full image
    via `page.asarray()`. To avoid that, use zarr-backed lazy access:
      TiffFile(path).series[0].aszarr()
    and slice only the requested window.
    """

    def __init__(self, path, transform=None, num_threads=None):
        if not _HAS_TIFFFILE:
            raise RuntimeError("tifffile is required but not available.")
        self.path = path
        self.transform = transform
        self.num_threads = num_threads

        self._tf = TiffFile(path)

        self._zarr = None
        self._zarr_shape = None
        self._zarr_ndim = None
        try:
            self._zarr = self._tf.series[0].aszarr()
            self._zarr_shape = tuple(int(x) for x in self._zarr.shape)
            self._zarr_ndim = int(len(self._zarr_shape))
        except Exception:
            self._zarr = None

        self._page = self._tf.pages[0]
        page_shape = tuple(int(x) for x in getattr(self._page, "shape", ()))

        shape = self._zarr_shape if self._zarr is not None else page_shape
        if len(shape) < 2:
            raise RuntimeError(f"Unsupported TIFF shape for {path}: {shape}")

        self._stored_shape = shape

        self._is_chw = False
        if (
            len(shape) == 3
            and shape[0] in (1, 3, 4)
            and shape[1] > 16
            and shape[2] > 16
        ):
            self._is_chw = True

        if len(shape) == 2:
            h, w = shape
            self.count = 1
        elif len(shape) == 3:
            if self._is_chw:
                c, h, w = shape
                self.count = int(c)
            else:
                h, w, c = shape
                self.count = int(c)
        else:
            h, w = shape[-2], shape[-1]
            self.count = 3

        self.height = int(h)
        self.width = int(w)
        self.shape = (self.height, self.width)
        self.subdatasets = []  # not used in this fake reader

        self._mmap = None
        try:
            self._mmap = tifffile.memmap(path)
        except Exception:
            self._mmap = None

    def close(self):
        try:
            if self._tf is not None:
                self._tf.close()
        except Exception:
            pass
        self._tf = None
        self._page = None
        self._zarr = None
        self._mmap = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _slice_hwc_from_arr(
        self, arr: np.ndarray, r0: int, r1: int, c0: int, c1: int
    ) -> np.ndarray:
        if arr.ndim == 2:
            tile = arr[r0:r1, c0:c1][..., None]
        elif arr.ndim == 3:
            if self._is_chw:
                tile = np.moveaxis(arr[:, r0:r1, c0:c1], 0, -1)
            else:
                tile = arr[r0:r1, c0:c1, :]
        else:
            t = np.asarray(arr)
            if t.ndim >= 3:
                if t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
                    tile = np.moveaxis(t[:, r0:r1, c0:c1], 0, -1)
                else:
                    tile = t[r0:r1, c0:c1, ...]
                    if tile.ndim == 2:
                        tile = tile[..., None]
            else:
                tile = t[r0:r1, c0:c1][..., None]
        return tile

    def _slice_hwc_from_zarr(self, z, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        if self._zarr_ndim == 2:
            tile = np.asarray(z[r0:r1, c0:c1])[..., None]
        elif self._zarr_ndim == 3:
            if self._is_chw:
                tile = np.asarray(z[:, r0:r1, c0:c1])
                tile = np.moveaxis(tile, 0, -1)  # CHW -> HWC
            else:
                tile = np.asarray(z[r0:r1, c0:c1, :])
        else:
            a = np.asarray(z)
            tile = self._slice_hwc_from_arr(a, r0, r1, c0, c1)
        return tile

    def read(self, indexes, window=None):
        if window is None:
            r0, r1 = 0, self.height
            c0, c1 = 0, self.width
        else:
            (r0, r1), (c0, c1) = window
        r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)

        if isinstance(indexes, (int, np.integer)):
            idxs = [int(indexes) - 1]
        else:
            idxs = [int(i) - 1 for i in indexes]

        rr0 = max(0, r0)
        rr1 = min(self.height, r1)
        cc0 = max(0, c0)
        cc1 = min(self.width, c1)

        out_h = max(0, r1 - r0)
        out_w = max(0, c1 - c0)
        if out_h == 0 or out_w == 0:
            c = 1 if len(idxs) == 1 else max(1, len(idxs))
            return np.zeros((c, out_h, out_w), dtype=np.uint8)

        tile_hwc = None

        if self._zarr is not None:
            try:
                tile_hwc = self._slice_hwc_from_zarr(self._zarr, rr0, rr1, cc0, cc1)
            except Exception:
                tile_hwc = None

        if tile_hwc is None and self._mmap is not None:
            try:
                tile_hwc = self._slice_hwc_from_arr(self._mmap, rr0, rr1, cc0, cc1)
            except Exception:
                tile_hwc = None

        if tile_hwc is None:
            raise RuntimeError(
                f"Failed to read TIFF window without full decode for {self.path}. "
                f"zarr_available={self._zarr is not None}, memmap_available={self._mmap is not None}"
            )

        hh = max(0, rr1 - rr0)
        ww = max(0, cc1 - cc0)
        dst_r0 = rr0 - r0
        dst_c0 = cc0 - c0
        dst_r1 = dst_r0 + hh
        dst_c1 = dst_c0 + ww

        if tile_hwc.ndim == 2:
            tile_hwc = tile_hwc[..., None]

        ch = int(tile_hwc.shape[-1]) if tile_hwc.ndim == 3 else 1
        out = np.zeros((out_h, out_w, ch), dtype=tile_hwc.dtype)
        if hh > 0 and ww > 0:
            out[dst_r0:dst_r1, dst_c0:dst_c1, :ch] = tile_hwc[:hh, :ww, :ch]

        if out.shape[-1] == 4:
            out = out[..., :3]
        elif out.shape[-1] == 1:
            out = np.repeat(out, 3, axis=-1)
        elif out.shape[-1] < 3:
            out = np.pad(out, ((0, 0), (0, 0), (0, 3 - out.shape[-1])), mode="edge")
        elif out.shape[-1] > 3:
            out = out[..., :3]

        if out.dtype != np.uint8:
            if np.issubdtype(out.dtype, np.floating):
                out = np.clip(out, 0, 255).astype(np.uint8)
            else:
                mx = float(np.max(out)) if out.size else 0.0
                out = (out.astype(np.float32) / (mx + 1e-6) * 255.0).astype(np.uint8)

        idxs = [i for i in idxs if 0 <= i < out.shape[-1]]
        if len(idxs) == 0:
            idxs = [0, 1, 2]

        chw = np.moveaxis(out[..., idxs], -1, 0)
        return np.ascontiguousarray(chw)


def _fake_rasterio_open(path, transform=None, num_threads=None):
    return _TiffFileWindowReader(path, transform=transform, num_threads=num_threads)


class rasterio:  # noqa: N801
    Affine = _FakeAffine
    open = staticmethod(_fake_rasterio_open)


class Window:  # noqa: N801
    from_slices = staticmethod(_FakeWindow.from_slices)


torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

try:
    cv2.setNumThreads(0)
except Exception:
    pass




## === cell 1
sz = 256  # the size of tiles
reduce = 4  # reduce the original images by 4 times
TH = 0.35  # threshold for positive predictions

DATA = "../input/hubmap-kidney-segmentation/test/"
MODELS = [f"../input/b4-256-bce/efficientnet-b4-FOLD-{i}-model.pth" for i in range(5)]

df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"  # efficientnet-b4, se_resnext50_32x4d




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
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

s_th = 40  # saturation blanking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        p_tiff = os.path.join(DATA, idx + ".tiff")
        p_tif = os.path.join(DATA, idx + ".tif")
        if os.path.exists(p_tiff):
            img_path = p_tiff
        elif os.path.exists(p_tif):
            img_path = p_tif
        else:
            raise FileNotFoundError(
                f"Could not find image for id={idx} at {p_tiff} or {p_tif}"
            )

        self.data = rasterio.open(
            img_path,
            transform=identity,
            num_threads="all_cpus",
        )
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
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
        if self.data.count == 3:
            img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = np.moveaxis(
                self.data.read(
                    [1, 2, 3], window=Window.from_slices((p00, p01), (p10, p11))
                ),
                0,
                -1,
            )
        else:
            for i, layer in enumerate(self.layers):
                img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0), i] = layer.read(
                    1, window=Window.from_slices((p00, p01), (p10, p11))
                )

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // reduce, self.sz // reduce),
                interpolation=cv2.INTER_AREA,
            )

        if int(img.sum()) <= p_th:
            return img2tensor((img / 255.0 - mean) / std), -1

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        if cv2.countNonZero((s > s_th).astype(np.uint8)) <= p_th:
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
                    x = x[y >= 0].to(device, non_blocking=True)
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

                    for i in range(len(py)):
                        yield py[i], y[i]

    def __len__(self):
        return len(self.dl.dataset)




## === cell 5
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


class _UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.enc1 = _DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = _DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class HuBMAP(nn.Module):
    def __init__(self):
        super(HuBMAP, self).__init__()
        if _HAS_SMP:
            self.cnn_model = smp.Unet(model_name, encoder_weights=None, classes=1)
        else:
            self.cnn_model = _UNetSmall(in_ch=3, out_ch=1, base=32)

    def forward(self, imgs):
        img_segs = self.cnn_model(imgs)
        return img_segs




## === cell 6
models = []
missing = 0
for path in MODELS:
    model = HuBMAP()
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        del state_dict
    else:
        missing += 1
    model.float()
    model.eval()
    model.to(device)
    models.append(model)

gc.collect()
print(
    f"Loaded {len(models) - missing}/{len(models)} model weight files; missing={missing} (using random init for missing)."
)




## === cell 7
names, preds = [], []

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback

num_workers = 0
use_cuda = torch.cuda.is_available()

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]
    ds = HuBMAPDataset(idx)

    dl = DataLoader(
        ds,
        batch_size=bs,
        pin_memory=use_cuda,
        shuffle=False,
        num_workers=num_workers,
    )
    mp = Model_pred(models, dl)

    mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)

    for p, i in iter(mp):
        j = int(i) if not torch.is_tensor(i) else int(i.item())
        mask[j] = p.squeeze(-1) > TH

    mask = (
        mask.view(ds.n0max, ds.n1max, ds.sz, ds.sz)
        .permute(0, 2, 1, 3)
        .reshape(ds.n0max * ds.sz, ds.n1max * ds.sz)
    )

    mask = mask[
        ds.pad0 // 2 : -(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz,
        ds.pad1 // 2 : -(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz,
    ]

    rle = rle_encode_less_memory(mask.numpy().astype(np.uint8))
    names.append(idx)
    preds.append(rle)

    try:
        ds.data.close()
    except Exception:
        pass
    del mask, ds, dl
    gc.collect()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1544571976.py in <cell line: 0>()
     24     mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
     25 
---> 26     for p, i in iter(mp):
     27         j = int(i) if not torch.is_tensor(i) else int(i.item())
     28         mask[j] = p.squeeze(-1) > TH

/tmp/ipykernel_55/312945488.py in __iter__(self)
      8     def __iter__(self):
      9         with torch.no_grad():
---> 10             for x, y in iter(self.dl):
     11                 if (y >= 0).sum() > 0:  # exclude empty tiles
     12                     x = x[y >= 0].to(device, non_blocking=True)

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

/tmp/ipykernel_55/1223773148.py in __getitem__(self, idx)
     58         if self.data.count == 3:
     59             img[(p00 - x0) : (p01 - x0), (p10 - y0) : (p11 - y0)] = np.moveaxis(
---> 60                 self.data.read(
     61                     [1, 2, 3], window=Window.from_slices((p00, p01), (p10, p11))
     62                 ),

/tmp/ipykernel_55/3758232620.py in read(self, indexes, window)
    217         # Last resort: decode full image (may fail for JPEG without imagecodecs) -> raise a clearer error
    218         if tile_hwc is None:
--> 219             raise RuntimeError(
    220                 f"Failed to read TIFF window without full decode for {self.path}. "
    221                 f"zarr_available={self._zarr is not None}, memmap_available={self._mmap is not None}"

RuntimeError: Failed to read TIFF window without full decode for ../input/hubmap-kidney-segmentation/test/8242609fa.tiff. zarr_available=False, memmap_available=False

## === cell 8
sub = pd.DataFrame({"id": names, "predicted": preds})

sub = df_sample[["id"]].merge(sub, on="id", how="left")
sub["predicted"] = sub["predicted"].fillna("")  # empty mask allowed

sub.to_csv("submission.csv", index=False)
print(sub)
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))
