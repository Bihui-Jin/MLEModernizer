# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

try:
    from torchvision.io import decode_image  # uses built-in libjpeg etc.

    _HAS_TV_DECODE = True
except Exception:
    decode_image = None
    _HAS_TV_DECODE = False


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

    Primary goal: avoid tifffile JPEG decode requiring 'imagecodecs'.
    Strategy:
      - Try tifffile memmap first (works for uncompressed/compatible TIFFs).
      - If memmap/full decode fails due to JPEG+missing imagecodecs, fall back to
        a tile-wise JPEG decode using torchvision.io.decode_image on individual tile bytes.
        This decodes only required tiles and does not need imagecodecs.

    API:
      - read([1,2,3], window=((r0,r1),(c0,c1))) returns CHW uint8.
    """

    def __init__(self, path, transform=None, num_threads=None):
        if not _HAS_TIFFFILE:
            raise RuntimeError(
                "tifffile is required for TIFF metadata access but is not available."
            )
        self.path = path
        self.transform = transform
        self.num_threads = num_threads

        self._tf = TiffFile(path)
        self._page = self._tf.pages[0]

        shape = tuple(int(x) for x in getattr(self._page, "shape", ()))
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
        self.subdatasets = []

        self._mmap = None
        self._mmap_failed = False
        try:
            self._mmap = self._page.asarray(out="memmap")
        except Exception:
            self._mmap_failed = True
            self._mmap = None

        self._can_decode_tiles = False
        self._tilewidth = None
        self._tilelength = None
        self._fh = None

        if _HAS_TV_DECODE:
            try:
                tw = getattr(self._page, "tilewidth", None)
                tl = getattr(self._page, "tilelength", None)
                if tw is not None and tl is not None and int(tw) > 0 and int(tl) > 0:
                    self._tilewidth = int(tw)
                    self._tilelength = int(tl)
                    self._fh = open(self.path, "rb")
                    self._can_decode_tiles = True
            except Exception:
                self._can_decode_tiles = False

    def close(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
        self._fh = None
        try:
            if self._tf is not None:
                self._tf.close()
        except Exception:
            pass
        self._tf = None
        self._page = None
        self._mmap = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_array(self):
        if self._mmap is not None:
            return self._mmap
        return self._page.asarray()

    def _decode_tile_jpeg_to_rgb(self, tile_index: int) -> np.ndarray:
        """
        Decode one JPEG-compressed tile as HWC uint8 (RGB).
        """
        if not self._can_decode_tiles:
            raise RuntimeError("Tile JPEG decode backend not available.")
        if self._fh is None:
            raise RuntimeError("File handle closed.")

        offsets = getattr(self._page, "dataoffsets", None)
        bytecounts = getattr(self._page, "databytecounts", None)
        if offsets is None or bytecounts is None:
            raise RuntimeError("TIFF tile offsets/bytecounts not available.")

        off = int(offsets[tile_index])
        bc = int(bytecounts[tile_index])

        self._fh.seek(off)
        b = self._fh.read(bc)

        bt = torch.from_numpy(np.frombuffer(b, dtype=np.uint8))
        img = decode_image(bt)  # CHW, uint8, typically RGB or grayscale
        if img.ndim != 3:
            raise RuntimeError(f"Unexpected decoded tile ndim={img.ndim}")

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] >= 3:
            img = img[:3]

        tile = img.permute(1, 2, 0).contiguous().cpu().numpy()
        return tile

    def _read_window_via_tiles(self, r0, r1, c0, c1) -> np.ndarray:
        """
        Assemble window using JPEG-decoded tiles. Returns HWC uint8 with 3 channels.
        """
        out_h = max(0, r1 - r0)
        out_w = max(0, c1 - c0)
        if out_h == 0 or out_w == 0:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)

        tw, tl = self._tilewidth, self._tilelength
        if tw is None or tl is None:
            raise RuntimeError("Tile shape unknown.")

        tiles_across = (self.width + tw - 1) // tw
        tiles_down = (self.height + tl - 1) // tl

        tr0 = r0 // tl
        tr1 = (r1 - 1) // tl
        tc0 = c0 // tw
        tc1 = (c1 - 1) // tw

        out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        for tr in range(tr0, tr1 + 1):
            if tr < 0 or tr >= tiles_down:
                continue
            for tc in range(tc0, tc1 + 1):
                if tc < 0 or tc >= tiles_across:
                    continue

                tile_index = tr * tiles_across + tc
                tile = self._decode_tile_jpeg_to_rgb(
                    tile_index
                )  # tl x tw x 3 (usually)

                rr0 = tr * tl
                rr1 = min(rr0 + tl, self.height)
                cc0 = tc * tw
                cc1 = min(cc0 + tw, self.width)

                ir0 = max(r0, rr0)
                ir1 = min(r1, rr1)
                ic0 = max(c0, cc0)
                ic1 = min(c1, cc1)

                if ir1 <= ir0 or ic1 <= ic0:
                    continue

                src_r0 = ir0 - rr0
                src_r1 = src_r0 + (ir1 - ir0)
                src_c0 = ic0 - cc0
                src_c1 = src_c0 + (ic1 - ic0)

                dst_r0 = ir0 - r0
                dst_r1 = dst_r0 + (ir1 - ir0)
                dst_c0 = ic0 - c0
                dst_c1 = dst_c0 + (ic1 - ic0)

                out[dst_r0:dst_r1, dst_c0:dst_c1] = tile[src_r0:src_r1, src_c0:src_c1]

        return out

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

        tile = None
        try:
            arr = self._get_array()

            if arr.ndim == 2:
                tile = arr[r0:r1, c0:c1]
                tile = tile[..., None]  # H,W,1
            elif arr.ndim == 3:
                if self._is_chw:
                    t = arr[:, r0:r1, c0:c1]  # C,h,w
                    tile = np.moveaxis(t, 0, -1)  # h,w,C
                else:
                    tile = arr[r0:r1, c0:c1, :]
            else:
                t = np.asarray(arr)[..., r0:r1, c0:c1]
                if t.ndim >= 3:
                    tile = np.moveaxis(t, 0, -1)
                else:
                    tile = t[..., None]
        except Exception as e:
            msg = str(e)
            if ("requires the 'imagecodecs' package" in msg) and self._can_decode_tiles:
                tile = self._read_window_via_tiles(r0, r1, c0, c1)
            else:
                raise

        if tile is None:
            out_h = max(0, r1 - r0)
            out_w = max(0, c1 - c0)
            c = 1 if len(idxs) == 1 else max(1, len(idxs))
            return np.zeros((c, out_h, out_w), dtype=np.uint8)

        if tile.size == 0:
            out_h = max(0, r1 - r0)
            out_w = max(0, c1 - c0)
            c = 1 if len(idxs) == 1 else max(1, len(idxs))
            return np.zeros((c, out_h, out_w), dtype=np.uint8)

        if tile.ndim == 2:
            tile = tile[..., None]

        if tile.shape[-1] == 4:
            tile = tile[..., :3]
        elif tile.shape[-1] == 1:
            tile = np.repeat(tile, 3, axis=-1)
        elif tile.shape[-1] < 3:
            tile = np.pad(tile, ((0, 0), (0, 0), (0, 3 - tile.shape[-1])), mode="edge")
        elif tile.shape[-1] > 3:
            tile = tile[..., :3]

        if tile.dtype != np.uint8:
            if np.issubdtype(tile.dtype, np.floating):
                tile = np.clip(tile, 0, 255).astype(np.uint8)
            else:
                mx = float(np.max(tile)) if tile.size else 0.0
                tile = (tile.astype(np.float32) / (mx + 1e-6) * 255.0).astype(np.uint8)

        idxs = [i for i in idxs if 0 <= i < tile.shape[-1]]
        if len(idxs) == 0:
            idxs = [0, 1, 2]

        chw = np.moveaxis(tile[..., idxs], -1, 0)
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

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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

    rle = rle_encode_less_memory(mask.numpy().astype(np.uint8))
    names.append(idx)
    preds.append(rle)

    try:
        ds.data.close()
    except Exception:
        pass
    del mask, ds, dl
    gc.collect()



## === cell 8
sub = pd.DataFrame({"id": names, "predicted": preds})

sub = df_sample[["id"]].merge(sub, on="id", how="left")
sub["predicted"] = sub["predicted"].fillna("")  # empty mask allowed

sub.to_csv("submission.csv", index=False)
print(sub)
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))
