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
import sys

sys.path.append("../input/segmentation-models-pytorch-install")
import numpy as np
import pandas as pd
import torch
import cv2
import os
import gc
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio import Affine

    HAS_RASTERIO = True
except ModuleNotFoundError:
    HAS_RASTERIO = False
    rasterio = None
    Window = None
    Affine = None

try:
    import segmentation_models_pytorch as smp

    HAS_SMP = True
except ModuleNotFoundError:
    smp = None
    HAS_SMP = False

import tifffile
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True



## === cell 1
sz = 256
reduce = 4
TH = 0.48
DATA = "../input/hubmap-kidney-segmentation/test/"
MODELS = [
    f"../input/b4comprehensivedata256/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]
df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"
shift = False  # avoid undefined make_grid path
minoverlap = 300

INFER_HALF = True




## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc):
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


mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])
s_th = 40
p_th = 1000 * (sz // 256) ** 2
identity = Affine(1, 0, 0, 0, 1, 0) if HAS_RASTERIO else None


def img2tensor(img, dtype=np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def load_models():
    """Load ensemble models; returns empty list if smp not available."""
    if not HAS_SMP:
        return []
    loaded = []
    for path in MODELS:
        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            model = smp.Unet(model_name, encoder_weights=None, classes=1)
            model.load_state_dict(state_dict)
            model.eval().to(device)
            if INFER_HALF:
                model.half()  # half‑precision inference
            loaded.append(model)
    return loaded


def simple_fallback_mask(img):
    """Improved naive mask: Otsu threshold + removal of tiny components."""
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = img.squeeze()
    _, mask = cv2.threshold(
        gray.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    min_size = 100  # pixels; may be tuned later
    cleaned = np.zeros_like(mask)
    for i in range(1, nb_components):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[output == i] = 1
    return cleaned.astype(np.uint8)


if not HAS_RASTERIO:
    models = load_models()

    names = df_sample["id"].tolist()
    preds = []

    for img_id in tqdm(names, desc="Generating predictions (fallback)"):
        tiff_path = os.path.join(DATA, img_id + ".tiff")
        try:
            img_full = tifffile.imread(tiff_path)  # (H,W,C) or (C,H,W) or (H,W)
            if img_full.ndim == 3 and img_full.shape[0] in (1, 3):
                img_full = np.moveaxis(img_full, 0, -1)
        except Exception:
            preds.append("")
            continue

        try:
            original_shape = img_full.shape[:2]

            img_resized = cv2.resize(
                img_full,
                (sz, sz),
                interpolation=cv2.INTER_AREA,
            )
            tensor = (
                img2tensor((img_resized / 255.0 - mean) / std).unsqueeze(0).to(device)
            )
            if INFER_HALF:
                tensor = tensor.half()

            if models:
                with torch.no_grad():
                    ensemble_pred = None
                    for model in models:
                        p = torch.sigmoid(model(tensor)).detach()
                        ensemble_pred = (
                            p if ensemble_pred is None else ensemble_pred + p
                        )
                    ensemble_pred /= len(models)
                    pred_up = F.interpolate(
                        ensemble_pred,
                        scale_factor=reduce,
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_up = pred_up.squeeze().cpu().numpy()
                    mask = cv2.resize(
                        pred_up,
                        (original_shape[1], original_shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    mask_bin = (mask > TH).astype(np.uint8)
            else:
                mask_bin = simple_fallback_mask(img_full)

            rle = rle_encode_less_memory(mask_bin)
            preds.append(rle)
        except Exception:
            preds.append("")

else:
    from torch.utils.data import Dataset, DataLoader
    import multiprocessing

    if not shift:

        class HuBMAPDataset(Dataset):
            """
            Fast dataset that loads the whole TIFF once with tifffile
            and extracts tiled patches via NumPy slicing instead of many rasterio windows.
            """

            def __init__(self, idx, sz=sz, reduce=reduce):
                self.path = os.path.join(DATA, idx + ".tiff")
                img = tifffile.imread(self.path)
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.moveaxis(img, 0, -1)  # (H,W,C)
                self.img = img
                self.original_shape = img.shape[:2]  # (H, W)
                self.reduce = reduce
                self.sz = reduce * sz
                self.pad0 = (self.sz - self.original_shape[0] % self.sz) % self.sz
                self.pad1 = (self.sz - self.original_shape[1] % self.sz) % self.sz
                self.n0max = (self.original_shape[0] + self.pad0) // self.sz
                self.n1max = (self.original_shape[1] + self.pad1) // self.sz

            def __len__(self):
                return self.n0max * self.n1max

            def __getitem__(self, idx):
                n0, n1 = idx // self.n1max, idx % self.n1max
                x0 = -self.pad0 // 2 + n0 * self.sz
                y0 = -self.pad1 // 2 + n1 * self.sz

                patch = np.zeros((self.sz, self.sz, 3), np.uint8)

                x_start = max(0, x0)
                x_end = min(x0 + self.sz, self.original_shape[0])
                y_start = max(0, y0)
                y_end = min(y0 + self.sz, self.original_shape[1])

                sx = x_start - x0
                sy = y_start - y0
                patch[sx : sx + (x_end - x_start), sy : sy + (y_end - y_start)] = (
                    self.img[x_start:x_end, y_start:y_end]
                )

                if self.reduce != 1:
                    patch = cv2.resize(
                        patch,
                        (self.sz // self.reduce, self.sz // self.reduce),
                        interpolation=cv2.INTER_AREA,
                    )
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                if (hsv[..., 1] > s_th).sum() <= p_th or patch.sum() <= p_th:
                    return img2tensor((patch / 255.0 - mean) / std), -1
                else:
                    return img2tensor((patch / 255.0 - mean) / std), idx

        class Model_pred:
            def __init__(self, models, dl, tta=False, half=False):
                self.models = models
                self.dl = dl
                self.tta = tta
                self.half = half

            def __iter__(self):
                with torch.inference_mode():
                    for x, y in iter(self.dl):
                        if (y >= 0).sum() > 0:
                            x = x[y >= 0].to(device)
                            y = y[y >= 0]
                            if self.half:
                                x = x.half()
                            py = None
                            for model in self.models:
                                p = torch.sigmoid(model(x)).detach()
                                py = p if py is None else py + p
                            if self.tta:
                                flips = [[-1], [-2], [-2, -1]]
                                for f in flips:
                                    xf = torch.flip(x, f)
                                    for model in self.models:
                                        p = torch.sigmoid(model(xf)).detach()
                                        py += torch.flip(p, f)
                                py /= 1 + len(flips)
                            py /= len(self.models)
                            py = F.interpolate(py, scale_factor=reduce, mode="bilinear")
                            py = py.permute(0, 2, 3, 1).float().cpu()
                            for i in range(py.shape[0]):
                                yield py[i], y[i]

        models = load_models()

        names, preds = [], []
        cpu_count = max(1, multiprocessing.cpu_count() // 2)
        for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
            idx = row["id"]
            ds = HuBMAPDataset(idx)
            dl = DataLoader(
                ds,
                batch_size=bs,
                pin_memory=True,
                shuffle=False,
                num_workers=cpu_count,  # parallel loading of patches
            )
            mp = Model_pred(models, dl, tta=False, half=INFER_HALF)
            mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
            for p, i in mp:
                mask[i.item()] = (p.squeeze(-1) > TH).type(torch.int8)
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
            rle = rle_encode_less_memory(mask.numpy())
            names.append(idx)
            preds.append(rle)

    else:
        raise NotImplementedError(
            "Shift=True path requires make_grid, which is unavailable."
        )



## === cell 3
submission = pd.DataFrame({"id": names, "predicted": preds})
submission.to_csv("submission.csv", index=False)
display(submission.head())
