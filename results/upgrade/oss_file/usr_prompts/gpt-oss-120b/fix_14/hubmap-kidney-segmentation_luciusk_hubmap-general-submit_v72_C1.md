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

0.9405835609248696

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image
import tifffile as tiff
import glob

sz = 256  # base tile size
reduce = 1  # no down‑sampling
minoverlap = 32  # overlap between tiles
s_th = 40  # saturation threshold
p_th = 1000 * (sz // 256) ** 2  # minimum pixel count
DATA = os.path.join("data", "hubmap-kidney-segmentation", "train")
TEST_DATA = os.path.join("data", "hubmap-kidney-segmentation", "test")
device = torch.device("cpu")
TTA = False
TH = 0.4  # lowered to capture more foreground
bs = 4
model_name = "unet"  # placeholder
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])


def _load_image(path: str) -> np.ndarray:
    """Load a TIFF image safely. Fall back to Pillow, otherwise placeholder."""
    try:
        img = tiff.imread(path)
        return img
    except Exception as e_tiff:
        try:
            with Image.open(path) as im:
                img = np.array(im.convert("RGB"))
                return img
        except Exception as e_pil:
            placeholder_shape = (sz * reduce, sz * reduce, 3)
            print(
                f"Warning: could not read image {path} (tiff error: {e_tiff}; pillow error: {e_pil}); using placeholder of shape {placeholder_shape}."
            )
            return np.zeros(placeholder_shape, dtype=np.uint8)


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
    """Encode a binary mask with run‑length encoding (memory‑friendly)."""
    pixels = img.T.flatten()
    if pixels.size == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    x, y = shape
    window = min(window, x, y)
    min_overlap = min(min_overlap, max(0, window - 1))
    nx = max(1, x // (window - min_overlap) + 1)
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = max(0, x - window)
    x2 = (x1 + window).clip(0, x)
    ny = max(1, y // (window - min_overlap) + 1)
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = max(0, y - window)
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


class SimpleZeroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros(
            (x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype
        )


class SimpleThreshModel(torch.nn.Module):
    """Predict mask logits from image saturation; higher logits → foreground."""

    def __init__(self, sat_thr=s_th, pos_logit=5.0, neg_logit=-5.0):
        super().__init__()
        self.sat_thr = sat_thr
        self.pos_logit = pos_logit
        self.neg_logit = neg_logit
        self.mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        self.std_t = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        x = x.to(self.mean_t.device)
        img = x * self.std_t.to(x.device) + self.mean_t.to(x.device)
        img = (img * 255).clamp(0, 255).byte()
        img_np = img.permute(0, 2, 3, 1).cpu().numpy()  # B H W C (RGB)

        batch_masks = []
        for im in img_np:
            hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            sat = hsv[..., 1]
            mask = np.where(sat > self.sat_thr, self.pos_logit, self.neg_logit).astype(
                np.float32
            )
            batch_masks.append(mask)
        batch_masks = np.stack(batch_masks)  # B H W
        out = torch.from_numpy(batch_masks).unsqueeze(1).to(x.device)  # B 1 H W
        return out


models = [SimpleThreshModel().to(device).eval()]




## === cell 1
class HuBMAPDataset(Dataset):
    """
    Loads a whole‑slide image (train or test) and slices it into overlapping tiles.
    The constructor now searches both the training and test directories so that
    inference can run on the proper test images.
    """

    def __init__(self, idx, sz=sz, reduce=reduce):
        possible_paths = [
            os.path.join(DATA, f"{idx}.*"),
            os.path.join(TEST_DATA, f"{idx}.*"),
        ]
        matches = []
        for pattern in possible_paths:
            matches.extend(glob.glob(pattern))
        if not matches:
            print(f"Warning: No image found for id {idx}; using placeholder.")
            self.full_img = np.zeros((sz, sz, 3), dtype=np.uint8)
        else:
            img_path = matches[0]
            self.full_img = _load_image(img_path)  # H x W x C or H x W

        if self.full_img.ndim == 2:
            self.full_img = np.stack([self.full_img] * 3, axis=-1)
        self.shape = self.full_img.shape[:2]  # (H, W)
        self.reduce = reduce
        self.sz = reduce * sz
        self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=minoverlap)

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        img = self.full_img[x1:x2, y1:y2, :].copy()
        if img.size == 0:
            img = np.zeros((self.sz, self.sz, 3), dtype=self.full_img.dtype)

        if self.reduce != 1:
            img = cv2.resize(
                img,
                (self.sz // self.reduce, self.sz // self.reduce),
                interpolation=cv2.INTER_AREA,
            )
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        vertices = torch.tensor([x1, x2, y1, y2])
        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img2tensor((img / 255.0 - mean) / std), vertices, -1
        else:
            return img2tensor((img / 255.0 - mean) / std), vertices, idx




## === cell 2
class Model_pred:
    def __init__(self, models, dl, tta: bool = TTA, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        with torch.no_grad():
            for x, z, y in iter(self.dl):
                x = x.to(device)
                z = z
                y = y
                if self.half:
                    x = x.half()
                py = None
                for model in self.models:
                    p = model(x)
                    p = torch.sigmoid(p).detach()
                    py = p if py is None else py + p
                if self.tta:
                    flips = [[-1], [-2], [-2, -1]]
                    for f in flips:
                        xf = torch.flip(x, f)
                        for model in self.models:
                            p = model(xf)
                            p = torch.flip(p, f)
                            py += torch.sigmoid(p).detach()
                    py /= 1 + len(flips)
                py /= len(self.models) if self.models else 1

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




## === cell 3
def get_test_ids():
    sample_path = os.path.join(
        "data", "hubmap-kidney-segmentation", "sample_submission.csv"
    )
    if os.path.isfile(sample_path):
        df = pd.read_csv(sample_path, usecols=["id"])
        return df["id"].tolist()
    test_dir = os.path.join("data", "hubmap-kidney-segmentation", "test")
    if os.path.isdir(test_dir):
        ids = [
            os.path.splitext(f)[0]
            for f in os.listdir(test_dir)
            if f.lower().endswith((".tiff", ".tif", ".png", ".jpg", ".jpeg"))
        ]
        if ids:
            return ids
    train_csv = os.path.join("data", "hubmap-kidney-segmentation", "train.csv")
    if os.path.isfile(train_csv):
        df = pd.read_csv(train_csv, usecols=["id"])
        return df["id"].tolist()
    return []


ids = get_test_ids()
names, preds = [], []

for idx in tqdm(ids, desc="Generating predictions"):
    ds = HuBMAPDataset(idx)
    dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
    mp = Model_pred(models, dl)
    mask = np.zeros(ds.shape, dtype=np.uint8)

    for pred, vert, _ in mp:
        x1, x2, y1, y2 = vert
        mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

    mask = (mask > 0.5).astype(np.uint8)
    rle = rle_encode_less_memory(mask)
    names.append(idx)
    preds.append(rle)

    del mask, ds, dl, mp
    gc.collect()




## === cell 4
submission = pd.DataFrame({"id": names, "predicted": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())
