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

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
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

0.9499991086033368

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (not installed) by providing a small fallback `smp.Unet` implementation that preserves the same call site and `state_dict` loading behavior, so inference can run. I also fix the broken cell ordering/numbering by ensuring all required imports and variables (`pd`, `Dataset`, `dataclass`, etc.) are defined before use, which eliminates the `NameError`s. I make weight loading robust across Kaggle checkpoints (handle nested keys like `state_dict` and `module.` prefixes) without changing the model ensemble logic. Finally, I ensure the script always writes a valid `submission.csv` with the exact `id,predicted` columns required by the provided `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `rasterio` (not available in this Kaggle environment) by switching TIFF reading/tiling to `cv2.imreadmulti`, keeping the same crop/step inference logic and transform pipeline. I also fix the cascading `NameError`s by ensuring all required imports and symbols (`Dataset`, `nn`, etc.) are defined before they’re referenced, while preserving your model ensemble/inference semantics. Finally, I disable the `SKIP_COMMIT` inference skip (which is the direct reason you get an all-empty submission and ~0.0 Dice) so the notebook actually runs inference and writes a valid `submission.csv` with `id,predicted`.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import gc
from pathlib import Path
from typing import List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


class _SMPFallback:
    class Unet(nn.Module):
        """
        Minimal UNet-like model for compatibility with checkpoints.
        Used only if segmentation_models_pytorch cannot be imported.
        Returns a 1-channel logit mask.
        """

        def __init__(
            self,
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        ):
            super().__init__()
            chs = [64, 128, 256, 512]
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_channels, chs[0], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chs[0], chs[0], 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(
                nn.Conv2d(chs[0], chs[1], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chs[1], chs[1], 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.pool2 = nn.MaxPool2d(2)
            self.enc3 = nn.Sequential(
                nn.Conv2d(chs[1], chs[2], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chs[2], chs[2], 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.pool3 = nn.MaxPool2d(2)
            self.enc4 = nn.Sequential(
                nn.Conv2d(chs[2], chs[3], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chs[3], chs[3], 3, padding=1),
                nn.ReLU(inplace=True),
            )

            self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
            self.dec3 = nn.Sequential(
                nn.Conv2d(chs[2] + chs[2], chs[2], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chs[2], chs[2], 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv2d(chs[1] + chs[1], chs[1], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chs[1], chs[1], 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
            self.dec1 = nn.Sequential(
                nn.Conv2d(chs[0] + chs[0], chs[0], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chs[0], chs[0], 3, padding=1),
                nn.ReLU(inplace=True),
            )

            self.head = nn.Conv2d(chs[0], classes, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            e4 = self.enc4(self.pool3(e3))

            d3 = self.up3(e4)
            d3 = torch.cat([d3, e3], dim=1)
            d3 = self.dec3(d3)

            d2 = self.up2(d3)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.dec2(d2)

            d1 = self.up1(d2)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec1(d1)

            return self.head(d1)


try:
    sys.path.append("../input/segmentation-models-pytorch-install")
    import segmentation_models_pytorch as smp  # type: ignore
except Exception:
    smp = _SMPFallback()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True



## === cell 1
if Path("../input/hubmap-folds-2").exists():
    os.system("ls -l ../input/hubmap-folds-2")
    os.system("mkdir -p model_1")
    os.system("tar -xvf ../input/hubmap-folds-2/1024_avg_last.tar -C model_1")
    os.system("mkdir -p model_2")
    os.system("tar -xvf ../input/hubmap-folds-2/1024_512_pseudo_v1_avg.tar -C model_2")
else:
    print(
        "Warning: ../input/hubmap-folds-2 not found; will rely on any weights paths that exist."
    )



## === cell 2
BATCH_SIZE = 1
NUM_WORKERS = 0
CROP_SIZE = 1024 * 4
STEP = 1024 * 2
THR = 0.5
VOTE = 0
SKIP_BACKGROUND = True

SKIP_COMMIT = False

RAMKA = 512

df_sub = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

if "id" not in df_sub.columns:
    raise ValueError("sample_submission.csv must contain 'id' column")
if "predicted" not in df_sub.columns:
    if "pixels" in df_sub.columns:
        df_sub = df_sub.rename(columns={"pixels": "predicted"})
    else:
        df_sub["predicted"] = ""




## === cell 3
@dataclass
class ModelConfig:
    encoder: str
    weights_path: List[str]
    img_size: int
    weight_blend: float


my_models = [
    ModelConfig(
        "resnet34",
        [
            "./model_1/fold0_avg_0.9238.pth",
            "./model_1/fold1_avg_0.9162.pth",
            "./model_1/fold2_avg_0.9303.pth",
            "./model_1/fold3_avg_0.9336.pth",
            "./model_1/fold4_avg_0.9407.pth",
        ],
        4096,
        0.35,
    ),
    ModelConfig(
        "timm-efficientnet-b3",
        [
            "../input/hubmap5/exp_25_fold_0.pt",
            "../input/hubmap5/exp_24_fold_1.pt",
            "../input/hubmap5/exp_23_fold_2.pt",
            "../input/hubmap5/exp_22_fold_3.pt",
            "../input/hubmap5/exp_21_fold_4.pt",
        ],
        1024,
        0.3,
    ),
    ModelConfig(
        "se_resnext50_32x4d",
        [
            "./model_2/fold0_avg_0.9457.pth",
            "./model_2/fold1_avg_0.9566.pth",
            "./model_2/fold2_avg_0.9379.pth",
            "./model_2/fold3_avg_0.9095.pth",
            "./model_2/fold4_avg_0.9338.pth",
        ],
        2048,
        0.35,
    ),
]

TRAIN_IMG_SIZE = max([cfg.img_size for cfg in my_models])
TRAIN_IMG_SIZE




## === cell 4
def rle_encode_less_memory(img: np.ndarray) -> str:
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def valid_transform(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def _read_tiff_rgb_cv2(tiff_path: str) -> np.ndarray:
    ok, pages = cv2.imreadmulti(tiff_path, flags=cv2.IMREAD_UNCHANGED)
    if not ok or pages is None or len(pages) == 0:
        raise FileNotFoundError(f"cv2.imreadmulti failed to read: {tiff_path}")

    if len(pages) >= 3:
        chs = []
        for i in range(3):
            p = pages[i]
            if p.ndim == 3:
                p = p[:, :, 0]
            chs.append(p)
        img = np.stack(chs, axis=2)  # HWC
    else:
        img = pages[0]
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.ndim == 3 and img.shape[2] >= 3:
            img = img[:, :, :3]
        else:
            raise ValueError(f"Unexpected TIFF page shape: {img.shape}")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img




## === cell 5
def _check_background(img, crop_size) -> bool:
    s_th = 40  # saturation blanking threshold
    p_th = 1000 * (crop_size // 256) ** 2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, ss, _ = cv2.split(hsv)
    background = False if (ss > s_th).sum() <= p_th or img.sum() <= p_th else True
    return background


class SingleTiffDataset(Dataset):
    def __init__(self, tiff_path, all_img_sizes, crop_size=1024, step=512):
        self.crop_size = crop_size
        self.all_img_sizes = all_img_sizes
        self.step = step

        self.img = _read_tiff_rgb_cv2(tiff_path)
        self.h, self.w = self.img.shape[:2]

        self.row_count = 1 + math.ceil((self.h - self.crop_size) / self.step)
        self.col_count = 1 + math.ceil((self.w - self.crop_size) / self.step)

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = self.w - self.crop_size
        if y + self.crop_size > self.h:
            y = self.h - self.crop_size

        img = self.img[y : y + self.crop_size, x : x + self.crop_size]

        transformed_imgs = {
            img_size: valid_transform(img_size)(image=img)["image"]
            for img_size in self.all_img_sizes
        }
        transformed_imgs["crop_names"] = f"{x}_{y}"
        transformed_imgs["not_background"] = _check_background(img, self.crop_size)
        return transformed_imgs




## === cell 6
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def safe_load_weights(model: nn.Module, w_path: str):
    ckpt = torch.load(w_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    sd = _strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected




## === cell 7
def inference(data_loader, models_with_config, crop_size):
    img_size = (data_loader.dataset.h, data_loader.dataset.w)
    mask_pred = np.zeros(img_size, dtype=np.uint8)

    for batch in tqdm(data_loader, ncols=70, leave=True):
        if SKIP_BACKGROUND is True and bool(batch["not_background"].item()) is False:
            continue

        with torch.no_grad():
            pred_total = None
            for cfg, model_group in models_with_config:
                image = batch[cfg.img_size].to(device, non_blocking=True)
                pred_group = None

                for model in model_group:
                    pred = model(image)
                    pred = pred.sigmoid()
                    if cfg.img_size != TRAIN_IMG_SIZE:
                        pred = F.interpolate(
                            pred,
                            size=TRAIN_IMG_SIZE,
                            mode="bilinear",
                            align_corners=False,
                        )
                    pred_group = pred if pred_group is None else (pred_group + pred)

                pred_group = pred_group.squeeze()
                if len(pred_group.shape) == 2:
                    pred_group = pred_group.unsqueeze(0)

                pred_group = cfg.weight_blend * (pred_group / len(model_group))
                pred_total = (
                    pred_group if pred_total is None else (pred_total + pred_group)
                )

            pred_total = (pred_total.detach().cpu().numpy() > THR).astype(np.uint8)

            for predict_single, crop_name in zip(pred_total, batch["crop_names"]):
                x = int(crop_name.split("_")[-2])
                y = int(crop_name.split("_")[-1])

                if crop_size != TRAIN_IMG_SIZE:
                    predict_single = cv2.resize(
                        predict_single,
                        (crop_size, crop_size),
                        interpolation=cv2.INTER_NEAREST,
                    )

                predict_single[0:RAMKA] = 0
                predict_single[-RAMKA:] = 0
                predict_single[:, 0:RAMKA] = 0
                predict_single[:, -RAMKA:] = 0

                mask_pred[y : y + crop_size, x : x + crop_size] += predict_single

    mask_pred = mask_pred > VOTE
    mask_rle = rle_encode_less_memory(mask_pred.astype(np.uint8))
    del mask_pred
    gc.collect()
    gc.collect()
    return mask_rle




## === cell 8
all_img_sizes = set([cfg.img_size for cfg in my_models])
models_with_config = []

for cfg in my_models:
    models_group = []
    for w_path in cfg.weights_path:
        if not Path(w_path).exists():
            print(f"Warning: weight file not found, skipping: {w_path}")
            continue
        model = smp.Unet(cfg.encoder, encoder_weights=None).to(device)
        missing, unexpected = safe_load_weights(model, w_path)
        if missing or unexpected:
            print(
                f"Loaded {w_path} with strict=False; missing={len(missing)}, unexpected={len(unexpected)}"
            )
        model.eval()
        models_group.append(model)

    if len(models_group) > 0:
        models_with_config.append((cfg, models_group))
    else:
        print(
            f"Warning: no models loaded for config {cfg.encoder} @ img_size={cfg.img_size}"
        )

if len(models_with_config) == 0:
    print(
        "Warning: No models loaded at all. Submission will contain empty predictions."
    )



## === cell 9
count_thr = 5 if SKIP_COMMIT is True else 4
if len(df_sub) > count_thr:
    print(
        f"Note: SKIP_COMMIT=True and len(df_sub)={len(df_sub)}; skipping inference loop."
    )
else:
    for idx, row in df_sub.iterrows():
        tiff_path = f"../input/hubmap-kidney-segmentation/test/{row['id']}.tiff"
        if not Path(tiff_path).exists():
            print(f"Warning: missing test tiff {tiff_path}; predicting empty mask.")
            df_sub.loc[idx, "predicted"] = ""
            continue

        test_ds = SingleTiffDataset(
            tiff_path=tiff_path,
            all_img_sizes=all_img_sizes,
            crop_size=CROP_SIZE,
            step=STEP,
        )

        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        if len(models_with_config) == 0:
            df_sub.loc[idx, "predicted"] = ""
        else:
            rle = inference(test_loader, models_with_config, CROP_SIZE)
            df_sub.loc[idx, "predicted"] = rle



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/3332614325.py in <cell line: 0>()
     12             continue
     13 
---> 14         test_ds = SingleTiffDataset(
     15             tiff_path=tiff_path,
     16             all_img_sizes=all_img_sizes,

/tmp/ipykernel_55/1085505847.py in __init__(self, tiff_path, all_img_sizes, crop_size, step)
     14         self.step = step
     15 
---> 16         self.img = _read_tiff_rgb_cv2(tiff_path)
     17         self.h, self.w = self.img.shape[:2]
     18 

/tmp/ipykernel_55/852207467.py in _read_tiff_rgb_cv2(tiff_path)
     21 # This preserves "core logic" (tiled sliding-window inference) but changes only the IO backend.
     22 def _read_tiff_rgb_cv2(tiff_path: str) -> np.ndarray:
---> 23     ok, pages = cv2.imreadmulti(tiff_path, flags=cv2.IMREAD_UNCHANGED)
     24     if not ok or pages is None or len(pages) == 0:
     25         raise FileNotFoundError(f"cv2.imreadmulti failed to read: {tiff_path}")

error: OpenCV(4.12.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:79: error: (-215:Assertion failed) pixels <= CV_IO_MAX_IMAGE_PIXELS in function 'validateInputImageSize'


## === cell 10
if Path("model_1").exists():
    os.system("rm -rf model_1")
if Path("model_2").exists():
    os.system("rm -rf model_2")



## === cell 11
out_path = "submission.csv"
df_sub[["id", "predicted"]].to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape={df_sub.shape}")
print(df_sub.head())



## === cell 12
df_sub
