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

0.9489723373737288

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys
sys.path.append("../input/segmentation-models-pytorch-install")
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import rasterio
from rasterio.windows import Window
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from tqdm import tqdm
import math
import gc
from typing import List, Dict
from dataclasses import dataclass

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3810322993.py in <cell line: 0>()
      1 import sys
      2 sys.path.append("../input/segmentation-models-pytorch-install")
----> 3 import segmentation_models_pytorch as smp
      4 
      5 from torch.utils.data import Dataset, DataLoader

ModuleNotFoundError: No module named 'segmentation_models_pytorch'

## === cell 2
!ls -l ../input/hubmap-folds-2
!mkdir model_1
!tar -xvf ../input/hubmap-folds-2/1024_avg_last.tar -C model_1
!mkdir model_2
!tar -xvf ../input/hubmap-folds-2/1024_512_pseudo_v1_avg.tar -C model_2

## === cell 4
BATCH_SIZE = 1
NUM_WORKERS = 0
CROP_SIZE = 1024 * 4
STEP = 1024 * 2
THR = 0.4
VOTE = 0
SKIP_BACKGROUND = True
SKIP_COMMIT = True
RAMKA = 256

df_sub = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4147576566.py in <cell line: 0>()
      9 RAMKA = 256
     10 
---> 11 df_sub = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")

NameError: name 'pd' is not defined

## === cell 6
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
        0.4
    ),
    ModelConfig(
        "timm-efficientnet-b3", 
        [
            f'../input/hubmap5/exp_25_fold_0.pt',
            f'../input/hubmap5/exp_24_fold_1.pt',
            f'../input/hubmap5/exp_23_fold_2.pt',
            f'../input/hubmap5/exp_22_fold_3.pt',
            f'../input/hubmap5/exp_21_fold_4.pt',
        ],
        1024, 
        0.2
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
        0.4
    ),
]

TRAIN_IMG_SIZE = max([cfg.img_size for cfg in my_models])
TRAIN_IMG_SIZE

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/532171653.py in <cell line: 0>()
----> 1 @dataclass
      2 class ModelConfig:
      3     encoder: str
      4     weights_path: List[str]
      5     img_size: int

NameError: name 'dataclass' is not defined

## === cell 9
def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

def valid_transform(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )

def read_from_layers(layers, window):
    if len(layers) == 1:
        return np.stack([layers[0].read(x, window=window) for x in [1, 2, 3]], 2)
    else:
        return np.stack([layers[x].read(1, window=window) for x in range(3)], 2)

## === cell 11
def _check_background(img, crop_size) -> bool:
    s_th = 40  # saturation blancking threshold
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
        dataset = rasterio.open(tiff_path, num_threads="all_cpus")
        self.h = dataset.height
        self.w = dataset.width
        self.row_count = 1 + math.ceil((self.h - self.crop_size) / self.step)
        self.col_count = 1 + math.ceil((self.w - self.crop_size) / self.step)

        self.layers = []
        if dataset.count != 3:
            subdatasets = dataset.subdatasets
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(
                        rasterio.open(subdataset, num_threads="all_cpus")
                    )
        else:
            self.layers.append(rasterio.open(tiff_path, num_threads="all_cpus"))

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = self.w - self.crop_size
        if y + self.crop_size > self.h:
            y = self.h - self.crop_size
        window = Window(x, y, self.crop_size, self.crop_size)
        img = read_from_layers(self.layers, window=window)

        transormed_imgs = {img_size: valid_transform(img_size)(image=img)['image'] for img_size in self.all_img_sizes}
        transormed_imgs["crop_names"] = f"{x}_{y}"
        transormed_imgs["not_background"] = _check_background(img, self.crop_size)
        return transormed_imgs

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3551172542.py in <cell line: 0>()
      9 
     10 # Dataset class returns multiple instances of image with different sizes
---> 11 class SingleTiffDataset(Dataset):
     12     def __init__(self, tiff_path, all_img_sizes, crop_size=1024, step=512):
     13         self.crop_size = crop_size

NameError: name 'Dataset' is not defined

## === cell 13
    
def inference(data_loader, models_with_config, crop_size):
    img_size = (data_loader.dataset.h, data_loader.dataset.w)
    mask_pred = np.zeros(img_size, dtype=np.uint8)
    for batch in tqdm(data_loader, ncols=70, leave=True):
        if SKIP_BACKGROUND is True and batch["not_background"].item() is False:
            continue
        with torch.no_grad():
            pred_total = None # accumulation for averaged by fold all models predictions
            for cfg, model_group in models_with_config:
                image = batch[cfg.img_size].cuda()
                pred_group = None # accumulation for all folds

                for model in model_group:
                    pred = model(image)
                    pred = pred.sigmoid()
                    if cfg.img_size != TRAIN_IMG_SIZE:
                        pred = torch.nn.functional.interpolate(pred, size=TRAIN_IMG_SIZE, mode='bilinear')
                    if pred_group == None:
                        pred_group = pred
                    else:
                        pred_group += pred
                        
                pred_group = pred_group.squeeze()
                if len(pred_group.shape) == 2:
                    pred_group = pred_group.unsqueeze(0)
                image.cpu()
                del image
                pred_group = cfg.weight_blend * (pred_group / len(model_group))
                if pred_total == None:
                    pred_total = pred_group
                else:
                    pred_total += pred_group

            pred_total = (pred_total.cpu().data.numpy() > THR).astype(np.uint8)
            for predict_single, crop_name in zip(pred_total, batch['crop_names']):
                x = int(crop_name.split("_")[-2])
                y = int(crop_name.split("_")[-1])

                if crop_size != TRAIN_IMG_SIZE:
                    predict_single = cv2.resize(predict_single, (crop_size, crop_size))
      
                predict_single[0:RAMKA] = 0
                predict_single[-RAMKA:] = 0
                predict_single[:, 0:RAMKA] = 0
                predict_single[:, -RAMKA:] = 0
                    
                mask_pred[y : y + crop_size, x : x + crop_size] += predict_single

    mask_pred = (mask_pred > VOTE)
    mask_rle = rle_encode_less_memory(mask_pred)
    del mask_pred, pred
    gc.collect()
    gc.collect()
    return mask_rle

## === cell 15
all_img_sizes = set([cfg.img_size for cfg in my_models])
models_with_config = []
for cfg in my_models:
    models_group = []
    for w_path in cfg.weights_path:
        model = smp.Unet(cfg.encoder, encoder_weights=None).cuda()
        model.load_state_dict(torch.load(w_path))
        model.eval()
        models_group.append(model)
    models_with_config.append((cfg, models_group))

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3560313178.py in <cell line: 0>()
----> 1 all_img_sizes = set([cfg.img_size for cfg in my_models])
      2 models_with_config = []
      3 for cfg in my_models:
      4     models_group = []
      5     for w_path in cfg.weights_path:

NameError: name 'my_models' is not defined

## === cell 17
count_thr = 5 if SKIP_COMMIT is True else 4
if len(df_sub) > count_thr:
    for idx, row in df_sub.iterrows():
        test_ds = SingleTiffDataset(
                tiff_path=f"../input/hubmap-kidney-segmentation/test/{row['id']}.tiff",
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
        rle = inference(test_loader, models_with_config, CROP_SIZE)
        df_sub.loc[idx, "predicted"] = rle

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4111393201.py in <cell line: 0>()
      1 count_thr = 5 if SKIP_COMMIT is True else 4
----> 2 if len(df_sub) > count_thr:
      3     for idx, row in df_sub.iterrows():
      4         test_ds = SingleTiffDataset(
      5                 tiff_path=f"../input/hubmap-kidney-segmentation/test/{row['id']}.tiff",

NameError: name 'df_sub' is not defined

## === cell 18
!rm -rf model_1
!rm -rf model_2


## === cell 19
df_sub.to_csv("submission.csv", index=False)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/622484759.py in <cell line: 0>()
----> 1 df_sub.to_csv("submission.csv", index=False)

NameError: name 'df_sub' is not defined

## === cell 20
df_sub

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/944308094.py in <cell line: 0>()
----> 1 df_sub

NameError: name 'df_sub' is not defined
