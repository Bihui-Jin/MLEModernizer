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

0.9328032830223908

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
import gc
from tqdm.notebook import tqdm
import rasterio
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet
from torch.nn import functional as F
from rasterio.windows import Window
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2693864695.py in <cell line: 0>()
     10 import gc
     11 from tqdm.notebook import tqdm
---> 12 import rasterio
     13 import segmentation_models_pytorch as smp
     14 from segmentation_models_pytorch import Unet

ModuleNotFoundError: No module named 'rasterio'

## === cell 1
sz = 256 # 256   #the size of tiles
reduce = 4 # 4 #reduce the original images by 4 times
TH = 0.35  #threshold for positive predictions
DATA = '../input/hubmap-kidney-segmentation/test/'
MODELS = [f'../input/b4256shiftfreezebce/efficientnet-b4-256-FOLD-{i}-model.pth' for i in range(5)]
df_sample = pd.read_csv('../input/hubmap-kidney-segmentation/sample_submission.csv')
bs = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'efficientnet-b4' # efficientnet-b4, se_resnext50_32x4d
shift = True
minoverlap = 300

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/499148736.py in <cell line: 0>()
      6 df_sample = pd.read_csv('../input/hubmap-kidney-segmentation/sample_submission.csv')
      7 bs = 32
----> 8 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      9 model_name = 'efficientnet-b4' # efficientnet-b4, se_resnext50_32x4d
     10 shift = True

NameError: name 'torch' is not defined

## === cell 2
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc): 
            continue
        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start: start + length] = 1 + m
    return img.reshape(shape).T

def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

## === cell 3
mean = np.array([0.65459856,0.48386562,0.69428385])
std = np.array([0.15167958,0.23584107,0.13146145])

s_th = 40  #saturation blancking threshold
p_th = 1000 * (sz // 256) ** 2 #threshold for the minimum number of pixels
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


def img2tensor(img, dtype:np.dtype=np.float32):
    if img.ndim == 2 : 
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2 
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


if not shift:
    class HuBMAPDataset(Dataset):
        def __init__(self, idx, sz=sz, reduce=reduce):
            self.data = rasterio.open(os.path.join(DATA, idx + '.tiff'), transform = identity,
                                     num_threads='all_cpus')
            if self.data.count != 3:
                subdatasets = self.data.subdatasets
                self.layers = []
                if len(subdatasets) > 0:
                    for i, subdataset in enumerate(subdatasets, 0):
                        self.layers.append(rasterio.open(subdataset))
            self.shape = self.data.shape
            self.reduce = reduce
            self.sz = reduce * sz
            self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz # 1024 - (25784 % 1024) = 840
            self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz # 1024 - (34937 % 1024) = 903
            self.n0max = (self.shape[0] + self.pad0) // self.sz # (25784 + 840) / 1024 = 26
            self.n1max = (self.shape[1] + self.pad1) // self.sz # (34937 + 903) / 1024 = 35

        def __len__(self):
            return self.n0max * self.n1max # 26 * 35 = 910

        def __getitem__(self, idx):
            n0, n1 = idx // self.n1max, idx % self.n1max
            x0, y0 = - self.pad0 // 2 + n0 * self.sz, - self.pad1 // 2 + n1 * self.sz
            p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
            p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])
            img = np.zeros((self.sz, self.sz, 3), np.uint8)
            if self.data.count == 3:
                img[(p00 - x0): (p01 - x0), (p10 - y0): (p11 - y0)] = np.moveaxis(self.data.read([1,2,3],
                    window=Window.from_slices((p00, p01), (p10, p11))), 0, -1)
            else:
                for i, layer in enumerate(self.layers):
                    img[(p00 - x0): (p01 - x0), (p10 - y0): (p11 - y0), i] =\
                      layer.read(1, window=Window.from_slices((p00, p01), (p10, p11)))

            if self.reduce != 1:
                img = cv2.resize(img, (self.sz // reduce, self.sz // reduce),
                                 interpolation = cv2.INTER_AREA)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            if (s > s_th).sum() <= p_th or img.sum() <= p_th:
                return img2tensor((img/255.0 - mean)/std), -1
            else: 
                return img2tensor((img/255.0 - mean)/std), idx


    class Model_pred:
        def __init__(self, models, dl, tta:bool=False, half:bool=False):
            self.models = models
            self.dl = dl
            self.tta = tta
            self.half = half

        def __iter__(self):
            count = 0
            with torch.no_grad():
                for x, y in iter(self.dl):
                    if ((y >= 0).sum() > 0): #exclude empty images
                        x = x[y >= 0].to(device)
                        y = y[y >= 0]
                        if self.half: 
                            x = x.half()
                        py = None
                        for model in self.models:
                            p = model(x)
                            p = torch.sigmoid(p).detach()
                            if py is None: 
                                py = p
                            else: 
                                py += p
                        if self.tta:
                            flips = [[-1], [-2], [-2, -1]]
                            for f in flips:
                                xf = torch.flip(x, f)
                                for model in self.models:
                                    p = model(xf)
                                    p = torch.flip(p, f)
                                    py += torch.sigmoid(p).detach()
                            py /= (1 + len(flips))        
                        py /= len(self.models)

                        py = F.upsample(py, scale_factor=reduce, mode="bilinear")
                        py = py.permute(0, 2, 3, 1).float().cpu()
                        

                        batch_size = len(py)
                        for i in range(batch_size):
                            yield py[i], y[i]
                            count += 1

        def __len__(self):
            return len(self.dl.dataset)

    models = []
    for path in MODELS:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model = smp.Unet(model_name, encoder_weights=None, classes=1)
        model.load_state_dict(state_dict)
        model.float()
        model.eval()
        model.to(device)
        models.append(model)

    del state_dict

    names, preds = [],[]
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        idx = row['id']
        ds = HuBMAPDataset(idx)
        dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
        mp = Model_pred(models, dl)
        mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
        for p, i in iter(mp): 
            mask[i.item()] = p.squeeze(-1) > TH
        mask = mask.view(ds.n0max, ds.n1max, ds.sz, ds.sz).\
            permute(0, 2, 1, 3).reshape(ds.n0max * ds.sz, ds.n1max * ds.sz)
        mask = mask[ds.pad0 // 2: - (ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz,
            ds.pad1 // 2: - (ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz]



        rle = rle_encode_less_memory(mask.numpy())
        names.append(idx)
        preds.append(rle)
        del mask, ds, dl
        gc.collect()
        
        
        
if shift:
    class HuBMAPDataset(Dataset):
        def __init__(self, idx, sz=sz, reduce=reduce):
            self.data = rasterio.open(os.path.join(DATA, idx + '.tiff'), transform=identity,
                                      num_threads='all_cpus')
            if self.data.count != 3:
                subdatasets = self.data.subdatasets
                self.layers = []
                if len(subdatasets) > 0:
                    for i, subdataset in enumerate(subdatasets, 0):
                        self.layers.append(rasterio.open(subdataset))
            self.shape = self.data.shape  # 25784*34937
            self.reduce = reduce  # 4
            self.sz = reduce * sz  # 4*256 = 1024
            self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=minoverlap)

        def __len__(self):
            return len(self.mask_grid)

        def __getitem__(self, idx):
            x1, x2, y1, y2 = self.mask_grid[idx]
            if self.data.count == 3:
                img = np.moveaxis(self.data.read([1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2))), 0, -1)
            else:
                img = np.zeros((self.sz, self.sz, 3), np.uint8)
                for i, layer in enumerate(self.layers):
                    img[:, :, i] = layer.read(1, window=Window.from_slices((x1, x2), (y1, y2)))

            if self.reduce != 1:
                img = cv2.resize(img, (self.sz // reduce, self.sz // reduce),
                                 interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            vertices = torch.tensor([x1, x2, y1, y2])
            if (s > s_th).sum() <= p_th or img.sum() <= p_th:
                return img2tensor((img / 255.0 - mean) / std), vertices, -1
            else:
                return img2tensor((img / 255.0 - mean) / std), vertices, idx


    class Model_pred:
        def __init__(self, models, dl, tta: bool = False, half: bool = False):
            self.models = models
            self.dl = dl
            self.tta = tta
            self.half = half

        def __iter__(self):
            count = 0
            with torch.no_grad():
                for x, z, y in iter(self.dl):
                    if (y >= 0).sum() > 0:  # exclude empty images
                        x = x[y >= 0].to(device)
                        z = z[y >= 0]
                        y = y[y >= 0]
                        if self.half:
                            x = x.half()
                        py = None
                        for model in self.models:
                            p = model(x)
                            p = torch.sigmoid(p).detach()
                            if py is None:
                                py = p
                            else:
                                py += p
                        if self.tta:
                            flips = [[-1], [-2], [-2, -1]]
                            for f in flips:
                                xf = torch.flip(x, f)
                                for model in self.models:
                                    p = model(xf)
                                    p = torch.flip(p, f)
                                    py += torch.sigmoid(p).detach()
                            py /= (1 + len(flips))
                        py /= len(self.models)

                        py = F.upsample(py, scale_factor=reduce, mode="bilinear")
                        py = py.permute(0, 2, 3, 1).float().cpu()

                        py = py.squeeze(-1).numpy()
                        z = z.numpy()

                        batch_size = len(py)
                        for i in range(batch_size):
                            yield py[i], z[i], y[i]
                            count += 1

        def __len__(self):
            return len(self.dl.dataset)


    models = []
    for path in MODELS:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model = smp.Unet(model_name, encoder_weights=None, classes=1)
        model.load_state_dict(state_dict)
        model.float()
        model.eval()
        model.to(device)
        models.append(model)

    del state_dict

    names, preds = [], []

    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        idx = row['id']
        ds = HuBMAPDataset(idx)
        dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=False, num_workers=0)
        mp = Model_pred(models, dl)
        mask = np.zeros(ds.shape, dtype=np.uint8)
        for pred, vert, i in iter(mp):
            x1, x2, y1, y2 = vert
            mask[x1:x2, y1:y2] += (pred > TH).astype(np.uint8)

        mask = (mask > 0.5).astype(np.uint8)

        rle = rle_encode_less_memory(mask)
        names.append(idx)
        preds.append(rle)
        del mask, ds, dl
        gc.collect()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/853650143.py in <cell line: 0>()
      5 s_th = 40  #saturation blancking threshold
      6 p_th = 1000 * (sz // 256) ** 2 #threshold for the minimum number of pixels
----> 7 identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
      8 
      9 

NameError: name 'rasterio' is not defined

## === cell 4
df = pd.DataFrame({'id':names, 'predicted':preds})
df.to_csv('submission.csv', index=False)
display(df)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4048632861.py in <cell line: 0>()
----> 1 df = pd.DataFrame({'id':names, 'predicted':preds})
      2 df.to_csv('submission.csv', index=False)
      3 display(df)

NameError: name 'names' is not defined
