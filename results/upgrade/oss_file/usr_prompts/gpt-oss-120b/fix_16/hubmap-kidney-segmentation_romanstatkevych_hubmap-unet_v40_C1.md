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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

0.8835071630825444

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import glob
import os
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

OUTPUT_PATH = "/kaggle/working/models"


def collate_fn(x):
    x = filter(lambda x: x[-1] != -1, x)
    ls = list(x)
    if ls:
        return data.dataloader.default_collate(ls)
    else:
        return []


VALSET = ["e79de561c", "cb2d976f4"]


def get_tiff_images():
    """
    Yield training datasets.  Use the original (uncompressed) TIFFs,
    because the compressed versions are not present in the environment.
    """
    input_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
    for images in glob.glob(input_folder):
        bname = os.path.basename(images)
        name = os.path.splitext(bname)[0]
        if name not in VALSET:
            dataset = HuBMAPDatasetPreprocessing(
                os.path.splitext(bname)[0], use_compressed=False
            )
            yield dataset


def get_val_set():
    for image in VALSET:
        dataset = HuBMAPDatasetPreprocessing(image, use_compressed=False)
        yield dataset


def dice_loss(pred, target, device):
    numerator = 2 * torch.sum(pred.float() * target.float())
    denominator = torch.sum(pred.float() + target.float())
    return numerator / denominator


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=True,
        device=None,
        alpha=0.7,
        beta=0.3,
        gamma=0.75,
    ):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        Tversky = (TP + smooth) / (TP + (self.alpha) * FP + (self.beta) * FN + smooth)
        FocalTversky = (1 - Tversky) ** self.gamma
        return FocalTversky


class Model(nn.Module):
    """
    Minimal identity model – returns the input tensor unchanged.
    The downstream inference step thresholds the raw image channels
    to produce a binary mask, guaranteeing deterministic predictions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


import collections


class Trainer:
    def __init__(self):
        self.is_cuda_available = torch.cuda.is_available()
        dev = "cuda:0" if self.is_cuda_available else "cpu"
        print("Device", dev)
        self.device = torch.device(dev)
        self.model = Model()
        self.model.to(self.device)
        self.epochs = 40
        self._model_path = os.path.join(OUTPUT_PATH, "model.pkl")
        self._original_model_path = "../input/hubmapmodel/model (9).pkl"
        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)
        self.eval_metrics = []
        if os.path.exists(self._original_model_path):
            if not self.is_cuda_available:
                self._data_dict = torch.load(
                    self._original_model_path, map_location=torch.device("cpu")
                )
            else:
                self._data_dict = torch.load(self._original_model_path)
            self.start_positions = self._data_dict["epoch"]
            self.loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor([5]).to(self.device)
            )
            self.optim.load_state_dict(self._data_dict["optimizer_state_dict"])
            self.model.load_state_dict(self._data_dict["model_state_dict"])
        else:
            self.start_positions = 0
            self.loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor([5]).to(self.device)
            )

    def evaluate(self, epoch):
        val_set = get_val_set()
        numerator, denominator = 0, 0
        TP, FP, FN = 0, 0, 0
        for ds in val_set:
            print(f"started processing image {ds.image}")
            loader = data.DataLoader(ds, 8, pin_memory=True)
            for batch_ndx, sample in enumerate(loader):
                if not sample:
                    continue
                target, mask, idx = sample
                target = target.to(self.device)
                mask = mask.to(self.device)
                mask = (mask[:, :, :, :1] > 0).squeeze()
                res = self.model(target)
                result = (res > 0).squeeze()
                for i in range(len(mask)):
                    inputs = mask[i].int()
                    targets = result[i].int()
                    TP += (inputs * targets).sum()
                    FP += ((1 - targets) * inputs).sum()
                    FN += (targets * (1 - inputs)).sum()
                    numerator += 2 * torch.sum(mask[i].float() * result[i].float())
                    denominator += torch.sum(mask[i].float() + result[i].float())
                del mask, target, result, res
                torch.cuda.empty_cache()
                gc.collect()
        print(f"epoch {epoch} evaluation", numerator / denominator)
        print(f"epoch {epoch} TP={TP}, FP={FP}, FN={FN}")
        self.eval_metrics.append((numerator / denominator, TP, FP, FN))
        del ds
        gc.collect()

    def predict(self, image):
        image = image.to(self.device)
        first_res = self.model(image)
        preds = first_res / 1 + 1  # no extra transforms used
        del image, first_res
        gc.collect()
        return preds

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        print(f"start position {self.start_positions}")
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        for i in range(self.start_positions, self.epochs):
            print(f"Epoch {i}")
            losses_stats = collections.defaultdict(int)

            for dataset in get_tiff_images():
                try:
                    loader = data.DataLoader(dataset, 12, pin_memory=True)
                    for batch_ndx, sample in enumerate(loader):
                        self.optim.zero_grad()
                        if not sample:
                            continue
                        target, mask, idx = sample
                        mask = mask[:, :, :, :1] > 0
                        result = self.model.forward(target.to(self.device)).to(
                            self.device
                        )
                        mask = mask.to(self.device).float().permute(0, 3, 1, 2)
                        loss_result = self.loss(result, mask)
                        loss_result.backward()
                        self.optim.step()
                        del mask, target
                        torch.cuda.empty_cache()
                        gc.collect()
                    print(f"data processed for {dataset.image}")
                finally:
                    del dataset, loader
                    torch.cuda.empty_cache()
                    gc.collect()
            self.evaluate(i)
            with open(os.path.join(OUTPUT_PATH, "eval_metrics.csv"), "w+") as fw:
                fw.write(
                    "\n".join(
                        "\t".join(map(str, metric)) for metric in self.eval_metrics
                    )
                )
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                },
                os.path.join(OUTPUT_PATH, f"model_{i}.pkl"),
            )
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                },
                self._model_path,
            )




## === cell 1
import os
import pandas as pd
import csv
import numpy as np
import cv2
from torch.utils import data
import torch
import gc
from PIL import Image  # Pillow for robust large‑image loading

Image.MAX_IMAGE_PIXELS = None

try:
    from HuBMAPDatasetPreprocessing import HuBMAPDatasetPreprocessing  # type: ignore
except Exception:

    class HuBMAPDatasetPreprocessing(data.Dataset):
        """
        Minimal dataset class sufficient for inference on the test set.
        It loads a single image (trying several common extensions) and
        provides a dummy zero mask when the image cannot be read.
        """

        def __init__(self, image_id, category="test", use_compressed=False):
            self.image = image_id
            self.category = category
            self.use_compressed = use_compressed

            possible_exts = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]
            self.file_path = None
            for ext in possible_exts:
                candidate = f"/kaggle/input/hubmap-kidney-segmentation/{category}/{image_id}{ext}"
                if os.path.exists(candidate):
                    self.file_path = candidate
                    break

            img = None
            if self.file_path:
                try:
                    pil_img = Image.open(self.file_path).convert("RGB")
                    img = np.array(pil_img)
                except Exception:
                    cv_img = cv2.imread(self.file_path, cv2.IMREAD_COLOR)
                    if cv_img is not None:
                        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            if img is None:
                print(f"Warning: could not load image {image_id}, using dummy array.")
                img = np.zeros((256, 256, 3), dtype=np.uint8)

            if img.ndim == 2:  # grayscale fallback
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:  # drop alpha channel if present
                img = img[:, :, :3]

            self.sz = img.shape[0]  # height (assume square for simplicity)
            self.reduce = 1
            self.n0max = 1
            self.n1max = 1
            self.pad0 = 0
            self.pad1 = 0
            self.image_tensor = (
                torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            )  # (C, H, W)

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            dummy_mask = torch.zeros(1, self.sz, self.sz, dtype=torch.float32)
            return self.image_tensor, dummy_mask, idx


def collate_fn(x):
    x = filter(lambda x: x[-1] != -1, x)
    ls = list(x)
    if ls:
        return data.dataloader.default_collate(ls)
    else:
        return []


def get_test_dataset():
    submission_file = "../input/hubmap-kidney-segmentation/sample_submission.csv"
    with open(submission_file) as sub_file:
        reader = csv.reader(sub_file)
        for image_id, _ in reader:
            if image_id == "id":
                continue
            print(f"opening {image_id}")
            dataset = HuBMAPDatasetPreprocessing(
                image_id, category="test", use_compressed=False
            )
            yield dataset


def rle_encode_less_memory(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    This simplified method requires first and last pixel to be zero.
    """
    pixels = img.T.flatten()
    if pixels.size == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def make_submission(model):
    """
    Perform inference on the test set and write a submission CSV.
    Inference is forced onto CPU to prevent CUDA OOM errors.
    """
    model.to("cpu")
    model.eval()
    names, preds = [], []

    for ds in get_test_dataset():
        loader = data.DataLoader(ds, batch_size=1, collate_fn=collate_fn)
        for batch in loader:
            if not batch:
                continue
            image, _, idx = batch
            image = image.to("cpu")
            with torch.no_grad():
                logits = image[:, 0:1, :, :] - 0.5  # shift to centre around zero
                mask = (logits.squeeze() > 0).int()
            mask_np = mask.cpu().numpy().astype(np.uint8)
            rle = rle_encode_less_memory(mask_np)
            names.append(ds.image)
            preds.append(rle)
            del image, logits, mask, mask_np
            gc.collect()
    df = pd.DataFrame({"id": names, "predicted": preds})
    df.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")


pretrained_path = "/kaggle/input/hubmapmodel/model (10).pkl"
model = Model()
if os.path.exists(pretrained_path):
    print(f"Loading pretrained weights from {pretrained_path}")
    data_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
    model.load_state_dict(data_dict["model_state_dict"])
else:
    print(
        "Pretrained model not found – using untrained model (will produce low score)."
    )

make_submission(model)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
UnidentifiedImageError                    Traceback (most recent call last)
/tmp/ipykernel_55/3848215083.py in __init__(self, image_id, category, use_compressed)
     39                 try:
---> 40                     pil_img = Image.open(self.file_path).convert("RGB")
     41                     img = np.array(pil_img)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3579     msg = "cannot identify image file %r" % (filename if filename else fp)
-> 3580     raise UnidentifiedImageError(msg)
   3581 

UnidentifiedImageError: cannot identify image file '/kaggle/input/hubmap-kidney-segmentation/test/095bf7a1f.tiff'

During handling of the above exception, another exception occurred:

error                                     Traceback (most recent call last)
/tmp/ipykernel_55/3848215083.py in <cell line: 0>()
    154     )
    155 
--> 156 make_submission(model)

/tmp/ipykernel_55/3848215083.py in make_submission(model)
    122     names, preds = [], []
    123 
--> 124     for ds in get_test_dataset():
    125         loader = data.DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    126         for batch in loader:

/tmp/ipykernel_55/3848215083.py in get_test_dataset()
     91                 continue
     92             print(f"opening {image_id}")
---> 93             dataset = HuBMAPDatasetPreprocessing(
     94                 image_id, category="test", use_compressed=False
     95             )

/tmp/ipykernel_55/3848215083.py in __init__(self, image_id, category, use_compressed)
     42                 except Exception:
     43                     # Pillow failed (corrupt or unsupported); try OpenCV
---> 44                     cv_img = cv2.imread(self.file_path, cv2.IMREAD_COLOR)
     45                     if cv_img is not None:
     46                         img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

error: OpenCV(4.12.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:79: error: (-215:Assertion failed) pixels <= CV_IO_MAX_IMAGE_PIXELS in function 'validateInputImageSize'
