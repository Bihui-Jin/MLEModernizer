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

0.9446739630779464

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We fix the OpenCV resize error by guarding against empty source images and ensure the loop over the test set never aborts on a single failure, which also restores the correct number of rows in the submission file.'
- What this solution (achieved 0.0) has done: 'I add a lightweight fallback segmentation that thresholds the saturation channel when the loaded models produce only empty predictions, ensuring a non‑zero Dice and a valid .csv submission.'
- What this solution (achieved 0.0) has done: 'I add a simple fallback mask when the model produces no foreground (or the image cannot be read). Instead of leaving the mask empty, the code now creates a small central square mask (or uses the saturation mask if the image exists). This guarantees non‑zero predictions, turning a 0.0 Dice score into a small positive value and moving the result toward the target while preserving the original workflow.'
- What this solution (achieved 0.0) has done: 'To resolve the import error, provide a lightweight fallback for `segmentation_models_pytorch`, ensure the `shift` flag is always defined, fix the fallback mask handling for empty images, and simplify the inference loop to always produce a valid mask (using the fallback) so a correct `.csv` submission is generated quickly.'
- What this solution (achieved 0.0) has done: 'I add a lightweight inference step that uses the loaded models to predict a mask for each test image, resize the prediction back to the original image size, and only fall back to the simple saturation‑based mask when the model output is empty. This introduces the missing model‑based predictions while keeping the rest of the pipeline unchanged, moving the Dice score toward the target.'

# 9. Code solution

## === cell 0
import os
import gc
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tifffile as tiff

try:
    import segmentation_models_pytorch as smp
except ModuleNotFoundError:  # pragma: no cover

    class _DummyUnet(torch.nn.Module):
        """Returns a zero‑logit tensor with the expected shape (B,1,H,W)."""

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            B, _, H, W = x.shape
            return torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)

    class smp:  # type: ignore
        Unet = _DummyUnet


sz = 256  # tile size before up‑sampling
reduce = 4  # down‑sampling factor
TH = 0.52  # threshold for positive predictions
DATA = "../input/hubmap-kidney-segmentation/test/"
MODELS = [
    f"../input/skfoldalldata/efficientnet-b4-unet-BCELoss-256-FOLD-{i}-model.pth"
    for i in range(5)
]
df_sample = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "efficientnet-b4"
shift = True  # use the shift‑based tiling branch (kept for compatibility)
minoverlap = 300

mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])
s_th = 15  # lowered saturation threshold for fallback
p_th = 500  # enforce a minimal number of foreground pixels


def fallback_mask(full_img):
    """
    Produce a simple mask when the model yields no foreground.
    Uses a relaxed saturation threshold and Otsu on the V channel.
    Guarantees a small central square if the image cannot be read.
    """
    if full_img is None or full_img.size == 0:
        h, w = sz, sz
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = h // 2, w // 2
        sz2 = max(1, min(h, w) // 20)  # ~5 % of size
        mask[cx - sz2 : cx + sz2, cy - sz2 : cy + sz2] = 1
        return mask

    hsv = cv2.cvtColor(full_img, cv2.COLOR_BGR2HSV)
    sat_mask = (hsv[:, :, 1] > s_th).astype(np.uint8)

    _, otsu_mask = cv2.threshold(
        hsv[:, :, 2], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_mask = otsu_mask.astype(np.uint8)

    combined = np.clip(sat_mask + otsu_mask, 0, 1)

    if combined.sum() == 0:
        h, w = combined.shape
        cx, cy = h // 2, w // 2
        sz2 = max(1, min(h, w) // 20)
        combined[cx - sz2 : cx + sz2, cy - sz2 : cy + sz2] = 1
    return combined


def rle_encode_less_memory(mask):
    """
    Encode a binary mask (numpy 2‑D uint8) using run‑length encoding.
    Pixels are ordered top‑to‑bottom then left‑to‑right (column‑major).
    Returns an empty string for an empty mask.
    """
    if mask.sum() == 0:
        return ""
    flat = mask.T.ravel()
    starts = np.where((flat == 1) & (np.concatenate(([0], flat[:-1])) == 0))[0] + 1
    ends = np.where((flat == 1) & (np.concatenate((flat[1:], [0])) == 0))[0] + 1
    lengths = ends - starts + 1
    rle = " ".join(str(s) + " " + str(l) for s, l in zip(starts, lengths))
    return rle


def make_grid(shape, window, min_overlap):
    h, w = shape
    stride = max(1, window - min_overlap)
    grid = []
    for x in range(0, h, stride):
        x2 = min(x + window, h)
        for y in range(0, w, stride):
            y2 = min(y + window, w)
            grid.append((x, x2, y, y2))
    return grid


def img2tensor(img):
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return tensor


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.idx = idx
        self.path = os.path.join(DATA, idx + ".tiff")
        try:
            self.full_img = tiff.imread(self.path)  # H,W,3 (uint8)
        except Exception:
            self.full_img = np.zeros((sz, sz, 3), dtype=np.uint8)
        if self.full_img.ndim == 2:
            self.full_img = np.expand_dims(self.full_img, -1)
        self.shape = self.full_img.shape[:2]  # (H, W)
        self.reduce = reduce
        self.sz = reduce * sz  # tile size after up‑sampling
        self.mask_grid = make_grid(self.shape, window=self.sz, min_overlap=minoverlap)

    def __len__(self):
        return len(self.mask_grid)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.mask_grid[idx]
        img = self.full_img[x1:x2, y1:y2].copy()
        target_h = self.sz // self.reduce
        target_w = self.sz // self.reduce
        if img.shape[0] == 0 or img.shape[1] == 0:
            img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if self.reduce != 1 and img.shape[0] > 0 and img.shape[1] > 0:
            img = cv2.resize(
                img,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )
        vertices = torch.tensor([x1, x2, y1, y2], dtype=torch.int64)
        return img2tensor((img / 255.0 - mean) / std), vertices, idx


models = []
for path in MODELS:
    try:
        state_dict = torch.load(path, map_location="cpu")
        model = smp.Unet(model_name, encoder_weights=None, classes=1)
        model.load_state_dict(state_dict)
        model.eval().to(device)
        models.append(model)
    except Exception as e:
        print(f"Warning: could not load model {path}: {e}")

if not models:  # ensure at least one model exists (dummy that returns zeros)
    models.append(
        smp.Unet(model_name, encoder_weights=None, classes=1).eval().to(device)
    )

names, preds = [], []
for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row["id"]
    try:
        ds = HuBMAPDataset(idx)
        full_img = ds.full_img
        H, W = full_img.shape[:2]
        img_resized = cv2.resize(full_img, (sz, sz), interpolation=cv2.INTER_LINEAR)
        norm_img = (img_resized / 255.0 - mean) / std
        img_tensor = (
            torch.from_numpy(norm_img.transpose(2, 0, 1))
            .unsqueeze(0)
            .float()
            .to(device)
        )

        with torch.no_grad():
            probs = []
            for model in models:
                out = model(img_tensor)  # (1,1,sz,sz)
                prob = torch.sigmoid(out).squeeze().cpu().numpy()
                probs.append(prob)
            avg_prob = np.mean(probs, axis=0)

        mask_resized = (avg_prob > TH).astype(np.uint8)
        mask = cv2.resize(mask_resized, (W, H), interpolation=cv2.INTER_NEAREST)

        if mask.sum() < p_th:
            mask = fallback_mask(full_img)

        rle = rle_encode_less_memory(mask)
    except Exception as e:
        print(f"Error processing {idx}: {e}")
        rle = ""
    names.append(idx)
    preds.append(rle)
    del ds
    gc.collect()




## === cell 1
submission = pd.DataFrame({"img": names, "pixels": preds})
submission.to_csv("submission.csv", index=False)
display(submission)
