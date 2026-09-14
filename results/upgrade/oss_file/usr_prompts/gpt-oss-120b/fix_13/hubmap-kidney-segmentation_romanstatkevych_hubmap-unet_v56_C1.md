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

0.869651476780555

# 6. Current score

0.00624

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Implemented a lightweight test‑data loader and a simplified submission routine that no longer depends on the missing `HuBMAPDatasetPreprocessing` class. The new logic reads the sample‑submission IDs, grabs each image’s dimensions from the dataset metadata, creates a zero mask (producing a valid but empty RLE), and writes a proper `submission.csv`. This fixes the NameError and ensures the script runs end‑to‑end, producing a correctly formatted submission file.'
- What this solution (achieved 0.0) has done: 'The fixes add missing PyTorch imports, define a simple placeholder `Model` that returns zero logits (so the script runs and produces a valid RLE submission), and clean up the cell order to start at 1. These changes resolve the NameError and undefined‑model issues while keeping the original workflow intact.'
- What this solution (achieved 0.00624) has done: 'The update adds a simple Otsu‑threshold fallback that is used when the pretrained checkpoint is not found. Instead of always outputting an empty mask, the script now creates a binary mask from the original image intensities, which yields a non‑zero Dice score and moves the validation metric toward the target while keeping the original workflow unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import csv
import gc
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from skimage import io  # retained for potential fallback use


class Model(nn.Module):
    """
    Minimal placeholder model. It will only be used if a pretrained checkpoint is
    available. Otherwise an Otsu‑threshold fallback will generate masks.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch, _, H, W = x.shape
        return torch.zeros((batch, 1, H, W), device=x.device, dtype=x.dtype)


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
    input_folder = "/kaggle/input/hubmap-kidney-segmentation/train/*.tiff"
    for images in glob.glob(input_folder):
        bname = os.path.basename(images)
        name = os.path.splitext(bname)[0]
        if name not in VALSET:
            yield None


def get_val_set():
    for image in VALSET:
        yield None


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
        super().__init__()
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




## === cell 1
SAMPLE_SUBMISSION = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
METADATA_CSV = (
    "/kaggle/input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv"
)
TEST_IMAGES_DIR = "/kaggle/input/hubmap-kidney-segmentation/test"

MAX_DIM = 1024  # maximum width/height for inference to keep memory low


def load_image_shapes():
    """
    Reads the metadata CSV and returns a dict mapping image id (without extension)
    to its (height, width) tuple.
    """
    df = pd.read_csv(METADATA_CSV)
    shape_dict = {}
    for _, row in df.iterrows():
        img_id = os.path.splitext(row["image_file"])[0]
        shape_dict[img_id] = (int(row["height_pixels"]), int(row["width_pixels"]))
    return shape_dict


def rle_encode_less_memory(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as space‑separated string.
    """
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def load_and_preprocess(image_path, target_shape=None):
    """
    Loads a TIFF using OpenCV (avoids imagecodecs dependency), rescales it so the
    largest side is <= MAX_DIM, and returns a torch tensor (1,3,H,W) together
    with the scaling factor and original dimensions.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to read image {image_path}")

    if img.ndim == 2:  # gray -> replicate channels
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:  # RGBA -> drop alpha
        img = img[:, :, :3]
    elif img.shape[2] > 3:  # keep first three channels
        img = img[:, :, :3]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_np = img.astype(np.float32) / 255.0  # normalize to [0,1]

    original_h, original_w = img_np.shape[:2]
    scale = 1.0
    if max(original_h, original_w) > MAX_DIM:
        scale = MAX_DIM / max(original_h, original_w)
        new_h = int(original_h * scale)
        new_w = int(original_w * scale)
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor, scale, (original_h, original_w)


def otsu_mask_from_tensor(tensor):
    """
    Convert a normalized (0‑1) tensor back to uint8 and apply Otsu threshold.
    Returns a binary mask of the same (H,W) size as the tensor.
    """
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # H,W,3
    img_uint8 = (img_np * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)


def make_submission():
    """
    Generates a submission CSV using a pretrained U‑Net if available.
    If the checkpoint is missing, falls back to an Otsu‑threshold mask.
    """
    shape_dict = load_image_shapes()

    model = Model()
    pretrained_path = "/kaggle/input/hubmapmodel/model (13).pkl"
    use_pretrained = False
    if os.path.exists(pretrained_path):
        data_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
        if "model_state_dict" in data_dict:
            model.load_state_dict(data_dict["model_state_dict"])
            print("Loaded pretrained model for inference.")
            use_pretrained = True
        else:
            print("Pretrained file found but missing state dict – using Otsu fallback.")
    else:
        print("Pretrained model not found; using Otsu fallback (simple baseline).")

    model.eval()
    device = torch.device("cpu")
    model.to(device)

    ids = []
    preds = []

    with open(SAMPLE_SUBMISSION) as f:
        reader = csv.reader(f)
        for image_id, _ in reader:
            if image_id == "id":
                continue
            ids.append(image_id)

            tif_path = os.path.join(TEST_IMAGES_DIR, f"{image_id}.tiff")
            if not os.path.exists(tif_path):
                matches = [
                    p
                    for p in glob.glob(os.path.join(TEST_IMAGES_DIR, f"{image_id}*"))
                    if p.lower().endswith(".tiff")
                ]
                tif_path = matches[0] if matches else None

            if tif_path and os.path.exists(tif_path):
                try:
                    target_shape = shape_dict.get(image_id)
                    input_tensor, scale, (orig_h, orig_w) = load_and_preprocess(
                        tif_path, target_shape
                    )
                    input_tensor = input_tensor.to(device)

                    if use_pretrained:
                        with torch.no_grad():
                            logits = model(input_tensor)
                            prob = torch.sigmoid(logits)
                            mask_small = (
                                (prob > 0.5).squeeze().cpu().numpy().astype(np.uint8)
                            )
                    else:
                        mask_small = otsu_mask_from_tensor(input_tensor)

                    if scale < 1.0:
                        mask = cv2.resize(
                            mask_small,
                            (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    else:
                        mask = mask_small
                except Exception as e:
                    print(f"Error processing {tif_path}: {e}")
                    h, w = shape_dict.get(image_id, (256, 256))
                    mask = np.zeros((h, w), dtype=np.uint8)
            else:
                h, w = shape_dict.get(image_id, (256, 256))
                mask = np.zeros((h, w), dtype=np.uint8)

            preds.append(rle_encode_less_memory(mask))

    submission_df = pd.DataFrame({"id": ids, "predicted": preds})
    output_path = "submission.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")


make_submission()
