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
import os
import pathlib
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split


def find_file(name):
    for root, _, files in os.walk("."):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(f"{name} not found")


train_path = find_file("train.csv")
info_path = find_file("HuBMAP-20-dataset_information.csv")
sample_path = find_file("sample_submission.csv")

sz = 256  # size to which masks are resized for averaging
TH = 0.30  # fallback threshold if tuning fails


def rle_decode_to_mask(rle_str, shape):
    """Decode a space‑separated RLE string to a binary mask of given shape."""
    if pd.isna(rle_str) or not isinstance(rle_str, str) or rle_str.strip() == "":
        return np.zeros(shape, dtype=np.uint8)
    s = np.fromstring(rle_str, sep=" ", dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]  # convert to zero‑based
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for st, ln in zip(starts, lengths):
        flat[st : st + ln] = 1
    return flat.reshape(shape, order="F")


def rle_encode_less_memory(mask):
    """Encode a binary mask (numpy 2‑D) to RLE string using column‑major order."""
    pixels = mask.flatten(order="F")
    padded = np.concatenate([[0], pixels, [0]])
    diffs = np.diff(padded)
    runs = np.where(diffs != 0)[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)


def dice_coef(pred, truth):
    """Dice coefficient for two binary masks."""
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    intersection = np.logical_and(pred, truth).sum()
    union = pred.sum() + truth.sum()
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


df_train = pd.read_csv(train_path)
df_info = pd.read_csv(info_path)

candidate_cols = [c for c in df_train.columns if c.lower() != "id"]
rle_col = None
for col in candidate_cols:
    sample_val = (
        df_train[col].dropna().astype(str).iloc[0]
        if not df_train[col].dropna().empty
        else ""
    )
    if " " in sample_val:  # typical RLE strings contain spaces
        rle_col = col
        break
if rle_col is None:
    rle_col = candidate_cols[0]  # fallback

df_info["id"] = df_info["image_file"].apply(lambda x: pathlib.Path(x).stem)
size_dict = df_info.set_index("id")[["height_pixels", "width_pixels"]].to_dict("index")
laterality_series = df_info.set_index("id")["laterality"]

masks = []
for _, row in df_train.iterrows():
    img_id = row["id"]
    shape = (sz, sz)
    if img_id in size_dict:
        h = size_dict[img_id]["height_pixels"]
        w = size_dict[img_id]["width_pixels"]
        shape = (int(h), int(w))
    masks.append(rle_decode_to_mask(row[rle_col], shape=shape))

laterality_groups = {}
for idx, row in df_train.iterrows():
    img_id = row["id"]
    lat = laterality_series.get(img_id, "unknown")
    laterality_groups.setdefault(lat, []).append(masks[idx])

mean_masks = {}
for lat, grp in laterality_groups.items():
    if not grp:
        continue
    resized = [cv2.resize(m, (sz, sz), interpolation=cv2.INTER_LINEAR) for m in grp]
    mean_masks[lat] = np.mean(np.stack(resized, axis=0), axis=0).astype(np.float32)

global_mean_mask = np.mean(
    np.stack(
        [cv2.resize(m, (sz, sz), interpolation=cv2.INTER_LINEAR) for m in masks],
        axis=0,
    ),
    axis=0,
).astype(np.float32)

indices = np.arange(len(masks))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

candidate_thresholds = np.arange(0.10, 0.51, 0.01)

laterality_thresh = {}
for lat, grp in laterality_groups.items():
    lat_val_idx = [
        i
        for i in val_idx
        if laterality_series.get(df_train.iloc[i]["id"], "unknown") == lat
    ]
    if not lat_val_idx:
        continue
    best_th = TH
    best_score = -1.0
    for cand in candidate_thresholds:
        pred_mask = (mean_masks[lat] > cand).astype(np.uint8)
        scores = [
            dice_coef(
                pred_mask,
                cv2.resize(masks[i], (sz, sz), interpolation=cv2.INTER_LINEAR),
            )
            for i in lat_val_idx
        ]
        avg_score = np.mean(scores) if scores else 0.0
        if avg_score > best_score:
            best_score = avg_score
            best_th = cand
    laterality_thresh[lat] = max(best_th, 0.05)  # enforce minimal sensible threshold

best_th = TH
best_score = -1.0
for cand in candidate_thresholds:
    pred_mask = (global_mean_mask > cand).astype(np.uint8)
    scores = [
        dice_coef(
            pred_mask,
            cv2.resize(masks[i], (sz, sz), interpolation=cv2.INTER_LINEAR),
        )
        for i in val_idx
    ]
    avg_score = np.mean(scores) if scores else 0.0
    if avg_score > best_score:
        best_score = avg_score
        best_th = cand
global_fallback_th = max(best_th, 0.05)

print(
    f"Tuned per‑laterality thresholds (fallback TH={global_fallback_th:.3f}, best global Dice≈{best_score:.4f})"
)



## === cell 1
df_sample = pd.read_csv(sample_path)
ids = df_sample["id"].tolist()

preds = []
for img_id in ids:
    shape = (sz, sz)
    if img_id in size_dict:
        h = size_dict[img_id]["height_pixels"]
        w = size_dict[img_id]["width_pixels"]
        shape = (int(h), int(w))

    laterality = laterality_series.get(img_id, "unknown")
    base_mask = mean_masks.get(laterality, global_mean_mask)
    th = laterality_thresh.get(laterality, global_fallback_th)

    resized_mask = cv2.resize(
        base_mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
    )
    binary_pred = (resized_mask > th).astype(np.uint8)

    preds.append(rle_encode_less_memory(binary_pred))

submission = pd.DataFrame({"id": ids, "predicted": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
