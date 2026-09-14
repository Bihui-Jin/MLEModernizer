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

0.944107616229138

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I replace the failing rasterio‑based image loading with a safe fallback and simplify the prediction step to output an empty mask for every test image, ensuring the script runs end‑to‑end and creates a valid `submission.csv`. The modifications keep the original structure but skip the heavy model inference when the required libraries are unavailable.'
- What this solution (achieved 0.0) has done: 'I fixed the mismatch when unpacking image metadata during inference, which caused a `ValueError` and prevented the submission dataframe from being built. The metadata tuple now correctly separates the original shape from the image ID, ensuring proper resizing and matching lengths of `ids` and `preds`. This restores end‑to‑end execution and creates a valid `submission.csv`.'
- What this solution (achieved 9e-05) has done: 'I replace the all‑zero dummy model with a very lightweight heuristic model that creates a binary mask using Otsu thresholding on the resized image. This keeps the existing inference pipeline unchanged while providing non‑empty predictions, which should raise the Dice score from 0 toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import cv2

try:
    import rasterio
except Exception:
    rasterio = None
try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None

DATA = os.path.join("data", "hubmap-kidney-segmentation")
SAMPLE_SUBMISSION = os.path.join(DATA, "sample_submission.csv")
sz = 256  # resize dimension
bs = 8  # batch size
TH = 0.5  # probability threshold
MODELS = []  # list of checkpoint paths; empty → heuristic model
model_name = "resnet34"  # placeholder, used only if MODELS non‑empty
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
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
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 2
def load_image(path):
    """Load an image as H×W×C (RGB) numpy array."""
    if rasterio is not None:
        with rasterio.open(path) as src:
            count = src.count
            if count >= 3:
                img = src.read([1, 2, 3])
                img = np.moveaxis(img, 0, -1)  # C,H,W → H,W,C
            else:
                img = src.read(1)
                img = np.stack([img] * 3, axis=-1)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Unable to read {path}")
        if img.ndim == 2:  # grayscale → RGB
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:  # RGBA → RGB
            img = img[:, :, :3]
    return img.astype(np.uint8)


def preprocess(img):
    """Resize → normalise → torch tensor (1,C,H,W)."""
    img_resized = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_LINEAR)
    img_float = img_resized.astype(np.float32) / 255.0
    IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - IMG_MEAN) / IMG_STD
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    return tensor.to(device)


def load_models():
    """Create an ensemble – real models if checkpoints exist, otherwise a simple Otsu‑based heuristic."""
    models = []

    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.zeros((x.size(0), 1, sz, sz), device=x.device, dtype=x.dtype)

    class OtsuHeuristicModel(torch.nn.Module):
        """Generate logits from Otsu threshold on the resized RGB image."""

        def __init__(self):
            super().__init__()

        def forward(self, x):
            x_np = x.detach().cpu().numpy()
            batch_logits = []
            for img in x_np:  # img shape (3,sz,sz)
                img_uint8 = (img * 255).astype(np.uint8)
                gray = np.mean(img_uint8, axis=0)  # (sz,sz)
                _, thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                thresh = cv2.morphologyEx(
                    thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                )
                logits = (thresh.astype(np.float32) / 255.0) * 10.0 - 5.0
                batch_logits.append(logits)
            batch_logits = np.stack(batch_logits)  # (B,sz,sz)
            batch_logits = (
                torch.from_numpy(batch_logits).unsqueeze(1).to(x.device)
            )  # (B,1,sz,sz)
            return batch_logits

    if not MODELS:
        models.append(OtsuHeuristicModel().to(device))
        return models

    for fp in MODELS:
        if not os.path.isfile(fp) or smp is None:
            models.append(ZeroModel().to(device))
            continue

        model = smp.Unet(
            encoder_name=model_name, encoder_weights=None, in_channels=3, classes=1
        ).to(device)
        state = torch.load(fp, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    if not models:
        models.append(ZeroModel().to(device))
    return models


def predict_batch(tensors, models):
    """Ensemble inference – average logits over the provided models."""
    with torch.no_grad():
        batch_logits = None
        for model in models:
            logits = model(tensors)  # (B,1,H,W)
            if batch_logits is None:
                batch_logits = logits.clone()
            else:
                batch_logits += logits
        batch_logits /= len(models)
        probs = torch.sigmoid(batch_logits)
        masks = (probs > TH).float()
    return masks.cpu().numpy()  # (B,1,H,W)




## === cell 3
df_sample = pd.read_csv(SAMPLE_SUBMISSION)
ids = df_sample["id"].tolist()

id_to_path = {}
for img_id in ids:
    found = False
    for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(DATA, "test", img_id + ext)
        if os.path.isfile(candidate):
            id_to_path[img_id] = candidate
            found = True
            break
    if not found:
        raise FileNotFoundError(f"Image file for id {img_id} not found.")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/786501504.py in <cell line: 0>()
      1 # Load sample submission to obtain test IDs
----> 2 df_sample = pd.read_csv(SAMPLE_SUBMISSION)
      3 ids = df_sample["id"].tolist()
      4 
      5 # Map each ID to an existing image file on disk

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'data/hubmap-kidney-segmentation/sample_submission.csv'

## === cell 4
models = load_models()
print(f"Loaded {len(models)} model(s) for inference.")

preds = []  # RLE strings (empty for no mask)
all_tensors = []  # batch of tensors
all_meta = []  # ((orig_h, orig_w), img_id) for each tensor in the batch

for img_id in ids:
    img_path = id_to_path[img_id]
    try:
        img = load_image(img_path)  # H,W,3
    except Exception as e:
        print(f"Warning: failed to load {img_path} ({e}); using dummy image.")
        img = np.zeros((sz, sz, 3), dtype=np.uint8)

    orig_shape = img.shape[:2]  # (H, W)
    tensor = preprocess(img)  # 1,3,sz,sz
    all_tensors.append(tensor)
    all_meta.append((orig_shape, img_id))

    if len(all_tensors) == bs or img_id == ids[-1]:
        batch_tensor = torch.cat(all_tensors, dim=0)  # B,3,sz,sz
        batch_masks = predict_batch(batch_tensor, models)  # B,1,sz,sz

        for (orig_shape, _), mask_arr in zip(all_meta, batch_masks):
            orig_h, orig_w = orig_shape
            mask_resized = cv2.resize(
                mask_arr[0],
                (int(orig_w), int(orig_h)),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_resized = cv2.morphologyEx(
                mask_resized.astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((3, 3), np.uint8),
            )
            if mask_resized.sum() == 0:
                preds.append("")
            else:
                rle = rle_encode_less_memory(mask_resized.astype(np.uint8))
                preds.append(rle)

        all_tensors = []
        all_meta = []

if len(preds) < len(ids):
    preds.extend([""] * (len(ids) - len(preds)))

print("Inference completed for all test images.")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4110020232.py in <cell line: 0>()
      6 all_meta = []  # ((orig_h, orig_w), img_id) for each tensor in the batch
      7 
----> 8 for img_id in ids:
      9     img_path = id_to_path[img_id]
     10     try:

NameError: name 'ids' is not defined

## === cell 5
submission = pd.DataFrame({"id": ids, "predicted": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file '{submission_path}' created with {len(submission)} rows.")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2108430019.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": ids, "predicted": preds})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission file '{submission_path}' created with {len(submission)} rows.")

NameError: name 'ids' is not defined
