# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify species of images of animals from camera traps.

## Metric
Categorization accuracy.

## Submission Format
```
Id,Predicted
58857ccf-23d2-11e8-a6a3-ec086b02610b,1
591e4006-23d2-11e8-a6a3-ec086b02610b,5
```

The `Id` column corresponds to the test image id. The `Category` is an integer value that indicates the class of the animal, or `0` to represent the absence of an animal.

## Dataset
Supplementary training data can be downloaded from https://github.com/visipedia/iwildcam_comp.

### Camera Trap Animal Detection Model
We are also providing a general animal detection model.

The model is a TensorFlow Faster-RCNN model with Inception-Resnet-v2 backbone and atrous convolution.

The model and sample code for running the detector over a folder of images can be found [here](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md).

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
google-ai-generativelanguage==0.6.15
google-api-core==2.28.1
google-api-python-client==2.177.0
google-auth==2.38.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==1.2.2
google-genai==1.48.0
google-generativeai==0.8.5
googleapis-common-protos==1.70.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pydata-google-auth==1.9.1
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (89 lines)
            iwildcam2020_megadetector_results.json (1 lines)
            iwildcam2020_test_information.json (1 lines)
            iwildcam2020_train_annotations.json (1 lines)
            sample_submission.csv (60761 lines)
            sample_submission.csv.zip (461.6 kB)
            test.zip (24.3 GB)
            train.zip (65.8 GB)
            iwildcam-2020-fgvc7/
                description.md (89 lines)
                iwildcam2020_megadetector_results.json (1 lines)
                ... and 6 other files
                iwildcam-2020-fgvc7/
                test/
                    8bf6870e-21bc-11ea-a13a-137349068a90.jpg (221.6 kB)
                    8d05adf0-21bc-11ea-a13a-137349068a90.jpg (294.3 kB)
                    ... and 60758 other files
                    test/
                train/
                    8a9da342-21bc-11ea-a13a-137349068a90.jpg (242.3 kB)
                    8fa04548-21bc-11ea-a13a-137349068a90.jpg (177.8 kB)
                    ... and 157197 other files
                    train/
            test/
                8bf6870e-21bc-11ea-a13a-137349068a90.jpg (221.6 kB)
                8d05adf0-21bc-11ea-a13a-137349068a90.jpg (294.3 kB)
                ... and 60758 other files
                test/
            train/
                8a9da342-21bc-11ea-a13a-137349068a90.jpg (242.3 kB)
                8fa04548-21bc-11ea-a13a-137349068a90.jpg (177.8 kB)
                ... and 157197 other files
                train/
        input/
            description.md (89 lines)
            iwildcam2020_megadetector_results.json (1 lines)
            iwildcam2020_test_information.json (1 lines)
            iwildcam2020_train_annotations.json (1 lines)
            sample_submission.csv (60761 lines)
            sample_submission.csv.zip (461.6 kB)
            test.zip (24.3 GB)
            train.zip (65.8 GB)
            iwildcam-2020-fgvc7/
                description.md (89 lines)
                iwildcam2020_megadetector_results.json (1 lines)
                ... and 6 other files
                iwildcam-2020-fgvc7/
                test/
                    8bf6870e-21bc-11ea-a13a-137349068a90.jpg (221.6 kB)
                    8d05adf0-21bc-11ea-a13a-137349068a90.jpg (294.3 kB)
                    ... and 60758 other files
                    test/
                train/
                    8a9da342-21bc-11ea-a13a-137349068a90.jpg (242.3 kB)
                    8fa04548-21bc-11ea-a13a-137349068a90.jpg (177.8 kB)
                    ... and 157197 other files
                    train/
            test/
                8bf6870e-21bc-11ea-a13a-137349068a90.jpg (221.6 kB)
                8d05adf0-21bc-11ea-a13a-137349068a90.jpg (294.3 kB)
                ... and 60758 other files
                test/
                    8bf6870e-21bc-11ea-a13a-137349068a90.jpg (221.6 kB)
                    8d05adf0-21bc-11ea-a13a-137349068a90.jpg (294.3 kB)
                    ... and 60758 other files
                    test/
            train/
                8a9da342-21bc-11ea-a13a-137349068a90.jpg (242.3 kB)
                8fa04548-21bc-11ea-a13a-137349068a90.jpg (177.8 kB)
                ... and 157197 other files
                train/
                    8a9da342-21bc-11ea-a13a-137349068a90.jpg (242.3 kB)
                    8fa04548-21bc-11ea-a13a-137349068a90.jpg (177.8 kB)
                    ... and 157197 other files
                    train/
        working/
            iwildcam-2020-fgvc7/
                description.md (89 lines)
                iwildcam2020_megadetector_results.json (1 lines)
                ... and 6 other files
                iwildcam-2020-fgvc7/
                test/
                    8bf6870e-21bc-11ea-a13a-137349068a90.jpg (221.6 kB)
                    8d05adf0-21bc-11ea-a13a-137349068a90.jpg (294.3 kB)
                    ... and 60758 other files
                    test/
                train/
                    8a9da342-21bc-11ea-a13a-137349068a90.jpg (242.3 kB)
                    8fa04548-21bc-11ea-a13a-137349068a90.jpg (177.8 kB)
                    ... and 157197 other files
                    train/
```

-> data/iwildcam-2020-fgvc7/iwildcam2020_megadetector_results.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "info": {
      "type": "object",
      "properties": {
        "format_version": {
          "type": "string"
        },
        "detector": {
          "type": "string"
        },
        "detection_completion_time": {
          "type": "string"
        }
      },
      "required": [
        "detection_completion_time",
        "detector",
        "format_version"
      ]
    },
    "images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "detections": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "category": {
                  "type": "string"
                },
                "bbox": {
                  "type": "array",
                  "items": {
                    "type": "number"
                  }
                },
                "conf": {
                  "type": "number"
                }
              },
              "required": [
                "bbox",
                "category",
                "conf"
              ]
            }
          },
          "id": {
            "type": "string"
          },
          "max_detection_conf": {
            "type": "number"
          }
        },
        "required": [
          "detections",
          "id",
          "max_detection_conf"
        ]
      }
    },
    "detection_categories": {
      "type": "object",
      "properties": {
        "2": {
          "type": "string"
        },
        "1": {
          "type": "string"
        }
      },
      "required": [
        "1",
        "2"
      ]
    }
  },
  "required": [
    "detection_categories",
    "images",
    "info"
  ]
}

-> data/iwildcam-2020-fgvc7/iwildcam2020_test_information.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "seq_num_frames": {
            "type": "integer"
          },
          "location": {
            "type": "integer"
          },
          "datetime": {
            "type": "string"
          },
          "id": {
            "type": "string"
          },
          "frame_num": {
            "type": "integer"
          },
          "seq_id": {
            "type": "string"
          },
          "width": {
            "type": "integer"
          },
          "height": {
            "type": "integer"
          },
          "file_name": {
            "type": "string"
          }
        },
        "required": [
          "datetime",
          "file_name",
          "frame_num",
          "height",
          "id",
          "location",
          "seq_id",
          "seq_num_frames",
          "width"
        ]
      }
    },
    "categories": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "count": {
            "type": "integer"
          },
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          }
        },
        "required": [
          "count",
          "id",
          "name"
        ]
      }
    },
    "info": {
      "type": "object",
      "properties": {
        "year": {
          "type": "string"
        },
        "description": {
          "type": "string"
        },
        "version": {
          "type": "string"
        },
        "contributor": {
          "type": "string"
        }
      },
      "required": [
        "contributor",
        "description",
        "version",
        "year"
      ]
    }
  },
  "required": [
    "categories",
    "images",
    "info"
  ]
}

-> data/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "annotations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "count": {
            "type": "integer"
          },
          "image_id": {
            "type": "string"
          },
          "id": {
            "type": "string"
          },
          "category_id": {
            "type": "integer"
          }
        },
        "required": [
          "category_id",
          "count",
          "id",
          "image_id"
        ]
      }
    },
    "images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "seq_num_frames": {
            "type": "integer"
          },
          "location": {
            "type": "integer"
          },
          "datetime": {
            "type": "string"
          },
          "id": {
            "type": "string"
          },
          "frame_num": {
            "type": "integer"
          },
          "seq_id": {
            "type": "string"
          },
          "width": {
            "type": "integer"
          },
          "height": {
            "type": "integer"
          },
          "file_name": {
            "type": "string"
          }
        },
        "required": [
          "datetime",
          "file_name",
          "frame_num",
          "height",
          "id",
          "location",
          "seq_id",
          "seq_num_frames",
          "width"
        ]
      }
    },
    "categories": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "count": {
            "type": "integer"
          },
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          }
        },
        "required": [
          "count",
          "id",
          "name"
        ]
      }
    },
    "info": {
      "type": "object",
      "properties": {
        "year": {
          "type": "string"
        },
        "description": {
          "type": "string"
        },
        "version": {
          "type": "string"
        },
        "contributor": {
          "type": "string"
        }
      },
      "required": [
        "contributor",
        "description",
        "version",
        "year"
      ]
    }
  },
  "required": [
    "annotations",
    "categories",
    "images",
    "info"
  ]
}

-> data/iwildcam-2020-fgvc7/sample_submission.csv has 60760 rows and 2 columns.
The columns are: Id, Category

-> data/iwildcam2020_megadetector_results.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "info": {
      "type": "object",
      "properties": {
        "format_version": {
          "type": "string"
        },
        "detector": {
          "type": "string"
        },
        "detection_completion_time": {
          "type": "string"
        }
      },
      "required": [
        "detection_completion_time",
        "detector",
        "format_version"
      ]
    },
    "images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "detections": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "category": {
                  "type": "string"
                },
                "bbox": {
                  "type": "array",
                  "items": {
                    "type": "number"
                  }
                },
                "conf": {
                  "type": "number"
                }
              },
              "required": [
                "bbox",
                "category",
                "conf"
              ]
            }
          },
          "id": {
            "type": "string"
          },
          "max_detection_conf": {
            "type": "number"
          }
        },
        "required": [
          "detections",
          "id",
          "max_detection_conf"
        ]
      }
    },
    "detection_categories": {
      "type": "object",
      "properties": {
        "2": {
          "type": "string"
        },
        "1": {
          "type": "string"
        }
      },
      "required": [
        "1",
        "2"
      ]
    }
  },
  "required": [
    "detection_categories",
    "images",
    "info"
  ]
}

-> data/iwildcam2020_test_information.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "seq_num_frames": {
            "type": "integer"
          },
          "location": {
            "type": "integer"
          },
          "datetime": {
            "type": "string"
          },
          "id": {
            "type": "string"
          },
          "frame_num": {
            "type": "integer"
          },
          "seq_id": {
            "type": "string"
          },
          "width": {
            "type": "integer"
          },
          "height": {
            "type": "integer"
          },
          "file_name": {
            "type": "string"
          }
        },
        "required": [
          "datetime",
          "file_name",
          "frame_num",
          "height",
          "id",
          "location",
          "seq_id",
          "seq_num_frames",
          "width"
        ]
      }
    },
    "categories": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "count": {
            "type": "integer"
          },
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          }
        },
        "required": [
          "count",
          "id",
          "name"
        ]
      }
    },
    "info": {
      "type": "object",
      "properties": {
        "year": {
          "type": "string"
        },
        "description": {
          "type": "string"
        },
        "version": {
          "type": "string"
        },
        "contributor": {
          "type": "string"
        }
      },
      "required": [
        "contributor",
        "description",
        "version",
        "year"
      ]
    }
  },
  "required": [
    "categories",
    "images",
    "info"
  ]
}

-> data/iwildcam2020_train_annotations.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "annotations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "count": {
            "type": "integer"
          },
          "image_id": {
            "type": "string"
          },
          "id": {
            "type": "string"
          },
          "category_id": {
            "type": "integer"
          }
        },
        "required": [
          "category_id",
          "count",
          "id",
          "image_id"
        ]
      }
    },
    "images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "seq_num_frames": {
            "type": "integer"
          },
          "location": {
            "type": "integer"
          },
          "datetime": {
            "type": "string"
          },
          "id": {
            "type": "string"
          },
          "frame_num": {
            "type": "integer"
          },
          "seq_id": {
            "type": "string"
          },
          "width": {
            "type": "integer"
          },
          "height": {
            "type": "integer"
          },
          "file_name": {
            "type": "string"
          }
        },
        "required": [
          "datetime",
          "file_name",
          "frame_num",
          "height",
          "id",
          "location",
          "seq_id",
          "seq_num_frames",
          "width"
        ]
      }
    },
    "categories": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "count": {
            "type": "integer"
          },
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          }
        },
        "required": [
          "count",
          "id",
          "name"
        ]
      }
    },
    "info": {
      "type": "object",
      "properties": {
        "year": {
          "type": "string"
        },
        "description": {
          "type": "string"
        },
        "version": {
          "type": "string"
        },
        "contributor": {
          "type": "string"
        }
      },
      "required": [
        "contributor",
        "description",
        "version",
        "year"
      ]
    }
  },
  "required": [
    "annotations",
    "categories",
    "images",
    "info"
  ]
}

-> data/sample_submission.csv has 60760 rows and 2 columns.
The columns are: Id, Category

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, sys, json, math, random, gc, glob
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision
from torchvision import transforms


def seed_everything(seed: int = 42):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def is_gpu_available():
    return torch.cuda.is_available()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.set_num_interop_threads(1)
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")




## === cell 1
def is_in_ipython():
    program_name = os.path.basename(os.getenv("_", ""))
    return (
        ("jupyter-notebook" in program_name)
        or ("ipython" in program_name)
        or ("jupyter" in program_name)
        or ("JPY_PARENT_PID" in os.environ)
    )


def is_in_colab():
    if not is_in_ipython():
        return False
    try:
        import google.colab  # noqa

        return True
    except Exception:
        return False


def is_in_kaggle_kernal():
    return bool(
        os.environ.get("KAGGLE_URL_BASE") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    )


print("in_kaggle:", is_in_kaggle_kernal(), "in_colab:", is_in_colab())



## === cell 2
BASE_INPUT_CANDIDATES = [
    "../input/iwildcam-2020-fgvc7",
    "/kaggle/input/iwildcam-2020-fgvc7",
    "/kaggle/data/iwildcam-2020-fgvc7",
]
BASE_INPUT = None
for p in BASE_INPUT_CANDIDATES:
    if os.path.exists(p):
        BASE_INPUT = p
        break
if BASE_INPUT is None:
    raise FileNotFoundError(
        "Cannot find iwildcam-2020-fgvc7 input folder in expected locations."
    )

TRAIN_JSON = os.path.join(BASE_INPUT, "iwildcam2020_train_annotations.json")
TEST_JSON = os.path.join(BASE_INPUT, "iwildcam2020_test_information.json")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")

TRAIN_DIR_CANDIDATES = [
    os.path.join(BASE_INPUT, "train"),
    os.path.join(os.path.dirname(BASE_INPUT), "train"),
    "/kaggle/data/train",
]
TEST_DIR_CANDIDATES = [
    os.path.join(BASE_INPUT, "test"),
    os.path.join(os.path.dirname(BASE_INPUT), "test"),
    "/kaggle/data/test",
]

TRAIN_DIR = next((p for p in TRAIN_DIR_CANDIDATES if os.path.exists(p)), None)
TEST_DIR = next((p for p in TEST_DIR_CANDIDATES if os.path.exists(p)), None)
if TRAIN_DIR is None or TEST_DIR is None:
    raise FileNotFoundError(
        f"Cannot find train/test image directories. TRAIN_DIR={TRAIN_DIR}, TEST_DIR={TEST_DIR}"
    )

print("BASE_INPUT:", BASE_INPUT)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("sample_submission exists:", os.path.exists(SAMPLE_SUB))



## === cell 3
with open(TRAIN_JSON, "r") as f:
    train_data = json.load(f)
with open(TEST_JSON, "r") as f:
    test_data = json.load(f)

train_images_df = pd.DataFrame(
    train_data["images"], columns=["id", "location", "file_name"]
)
train_ann_df = pd.DataFrame(
    train_data["annotations"], columns=["image_id", "category_id"]
)

train_df = train_ann_df.merge(
    train_images_df,
    left_on="image_id",
    right_on="id",
    how="left",
    suffixes=("_ann", "_img"),
)

df_train = train_df.loc[:, ["image_id", "category_id", "location", "file_name"]].copy()
df_train["image_path"] = (TRAIN_DIR + "/") + df_train["file_name"].astype(str)

test_images_df = pd.DataFrame(
    test_data["images"], columns=["id", "location", "file_name"]
)
df_test = test_images_df.copy()
df_test.rename(columns={"id": "image_id"}, inplace=True)
df_test["image_path"] = (TEST_DIR + "/") + df_test["file_name"].astype(str)

print(df_train.shape, df_test.shape)
print(df_train.head(2))



## === cell 4
df_category_train = pd.DataFrame(train_data["categories"])
df_category_test = pd.DataFrame(test_data["categories"])

animal_category_lists_train = sorted(df_train["category_id"].unique().tolist())
print("num train categories in annotations:", len(animal_category_lists_train))
print(
    "min/max category id:",
    min(animal_category_lists_train),
    max(animal_category_lists_train),
)

category2label = OrderedDict(
    (cid, i) for i, cid in enumerate(animal_category_lists_train)
)
label2category = OrderedDict((i, cid) for cid, i in category2label.items())

df_train["label"] = df_train["category_id"].map(category2label).astype(np.int64)

print("first labels:", df_train["label"].head().tolist())
print("label2category head:", list(label2category.items())[:5])



## === cell 5
from torchvision.io import ImageReadMode
from torchvision.io import decode_jpeg
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T

train_tfms = T.Compose(
    [
        T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        T.ToDtype(torch.float32, scale=True),  # uint8 -> float32 in [0,1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

test_tfms = T.Compose(
    [
        T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class IWildcamDataset(Dataset):
    def __init__(self, df, tfms, is_train=True, cache_resized_uint8=False):
        self.df = df.reset_index(drop=True)
        self.tfms = tfms
        self.is_train = is_train

        self.paths = self.df["image_path"].astype(str).to_numpy()
        self.ids = self.df["image_id"].astype(str).to_numpy()
        if self.is_train:
            self.labels = self.df["label"].astype(np.int64).to_numpy()
        else:
            self.labels = None

        self.cache_resized_uint8 = bool(cache_resized_uint8)
        self._cache = None  # initialized lazily per worker process

    def __len__(self):
        return len(self.paths)

    def _get_worker_cache(self):
        if not self.cache_resized_uint8:
            return None
        if self._cache is None:
            self._cache = {}
        return self._cache

    def _load_rgb_tensor_uint8(self, path: str):
        try:
            if (
                (path is None)
                or (not isinstance(path, str))
                or (not os.path.exists(path))
            ):
                return torch.zeros((3, 224, 224), dtype=torch.uint8)
            with open(path, "rb") as f:
                b = f.read()
            x = torch.frombuffer(b, dtype=torch.uint8)
            return decode_jpeg(x, mode=ImageReadMode.RGB)  # uint8 CxHxW
        except Exception:
            return torch.zeros((3, 224, 224), dtype=torch.uint8)

    def __getitem__(self, idx):
        cache = self._get_worker_cache()
        if cache is not None:
            x = cache.get(idx, None)
            if x is None:
                x = self._load_rgb_tensor_uint8(self.paths[idx])
                if len(cache) > 4096:
                    cache.pop(next(iter(cache)))
                cache[idx] = x
        else:
            x = self._load_rgb_tensor_uint8(self.paths[idx])

        x = self.tfms(x)
        if self.is_train:
            return x, int(self.labels[idx])
        else:
            return x, self.ids[idx]




## === cell 6
num_classes = len(animal_category_lists_train)


def build_effnet_b0(num_classes: int):
    m = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def build_vit_b16(num_classes: int):
    m = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    )
    in_features = m.heads.head.in_features
    m.heads.head = nn.Linear(in_features, num_classes)
    return m


net1 = build_effnet_b0(num_classes).to(device)
vit = build_vit_b16(num_classes).to(device)

print("models built:", num_classes, "classes")

for p in net1.parameters():
    p.requires_grad = False
for p in net1.classifier[1].parameters():
    p.requires_grad = True

for p in vit.parameters():
    p.requires_grad = False
for p in vit.heads.head.parameters():
    p.requires_grad = True

if device.type == "cuda":
    net1 = net1.to(memory_format=torch.channels_last)
    vit = vit.to(memory_format=torch.channels_last)


def maybe_compile(m):
    return m


net1 = maybe_compile(net1)
vit = maybe_compile(vit)




## === cell 7
class CUDAPrefetcher:
    def __init__(self, loader, device, channels_last=False):
        self.loader = loader
        self.device = device
        self.channels_last = channels_last and (device.type == "cuda")
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch
            return

        it = iter(self.loader)
        stream = self.stream

        def _to_device(batch):
            if isinstance(batch, (list, tuple)):
                xb, yb = batch
            else:
                raise TypeError("Unexpected batch type")
            xb = xb.to(self.device, non_blocking=True)
            if self.channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            if torch.is_tensor(yb):
                yb = yb.to(self.device, non_blocking=True)
            return xb, yb

        with torch.cuda.stream(stream):
            next_batch = next(it, None)
            if next_batch is not None:
                next_batch = _to_device(next_batch)

        while next_batch is not None:
            torch.cuda.current_stream().wait_stream(stream)
            batch = next_batch
            with torch.cuda.stream(stream):
                next_batch = next(it, None)
                if next_batch is not None:
                    next_batch = _to_device(next_batch)
            yield batch


_cpu = os.cpu_count() or 4
if device.type == "cuda":
    _num_workers = min(8, max(2, _cpu // 2))
    _prefetch = 8
else:
    _num_workers = min(4, max(2, _cpu // 4))
    _prefetch = 2 if _num_workers > 0 else None

max_train_samples = min(30000, len(df_train))
train_df_small = df_train.sample(n=max_train_samples, random_state=42).reset_index(
    drop=True
)

train_ds = IWildcamDataset(
    train_df_small, train_tfms, is_train=True, cache_resized_uint8=False
)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=_prefetch,
    drop_last=True,
)


def train_one_epoch(model, loader, lr=3e-4):
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    loader_it = (
        CUDAPrefetcher(loader, device, channels_last=True)
        if device.type == "cuda"
        else loader
    )
    for xb, yb in loader_it:
        if device.type != "cuda":
            xb = xb.to(device)
            yb = yb.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total += bs
        correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / max(1, total), correct / max(1, total)


for name, model, lr in [("net1", net1, 3e-4), ("vit", vit, 2e-4)]:
    loss, acc = train_one_epoch(model, train_loader, lr=lr)
    print(f"{name}: train_loss={loss:.4f} train_acc={acc:.4f}")



## === cell 8
sample_sub = pd.read_csv(SAMPLE_SUB)
if "Id" not in sample_sub.columns:
    raise ValueError("sample_submission must contain Id column")

id_list = sample_sub["Id"].astype(str).to_numpy()

test_id_to_path = dict(
    zip(df_test["image_id"].astype(str).to_numpy(), df_test["image_path"].to_numpy())
)
test_paths = np.fromiter(
    (test_id_to_path.get(i, None) for i in id_list), dtype=object, count=len(id_list)
)
missing = int(pd.isna(test_paths).sum())
print("ids:", len(id_list), "missing paths:", missing)

test_df_ordered = pd.DataFrame({"image_id": id_list, "image_path": test_paths})

test_ds = IWildcamDataset(
    test_df_ordered, test_tfms, is_train=False, cache_resized_uint8=False
)

_infer_bs = 384 if device.type == "cuda" else 64

test_loader = DataLoader(
    test_ds,
    batch_size=_infer_bs,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=_prefetch,
)



## === cell 9
net1.eval()
vit.eval()

pred_categories = np.empty(len(test_ds), dtype=np.int64)

most_common_category = int(df_train["category_id"].value_counts().idxmax())

label2cat_arr = np.fromiter(
    (int(label2category[i]) for i in range(num_classes)),
    dtype=np.int64,
    count=num_classes,
)

offset = 0
loader_it = (
    CUDAPrefetcher(test_loader, device, channels_last=True)
    if device.type == "cuda"
    else test_loader
)

label2cat_t = torch.as_tensor(label2cat_arr, device=device, dtype=torch.int64)

with torch.inference_mode():
    for xb, img_ids in loader_it:
        bsz = xb.size(0)
        if device.type != "cuda":
            xb = xb.to(device)

        out1 = net1(xb)
        out3 = vit(xb)
        logits = (2.5 * out1 + out3) / 3.5
        labels_t = logits.argmax(1).to(torch.int64)

        cats_t = label2cat_t[labels_t.clamp_(0, num_classes - 1)]
        pred_categories[offset : offset + bsz] = (
            cats_t.detach().cpu().numpy().astype(np.int64)
        )
        offset += bsz

if missing > 0:
    missing_mask = pd.isna(test_paths)
    pred_categories[missing_mask] = most_common_category

print("pred example:", list(zip(id_list[:3].tolist(), pred_categories[:3].tolist())))



## === cell 10
os.makedirs("./results", exist_ok=True)

sub_df = pd.DataFrame({"Id": id_list, "Category": pred_categories.astype(int)})
out_path = "./results/submission.csv"
sub_df.to_csv(out_path, index=False)
print("wrote:", out_path, "shape:", sub_df.shape)
print(sub_df.head())



## === cell 11
if is_gpu_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
gc.collect()
print("done")
