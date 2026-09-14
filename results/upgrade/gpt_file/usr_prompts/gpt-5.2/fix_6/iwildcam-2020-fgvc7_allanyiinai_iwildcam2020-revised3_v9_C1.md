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
import os
import json
import math
import gc
import random
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile
from tqdm.auto import tqdm

import torchvision
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True  # fixed input size => faster kernels
torch.backends.cuda.matmul.allow_tf32 = (
    False  # avoid precision changes beyond negligible FP diffs
)

try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

BASE_INPUT = "/kaggle/input/iwildcam-2020-fgvc7"
TRAIN_JSON = os.path.join(BASE_INPUT, "iwildcam2020_train_annotations.json")
TEST_JSON = os.path.join(BASE_INPUT, "iwildcam2020_test_information.json")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_DIR = os.path.join(BASE_INPUT, "test")

assert os.path.exists(TEST_JSON), f"Missing {TEST_JSON}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"




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
    return False


def is_in_kaggle_kernal():
    return os.path.exists("/kaggle")


print(
    "ipython:",
    is_in_ipython(),
    "colab:",
    is_in_colab(),
    "kaggle:",
    is_in_kaggle_kernal(),
)




## === cell 2
print("Using torchvision:", torchvision.__version__, "torch:", torch.__version__)




## === cell 3
with open(TRAIN_JSON, "r") as f:
    train_data = json.load(f)

with open(TEST_JSON, "r") as f:
    test_data = json.load(f)

train_images_df = pd.DataFrame(train_data["images"])[
    ["id", "file_name", "location"]
].rename(columns={"id": "image_id"})
train_ann_df = pd.DataFrame(train_data["annotations"])[
    ["id", "image_id", "category_id", "count"]
]

df_train = train_ann_df.merge(train_images_df, on="image_id", how="left")
df_train["image_path"] = (BASE_INPUT + "/train/") + df_train["file_name"].astype(str)

df_test = pd.DataFrame(test_data["images"])[["id", "file_name", "location"]].rename(
    columns={"id": "image_id"}
)
df_test["image_path"] = (TEST_DIR + "/") + df_test["file_name"].astype(str)

print("df_train:", df_train.shape, "df_test:", df_test.shape)
try:
    display(df_train.head(2))
    display(df_test.head(2))
except NameError:
    print(df_train.head(2))
    print(df_test.head(2))




## === cell 4
df_category_train = pd.DataFrame(train_data["categories"])[
    ["id", "name", "count"]
].copy()
df_category_test = pd.DataFrame(test_data["categories"])[["id", "name", "count"]].copy()

df_category_train = df_category_train.sort_values(["count"], ascending=False)
df_category_test = df_category_test.sort_values(["count"], ascending=False)

animal_category_lists = sorted(set(df_train["category_id"].astype(int).tolist()))
df_category_train = df_category_train[
    df_category_train["id"].isin(animal_category_lists)
]
df_category_test = df_category_test[df_category_test["count"] > 0]

animal_category_lists_train = df_category_train["id"].astype(int).tolist()
animal_category_lists_test = df_category_test["id"].astype(int).tolist()
category_missing_list = [
    cid for cid in animal_category_lists_test if cid not in animal_category_lists_train
]

print("train categories in annotations:", len(animal_category_lists_train))
print("test categories w/ count>0:", len(animal_category_lists_test))
print(
    "missing in train (from test list):",
    category_missing_list[:20],
    "n_missing:",
    len(category_missing_list),
)




## === cell 5
label2category = OrderedDict()
category2label = OrderedDict()
for i, cid in enumerate(animal_category_lists_train):
    category2label[int(cid)] = i
    label2category[i] = int(cid)

label_idxes = [category2label[int(x)] for x in df_train["category_id"].tolist()]
image_pathes = df_train["image_path"].tolist()

print("label_idxes:", label_idxes[:5])
print("image_pathes:", image_pathes[:2])
print("n_classes:", len(animal_category_lists_train))




## === cell 6
rare_animals = OrderedDict()
common_animals = OrderedDict()
df_animal_frequency = df_train["category_id"].value_counts()

df_rare_animal_frequency = df_animal_frequency[df_animal_frequency <= 10]
df_common_animal_frequency = df_animal_frequency[df_animal_frequency > 1000]

for k, v in df_rare_animal_frequency.items():
    rare_animals[int(k)] = int(v)

for k, v in df_common_animal_frequency.items():
    common_animals[int(k)] = int(v)

print("rare_animals:", len(rare_animals), "common_animals:", len(common_animals))




## === cell 7
model = torchvision.models.efficientnet_b0(
    weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
)
model.eval()
model.to(DEVICE)

if DEVICE.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

if DEVICE.type == "cuda":
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("torch.compile: enabled")
    except Exception as e:
        print("torch.compile: disabled (fallback). Reason:", repr(e))

weights_meta = torchvision.models.EfficientNet_B0_Weights.DEFAULT
preprocess = weights_meta.transforms()

IMAGENET_ANIMAL_RANGES = [
    (0, 397),
    (398, 574),
    (575, 606),
]
animal_indices = []
for a, b in IMAGENET_ANIMAL_RANGES:
    animal_indices.extend(list(range(a, b + 1)))
animal_indices = torch.tensor(animal_indices, device=DEVICE, dtype=torch.long)

most_common_category = int(df_train["category_id"].value_counts().index[0])
print("most_common_category:", most_common_category)




## === cell 8
class TestImageDataset(Dataset):
    def __init__(self, df_test, preprocess):
        self.df = df_test.reset_index(drop=True)
        self.preprocess = preprocess
        self._image_id = self.df["image_id"].astype(str).to_numpy()
        self._image_path = self.df["image_path"].astype(str).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self._image_id[idx]
        path = self._image_path[idx]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                x = self.preprocess(im)
        except Exception:
            im = Image.new("RGB", (224, 224), (0, 0, 0))
            x = self.preprocess(im)
        return image_id, x


def safe_collate(batch):
    ids, xs = zip(*batch)
    return list(ids), torch.stack(xs, dim=0)


test_ds = TestImageDataset(df_test, preprocess)

cpu_cnt = os.cpu_count() or 2
_num_workers = min(8, max(2, cpu_cnt // 2))
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    collate_fn=safe_collate,
)




## === cell 9
THRESH = 0.25

n_test = len(test_ds)
all_ids = [None] * n_test
all_preds = np.empty((n_test,), dtype=np.int64)

most_common_tensor = torch.tensor(most_common_category, device=DEVICE, dtype=torch.long)
zero_tensor = torch.tensor(0, device=DEVICE, dtype=torch.long)

start = 0
with torch.inference_mode():
    for ids, x in tqdm(test_loader, total=math.ceil(n_test / test_loader.batch_size)):
        bs = x.shape[0]
        end = start + bs

        if DEVICE.type == "cuda":
            x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            x = x.to(DEVICE)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        animal_prob = probs.index_select(1, animal_indices).max(dim=1).values
        pred = torch.where(animal_prob > THRESH, most_common_tensor, zero_tensor)
        pred = pred.detach().cpu().numpy().astype(np.int64)

        all_ids[start:end] = ids
        all_preds[start:end] = pred
        start = end

pred_by_id = pd.Series(all_preds, index=pd.Index(all_ids, name="Id"))

print("Pred count:", len(pred_by_id), "Expected:", len(df_test))




## === cell 10
sample = pd.read_csv(SAMPLE_SUB)
assert "Id" in sample.columns, "sample_submission missing Id column"

id_col = "Id"
out_pred_col = "Predicted"

sample[out_pred_col] = (
    pred_by_id.reindex(sample[id_col]).fillna(0).astype(np.int64).to_numpy()
)
sample = sample[[id_col, out_pred_col]]

out_path = "/kaggle/working/submission.csv"
sample.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sample.head())
print("Rows:", len(sample), "Unique Ids:", sample["Id"].nunique())
assert out_path.endswith(".csv") and os.path.exists(out_path)
assert len(sample) == 60760, "Submission row count mismatch vs expected 60760"
