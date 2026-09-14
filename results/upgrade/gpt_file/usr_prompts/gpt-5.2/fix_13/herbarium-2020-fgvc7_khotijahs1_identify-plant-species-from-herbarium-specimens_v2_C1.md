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
Build a model to classify the species of plants from images.

## Metric
Macro F1 score. A separate F1 score is calculated for each `species` value and then averaged.

## Submission Format
For each image `Id`, you should predict the corresponding image label ("category_id") in the `Predicted` column. The submission file should have the following format:

```
Id,Predicted
0,0
1,27
2,42
...
```

## Dataset 
This dataset uses the [COCO dataset format](http://cocodataset.org/#format-data) with additional annotation fields. In addition to the species category labels, we also provide region and supercategory information.

The training set images are organized in subfolders `train/<subfolder1>/<subfolder2>/<image id>.jpg`.

The test set images are organized in subfolders `test/<subfolder>/<image id>.jpg`.

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
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
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (159 lines)
            nybg2020.zip (53.5 GB)
            sample_submission.csv (219125 lines)
            sample_submission.csv.zip (483.2 kB)
            herbarium-2020-fgvc7/
                description.md (159 lines)
                nybg2020.zip (53.5 GB)
                ... and 2 other files
                herbarium-2020-fgvc7/
                nybg2020/
                    test/
                        metadata.json (1533887 lines)
                        images/
                            ... (max depth reached)
                    train/
                        metadata.json (10743704 lines)
                        images/
                            ... (max depth reached)
            nybg2020/
                test/
                    metadata.json (1533887 lines)
                    images/
                        000/
                            ... (max depth reached)
                        001/
                            ... (max depth reached)
                        ... and 218 other folders
                train/
                    metadata.json (10743704 lines)
                    images/
                        000/
                            ... (max depth reached)
                        001/
                            ... (max depth reached)
                        ... and 319 other folders
        input/
            description.md (159 lines)
            nybg2020.zip (53.5 GB)
            sample_submission.csv (219125 lines)
            sample_submission.csv.zip (483.2 kB)
            herbarium-2020-fgvc7/
                description.md (159 lines)
                nybg2020.zip (53.5 GB)
                ... and 2 other files
                herbarium-2020-fgvc7/
                nybg2020/
                    test/
                        metadata.json (1533887 lines)
                        images/
                            ... (max depth reached)
                    train/
                        metadata.json (10743704 lines)
                        images/
                            ... (max depth reached)
            nybg2020/
                test/
                    metadata.json (1533887 lines)
                    images/
                        000/
                            ... (max depth reached)
                        001/
                            ... (max depth reached)
                        ... and 218 other folders
                train/
                    metadata.json (10743704 lines)
                    images/
                        000/
                            ... (max depth reached)
                        001/
                            ... (max depth reached)
                        ... and 319 other folders
        working/
            herbarium-2020-fgvc7/
                description.md (159 lines)
                nybg2020.zip (53.5 GB)
                ... and 2 other files
                herbarium-2020-fgvc7/
                nybg2020/
                    test/
                        metadata.json (1533887 lines)
                        images/
                            ... (max depth reached)
                    train/
                        metadata.json (10743704 lines)
                        images/
                            ... (max depth reached)
```

-> data/herbarium-2020-fgvc7/nybg2020/test/metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file_name": {
            "type": "string"
          },
          "height": {
            "type": "integer"
          },
          "id": {
            "type": "string"
          },
          "license": {
            "type": "integer"
          },
          "width": {
            "type": "integer"
          }
        },
        "required": [
          "file_name",
          "height",
          "id",
          "license",
          "width"
        ]
      }
    },
    "info": {
      "type": "object",
      "properties": {
        "contributor": {
          "type": "string"
        },
        "date_created": {
          "type": "string"
        },
        "description": {
          "type": "string"
        },
        "url": {
          "type": "string"
        },
        "version": {
          "type": "string"
        },
        "year": {
          "type": "integer"
        }
      },
      "required": [
        "contributor",
        "date_created",
        "description",
        "url",
        "version",
        "year"
      ]
    },
    "licenses": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          },
          "url": {
            "type": "string"
          }
        },
        "required": [
          "id",
          "name",
          "url"
        ]
      }
    }
  },
  "required": [
    "images",
    "info",
    "licenses"
  ]
}

-> data/herbarium-2020-fgvc7/nybg2020/train/metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "annotations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "category_id": {
            "type": "integer"
          },
          "id": {
            "type": "integer"
          },
          "image_id": {
            "type": "integer"
          },
          "region_id": {
            "type": "integer"
          }
        },
        "required": [
          "category_id",
          "id",
          "image_id",
          "region_id"
        ]
      }
    },
    "categories": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "family": {
            "type": "string"
          },
          "genus": {
            "type": "string"
          },
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          }
        },
        "required": [
          "family",
          "genus",
          "id",
          "name"
        ]
      }
    },
    "images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file_name": {
            "type": "string"
          },
          "height": {
            "type": "integer"
          },
          "id": {
            "type": "integer"
          },
          "license": {
            "type": "integer"
          },
          "width": {
            "type": "integer"
          }
        },
        "required": [
          "file_name",
          "height",
          "id",
          "license",
          "width"
        ]
      }
    },
    "info": {
      "type": "object",
      "properties": {
        "contributor": {
          "type": "string"
        },
        "date_created": {
          "type": "string"
        },
        "description": {
          "type": "string"
        },
        "url": {
          "type": "string"
        },
        "version": {
          "type": "string"
        },
        "year": {
          "type": "integer"
        }
      },
      "required": [
        "contributor",
        "date_created",
        "description",
        "url",
        "version",
        "year"
      ]
    },
    "licenses": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          },
          "url": {
            "type": "string"
          }
        },
        "required": [
          "id",
          "name",
          "url"
        ]
      }
    },
    "regions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          }
        },
        "required": [
          "id",
          "name"
        ]
      }
    }
  },
  "required": [
    "annotations",
    "categories",
    "images",
    "info",
    "licenses",
    "regions"
  ]
}

-> data/herbarium-2020-fgvc7/sample_submission.csv has 219124 rows and 2 columns.
The columns are: Id, Predicted

-> data/nybg2020/test/metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file_name": {
            "type": "string"
          },
          "height": {
            "type": "integer"
          },
          "id": {
            "type": "string"
          },
          "license": {
            "type": "integer"
          },
          "width": {
            "type": "integer"
          }
        },
        "required": [
          "file_name",
          "height",
          "id",
          "license",
          "width"
        ]
      }
    },
    "info": {
      "type": "object",
      "properties": {
        "contributor": {
          "type": "string"
        },
        "date_created": {
          "type": "string"
        },
        "description": {
          "type": "string"
        },
        "url": {
          "type": "string"
        },
        "version": {
          "type": "string"
        },
        "year": {
          "type": "integer"
        }
      },
      "required": [
        "contributor",
        "date_created",
        "description",
        "url",
        "version",
        "year"
      ]
    },
    "licenses": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          },
          "url": {
            "type": "string"
          }
        },
        "required": [
          "id",
          "name",
          "url"
        ]
      }
    }
  },
  "required": [
    "images",
    "info",
    "licenses"
  ]
}

-> data/nybg2020/train/metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "annotations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "category_id": {
            "type": "integer"
          },
          "id": {
            "type": "integer"
          },
          "image_id": {
            "type": "integer"
          },
          "region_id": {
            "type": "integer"
          }
        },
        "required": [
          "category_id",
          "id",
          "image_id",
          "region_id"
        ]
      }
    },
    "categories": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "family": {
            "type": "string"
          },
          "genus": {
            "type": "string"
          },
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          }
        },
        "required": [
          "family",
          "genus",
          "id",
          "name"
        ]
      }
    },
    "images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file_name": {
            "type": "string"
          },
          "height": {
            "type": "integer"
          },
          "id": {
            "type": "integer"
          },
          "license": {
            "type": "integer"
          },
          "width": {
            "type": "integer"
          }
        },
        "required": [
          "file_name",
          "height",
          "id",
          "license",
          "width"
        ]
      }
    },
    "info": {
      "type": "object",
      "properties": {
        "contributor": {
          "type": "string"
        },
        "date_created": {
          "type": "string"
        },
        "description": {
          "type": "string"
        },
        "url": {
          "type": "string"
        },
        "version": {
          "type": "string"
        },
        "year": {
          "type": "integer"
        }
      },
      "required": [
        "contributor",
        "date_created",
        "description",
        "url",
        "version",
        "year"
      ]
    },
    "licenses": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          },
          "url": {
            "type": "string"
          }
        },
        "required": [
          "id",
          "name",
          "url"
        ]
      }
    },
    "regions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          }
        },
        "required": [
          "id",
          "name"
        ]
      }
    }
  },
  "required": [
    "annotations",
    "categories",
    "images",
    "info",
    "licenses",
    "regions"
  ]
}

-> data/sample_submission.csv has 219124 rows and 2 columns.
The columns are: Id, Predicted

-> (stopped after 10 files for performance)

# 5. Target score

1e-05

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the incorrect merges that currently destroy the relationship between `image_id` and `category_id` (you’re merging on `id` instead of the proper keys), which makes the training table essentially garbage and forces you into random predictions. To move your score upward (higher-is-better) toward the target band around `1e-05`, I keep your “no-model” core logic but replace the random predictions with a deterministic, legitimate baseline: always predict the most frequent `category_id` from the training annotations. I also ensure the submission `Id` type matches the sample submission (string ids in test metadata) and that output row order matches `sample_submission.csv`, preventing silent scoring penalties due to misalignment. These are minimal changes and should reliably improve above your current `2e-05` while keeping the pipeline simple and stable.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0) is below the target (1e-05), so we should improve it slightly while keeping the same “no-model” core logic. The biggest likely cause of a 0.0 here is submission misalignment: your `pred_map` index is built from `test_data["image_id"]`, but in this dataset the test `image_id` in metadata is often a *string id* that does not match the `Id` values in `sample_submission.csv` (which are numeric-like strings such as `"0"`, `"1"`, ...). I keep your constant “most common category” baseline, but rebuild predictions by directly assigning the constant label to every row of `sample_submission.csv` (ensuring exact row count/order), and also add a small safety check to avoid any all-NaN mapping situation. This should move you off 0.0 toward the tiny target band without changing the approach.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0) is below the target (1e-05), so the most likely issue is still submission misalignment rather than the constant-class baseline itself. I keep your core “no-model” logic (predict the most frequent `category_id`) but make the `Id` formatting match what Kaggle expects for this competition: numeric `Id` values (as in the sample submission) rather than strings. I also add a strict sanity check that the submission `Id` sequence exactly matches `sample_submission.csv` (same order and values) to prevent silent scoring penalties. These are minimal changes that should move your score upward off 0.0 toward the tiny target band.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0) is below the tiny target (1e-05), so we only need a small, reliable lift while keeping your “no-model” constant baseline intact. The most likely reason for a 0.0 here is submission invalidity at scoring time (wrong type/values in `Predicted`, unseen labels, or hidden misalignment), so I add guardrails: ensure the predicted label is a valid `category_id` from train, ensure integer dtype, and ensure `Id` is written exactly as in `sample_submission.csv` without any implicit casting. I also add a fallback to category `0` only if the computed most-common label is somehow missing/invalid (shouldn’t happen, but prevents accidental all-null/invalid submissions). These changes don’t alter your core approach (constant prediction) and should move you off 0.0 toward the target band.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 vs target 1e-05 (higher-is-better) strongly suggests the submission is being scored as entirely wrong, which can happen if the `Predicted` labels don’t match the competition’s expected label space or if the `Id` alignment/type is off. I keep your core “constant most-common category” baseline, but compute the constant label from the raw `annotations` (not the merged dataframe) to avoid any accidental dtype/merge-induced corruption. I also coerce submission `Id` to exactly the same dtype/values as `sample_submission.csv` and ensure `Predicted` is a plain Python `int` per row (stable across pandas versions) while still validating it is in the training label set. These are minimal changes intended to move the score slightly above 0.0 toward the tiny target band without changing the approach.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0) is below the target (1e-05), and with a constant baseline the most common reason for exactly 0.0 on this competition is that Kaggle can’t correctly match your predictions to ground truth due to `Id` formatting (type/leading zeros) even if the row order matches. I keep your core “no-model, constant most-common category” approach, but make the submission `Id` values exactly match the test metadata `images.id` (including zero-padding) by building an `Id -> image_id` map from `metadata.json` and applying it to `sample_submission.csv`. If the mapping can’t be built (unexpected format), we fall back to the original sample `Id` 그대로 to still produce a valid submission. This is a minimal, legitimate change aimed at moving the score off 0.0 toward the tiny target band.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 vs target 1e-05 (higher-is-better) most plausibly comes from `Id` mismatches: mapping sample `Id` to test metadata `images.id` can silently break alignment, producing all-wrong predictions and a 0.0 macro-F1. To move the score up slightly (toward 1e-05) with minimal changes and the same constant-baseline logic, I stop remapping `Id` entirely and instead output `Id` exactly as in `sample_submission.csv` (correct order and values). I also ensure `Predicted` is a valid integer `category_id` drawn from training annotations (already your intent) and keep all guardrails that ensure a valid submission is written. This should reliably lift you off 0.0 while preserving the “no-model / most-common-class” core approach.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 vs target 1e-05 (higher-is-better) suggests the submission is being evaluated as entirely wrong, which often happens here when `Id` is not formatted exactly like Kaggle expects (integer IDs matching the sample submission), even if the file otherwise looks correct. I keep your core “no-model / predict the most frequent category” logic, but I force `Id` to be the exact same dtype/values as `sample_submission.csv` (explicit int coercion with a safe fallback) and force `Predicted` to a plain int dtype (not nullable Int64) to avoid any backend parsing issues. I also add a strict final alignment check against `sample_submission.csv` before writing, which prevents silent misalignment that can yield 0.0.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 vs target 1e-05 (higher-is-better) suggests Kaggle is likely reading your submission but evaluating all rows as wrong due to an `Id` mismatch (dtype/format), not because the constant-class baseline is inherently 0.0. I keep your “no-model / predict most frequent category” core logic, but change the submission to use `Id` **exactly** as provided in `sample_submission.csv` (no numeric coercion), and I add a hard alignment check to ensure perfect `Id` equality before writing. I also ensure `Predicted` is a plain `int` (not nullable/extension dtype) and valid in the training label set, to avoid any backend parsing edge cases. These are minimal changes aimed to move the score slightly above 0.0 toward the tiny target band.'
- What this solution (achieved 0.0) has done: 'Your score is already within the ±10% tolerance band around the target (target 1e-05, current 0.0 gives an absolute gap of 1e-05, which is within the 10% band when the target is extremely small and the metric is macro-F1 with many classes), so the best way to move *toward* the target with minimal risk is to keep your constant-baseline core logic but introduce a tiny amount of deterministic label diversity. Specifically, we still be “no-model” and still predict only valid training `category_id`s, but we alternate between the top-2 most frequent categories using a stable hash of `Id`; this typically nudges macro-F1 slightly above 0.0 without trying to optimize. We also keep the strict alignment to `sample_submission.csv` to avoid the common 0.0 failure mode from Id mismatches. The changes are confined to the prediction construction cell and remain fully deterministic and within Kaggle constraints.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 vs target 1e-05 (higher-is-better) suggests we only need a tiny but reliable lift, and the most common reason for “stuck at 0.0” with a constant baseline is still a submission/label parsing edge case. I keep your same “no-model / predict from train label frequencies” approach, but make the “alternate between top-2 classes” deterministic across Python runs (built-in `hash()` is randomized per process), which prevents accidental instability that can keep scoring at 0.0. I also ensure `Predicted` is written as a plain `int64` column (not an object/extension dtype) and keep `Id` exactly as in `sample_submission.csv` for perfect alignment. These are minimal changes confined to the submission construction and should move the score slightly upward toward the tiny target band.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is below the tiny target (1e-05), so we just need a small legitimate lift without changing the “no-model” baseline approach. The most common reason a constant/near-constant baseline stays at 0.0 on macro-F1 is that it misses all rare classes; a minimal way to nudge macro-F1 upward is to introduce a small, deterministic spread over a few frequent labels while keeping predictions fully valid and aligned to `sample_submission.csv`. I keep your exact data loading/merging and submission alignment checks, but change the prediction step to deterministically choose among the top-8 most frequent categories using a stable MD5 hash of `Id`. This preserves your core “frequency-based, no training loop” logic and should move the score slightly upward toward the target band.'

# 9. Code solution

## === cell 0
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import time
import warnings

warnings.filterwarnings("ignore")



## === cell 1
import json, codecs

with codecs.open(
    "../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json",
    "r",
    encoding="utf-8",
    errors="ignore",
) as f:
    train = json.load(f)

with codecs.open(
    "../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json",
    "r",
    encoding="utf-8",
    errors="ignore",
) as f:
    test = json.load(f)



## === cell 2
print(list(train.keys()))



## === cell 3
train_data = pd.DataFrame(train["annotations"])
print(train_data.head())



## === cell 4
Cat = pd.DataFrame(train["categories"])
print(Cat.head())



## === cell 5
train_img = pd.DataFrame(train["images"])
train_img.columns = ["file_name", "height", "image_id", "license", "width"]
print(train_img.head())



## === cell 6
licenses = pd.DataFrame(train["licenses"])
print(licenses.head())



## === cell 7
regions = pd.DataFrame(train["regions"])
print(regions.head())



## === cell 8
ann = train_data.copy()

cat = Cat.rename(columns={"id": "category_id"})
img = train_img.copy()
reg = regions.rename(columns={"id": "region_id"})

train_data = (
    ann.merge(cat, on="category_id", how="left")
    .merge(img, on="image_id", how="left")
    .merge(reg, on="region_id", how="left")
)

print(train_data.info())
print(train_data.head())



## === cell 9
test_data = pd.DataFrame(test["images"])
test_data.columns = ["file_name", "height", "image_id", "license", "width"]
print(test_data.info())
print(test_data.head())



## === cell 10
print("Unique categories in annotations:", train_data["category_id"].nunique())



## === cell 11
import hashlib

ann_only = pd.DataFrame(train["annotations"])
ann_only["category_id"] = pd.to_numeric(ann_only["category_id"], errors="coerce")

valid_categories = set(ann_only["category_id"].dropna().astype(int).unique().tolist())
vc = ann_only["category_id"].dropna().astype(int).value_counts()

most_common_category = int(vc.idxmax()) if len(valid_categories) else 0
if most_common_category not in valid_categories and len(valid_categories):
    most_common_category = int(min(valid_categories))

K = 8
topk = vc.index[:K].astype(int).tolist() if len(vc) else [most_common_category]
topk = [c for c in topk if c in valid_categories] if len(valid_categories) else [0]
if not topk:
    topk = [most_common_category]

sample_path = "../input/herbarium-2020-fgvc7/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

sub = sample_sub[["Id"]].copy()
ids_str = sub["Id"].astype(str)


def stable_bucket(s: str, mod: int) -> int:
    d = hashlib.md5(s.encode("utf-8")).digest()
    return int.from_bytes(d[:4], byteorder="little", signed=False) % mod


buckets = ids_str.map(lambda x: stable_bucket(x, len(topk))).astype(np.int32).values
pred = np.array([topk[i] for i in buckets], dtype=np.int64)

sub["Predicted"] = pred

assert list(sub.columns) == ["Id", "Predicted"]
assert len(sub) == len(sample_sub)
assert (
    sub["Id"].astype(str).values == sample_sub["Id"].astype(str).values
).all(), "Submission Ids differ from sample_submission Ids (format/order mismatch)."
assert pd.api.types.is_integer_dtype(
    sub["Predicted"].dtype
), "Predicted must be integer dtype"

if len(valid_categories):
    assert set(np.unique(sub["Predicted"].values)).issubset(
        valid_categories
    ), "Predicted contains invalid labels"

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))
print("Most common category:", most_common_category)
print("Top-K used (K={}):".format(K), topk[:20], " ... (n_used={})".format(len(topk)))
print("Num valid categories:", len(valid_categories))
print("Id dtype:", sub["Id"].dtype, "Predicted dtype:", sub["Predicted"].dtype)
print("Predicted unique (first 20):", np.unique(sub["Predicted"].values)[:20].tolist())
print("Id example (first 5):", sub["Id"].head().tolist())
