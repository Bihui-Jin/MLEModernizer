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

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

# 5. Target score

0.001

# 6. Current score

0.0001

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.37038) has done: 'I make the smallest possible change to ensure your notebook actually generates a valid submission `.csv` in the required schema (`Id,Category`) so you can obtain a Kaggle score (right now you have “Not yielded” because no submission file is produced). Since your target score is extremely low (0.001 accuracy), I deliberately output a very weak but valid baseline by predicting the same class for every test image; this reliably stay near the low target without adding any modeling logic. I also keep paths within `/kaggle/input` and write the submission to `/kaggle/working/submission.csv` as Kaggle expects. This preserves your current “core logic” (no model yet) and only adds the missing submission-generation step.'
- What this solution (achieved 0.0) has done: 'Your current score (0.37038) is far above the target (0.001) and higher is better, so we should deliberately reduce accuracy to move closer to the target. The smallest reliable way (without changing the “core logic”, since there is no actual model) is to output a deterministic but “wrong-looking” constant label that’s unlikely to match the true distribution, instead of predicting all zeros (which often matches many empty images). I keep the same submission generation pipeline and schema, but change the constant prediction from `0` to a fixed non-zero class id. The script still run end-to-end and write `/kaggle/working/submission.csv` with `Id,Category`.'
- What this solution (achieved 0.37038) has done: 'Your current score (0.0) is below the target (0.001), and higher is better, so we need a tiny increase in accuracy while keeping your “intentionally weak constant prediction” core logic unchanged. The simplest way is to still predict a single constant class for all rows, but choose a constant that is more likely to appear than an out-of-range label like `999999` (which always be wrong). Predicting `0` (empty) is typically the most common label in iWildCam, so it should lift accuracy slightly while remaining low. I also keep the submission schema exactly as in `sample_submission.csv` (`Id,Category`) and write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.02039) has done: 'Your current score (0.37038) is far above the target (0.001) with a higher-is-better metric, so we should intentionally reduce accuracy while still producing a valid submission. The smallest, most reliable way is to keep the exact “constant prediction for all test rows” core logic, but switch the constant from `0` (often very common in iWildCam due to empty images) to a fixed non-zero class to make predictions wrong more often. To keep the file valid, we also ensure the constant is a legitimate category id taken from `iwildcam2020_test_information.json` if available (fallback to `1`). This should move the score downward toward the low target without changing any modeling/training logic.'
- What this solution (achieved 0.0) has done: 'Your current accuracy (0.02039) is still far above the very low target (0.001), so we should intentionally reduce accuracy with the smallest possible change while keeping the “constant prediction for all rows” core logic intact. The most reliable way is to pick a constant label that is valid but unlikely to occur in the test set; instead of choosing the smallest non-zero category, we choose the largest non-zero category id from the provided categories list. This preserves the same submission pipeline and schema (`Id,Category`) and still writes `/kaggle/working/submission.csv`. If the categories list can’t be read, we fall back to `1` as before to avoid breaking execution.'
- What this solution (achieved 0.37038) has done: 'To move your accuracy upward slightly from 0.0 toward the very low target 0.001 (without changing the “constant prediction for all rows” core logic), the smallest reliable tweak is to stop guessing a rare non-zero class and instead guess the majority class `0` (empty), which is typically common in iWildCam and should yield a small non-zero accuracy. I keep the same I/O, schema (`Id,Category`), and deterministic constant prediction approach; only the chosen constant change. The script still run end-to-end and write `/kaggle/working/submission.csv` exactly as required.'
- What this solution (achieved 0.0) has done: 'Your current accuracy (0.37038) is far above the target (0.001) and higher-is-better, so we should deliberately reduce accuracy to move closer to the target band (0.0009–0.0011). Keeping the exact same “constant prediction for all rows” core logic, the smallest effective change is to stop predicting `0` (often the majority “empty” class) and instead predict a fixed non-zero class that is less likely to be correct. To keep the submission valid, we pick a deterministic non-zero category id from the provided categories list (fallback to `1` if anything is missing). This should substantially lower accuracy toward the very low target while still producing a valid `/kaggle/working/submission.csv` with `Id,Category`.'
- What this solution (achieved 0.37038) has done: 'To move your score upward slightly from 0.0 toward the very low target (0.001) while keeping the exact same “constant prediction for all rows” core logic, I switch the constant prediction to `0`, which is typically the most frequent class (empty) and is the smallest, most reliable way to get a non-zero accuracy. I also stop deriving the constant from `iwildcam2020_test_information.json` categories (those categories are not the label ids for this task and can push you back to near-zero). The submission schema and path remain exactly `Id,Category` written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current accuracy (0.37038) is far above the target (0.001) and higher-is-better, so we should deliberately reduce accuracy to move closer to the target band (0.0009–0.0011). Keeping the exact same “constant prediction for all rows” core logic, the smallest effective change is to stop predicting `0` (often the majority “empty” class) and instead predict a fixed non-zero class that’s valid but typically much less frequent. To ensure the label is valid without changing the approach, we read the training annotations and pick the least frequent category id (a deterministic rare class), then predict it for every test image. The script still runs end-to-end and writes a valid `/kaggle/working/submission.csv` with `Id,Category`.'
- What this solution (achieved 0.37038) has done: 'To move accuracy up slightly from 0.0 toward the low target 0.001 (higher-is-better) with the smallest possible change, we keep the exact same “predict one constant label for every test image” core logic but pick a constant that is guaranteed to be a valid training label and is more likely to appear than an ultra-rare class. Specifically, instead of selecting the least frequent category (which can easily yield 0 correct), we select the most frequent category_id from the training annotations (a deterministic majority-class baseline). This should raise the score from 0.0 into a small-but-nonzero range, moving closer to the target while still producing a valid `submission.csv` with `Id,Category`.'
- What this solution (achieved 0.0001) has done: 'Your current accuracy (0.37038) is far above the very low target (0.001), so we should deliberately reduce accuracy while keeping the same “predict one constant label for every test image” core logic. The most controlled way is to choose a constant label that is valid but extremely unlikely to be correct: we pick the rarest `category_id` in the training annotations (deterministically breaking ties by choosing the smallest id). This keeps the exact same submission pipeline and schema (`Id,Category`) and only changes how the constant is selected, which should move the score downward toward the target. We also keep a safe fallback to `0` if anything goes wrong so a valid submission is always produced.'
- What this solution (achieved 0.0001) has done: 'To move your score upward from 0.0001 toward the target 0.001 (higher-is-better) while keeping the exact same “predict one constant label for every test image” core logic, I change only how the constant label is chosen. Instead of selecting the rarest class from training (which often yields ~0 correct), we select a low-but-not-minimum frequency class (a small quantile by training frequency), which should increase accuracy slightly but stay very low. I keep the same I/O paths, submission schema (`Id,Category`), and still write `/kaggle/working/submission.csv`. A safe fallback to `0` remains to guarantee a valid submission.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
SAMPLE_SUB_PATH = "/kaggle/input/iwildcam-2020-fgvc7/sample_submission.csv"
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

required_cols = ["Id", "Category"]
missing = [c for c in required_cols if c not in sample_sub.columns]
if missing:
    raise ValueError(
        f"sample_submission missing columns {missing}. Found columns: {list(sample_sub.columns)}"
    )

submission = sample_sub.copy()

TRAIN_ANN_PATH = "/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json"
if not os.path.exists(TRAIN_ANN_PATH):
    TRAIN_ANN_PATH = "/kaggle/input/iwildcam2020_train_annotations.json"

constant_label = (
    0  # fallback; will be overwritten by a valid training category if possible
)

try:
    with open(TRAIN_ANN_PATH, "r") as f:
        train_ann = json.load(f)

    ann = pd.DataFrame(train_ann.get("annotations", []))
    if "category_id" in ann.columns and len(ann) > 0:
        vc = ann["category_id"].value_counts()

        q = 0.02  # small quantile -> keeps accuracy very low but typically > rarest
        target_count = float(vc.quantile(q))
        counts = vc.sort_values(ascending=True)
        diffs = (counts.astype(float) - target_count).abs()
        best_count = counts.loc[diffs.idxmin()]
        candidate_ids = sorted(counts[counts == best_count].index.tolist())
        constant_label = int(candidate_ids[0])
    else:
        print(
            "Warning: training annotations missing/empty; using fallback constant_label=0"
        )
except Exception as e:
    print(
        "Warning: could not derive low-frequency class from train annotations, using fallback.",
        repr(e),
    )

submission["Category"] = int(constant_label)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Using constant label:", constant_label)
print("Head:\n", submission.head())
print("Shape:", submission.shape)
print("Category value counts (top):\n", submission["Category"].value_counts().head())
