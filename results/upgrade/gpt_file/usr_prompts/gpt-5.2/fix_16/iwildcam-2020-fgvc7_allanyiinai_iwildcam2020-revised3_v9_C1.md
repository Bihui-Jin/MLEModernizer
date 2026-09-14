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

# 5. Target score

0.5278425364683941

# 6. Current score

0.37979

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.37038) has done: 'The timeout is dominated by test-time image I/O + CPU preprocessing and by per-batch tensor allocations inside the prediction loop. I speed this up without changing the model or decision rule by (1) using a faster DataLoader configuration (more workers, persistent workers, prefetching, pinned memory) and (2) enabling CUDA inference optimizations (channels_last + cudnn benchmark) that preserve numerical semantics within negligible FP differences. I also remove repeated creation of constant tensors inside the loop and vectorize the final pandas mapping, which are provably equivalent but reduce overhead. All paths, architecture, and the thresholding logic remain identical.'
- What this solution (achieved 0.37038) has done: 'I fix the immediate runtime blocker by making the EfficientNet weights normalization robust to the torchvision version (where `weights_meta.meta` may not contain `mean/std`), falling back to the known ImageNet defaults. Then I ensure the test dataset/loader cell completes so later cells can access `test_ds` and `pred_by_id` without `NameError`. Finally, I keep the model/inference logic unchanged and only adjust the submission writing to match the competition’s required column name (`Category`) while keeping the `.csv` output valid.'
- What this solution (achieved 0.37038) has done: 'Your current bottlenecked rule (“animal present?” via ImageNet animal-prob threshold, then output either 0 or the most common training category) is extremely underpowered for this competition, so to move accuracy from 0.370 toward the 0.528 target we need a minimal but legitimate improvement in the *decision rule* while keeping the same EfficientNet-B0 backbone and single forward-pass inference. I keep the model and preprocessing intact, but replace the hard-coded “most_common_category” with a lightweight calibration that maps ImageNet top-1 predictions into the closest iWildCam category using only the provided train annotations (no extra data), falling back to 0 when animal_prob ≤ THRESH. This preserves the core logic (EfficientNet inference + thresholding) but makes “animal present” images produce a more informative class than a constant, which should increase accuracy substantially toward your target. I also keep the submission writing and all paths unchanged and ensure the output CSV schema matches `Id,Category`.'
- What this solution (achieved 0.36592) has done: 'Your current score gap is large (0.37038 → target 0.52784, higher-is-better), so we need a small but meaningful accuracy lift while keeping the same EfficientNet-B0 forward pass, preprocessing, and “animal present?” thresholding core logic. The most effective minimal change here is to avoid label noise from multiple annotations per image by training a better “most common category” prior: compute it after aggregating counts per image (so each image contributes once, weighted by its `count`) instead of counting per-annotation rows. Then, for the ImageNet→iWildCam name-token mapping, we keep the same mapping approach but make it slightly less strict (accept overlap ≥1 instead of ≥2) to reduce fallback-to-constant on animal images, which should lift accuracy toward your target without changing the model or adding training. Finally, we keep submission formatting identical but add a strict alignment check to ensure every predicted Id is present and mapped correctly.'
- What this solution (achieved 0.37057) has done: 'Your current gap to the target is large (0.36592 → 0.52784, higher-is-better), so the smallest legitimate accuracy lift without changing the EfficientNet model or the “animal present?” thresholding core is to make the ImageNet→iWildCam mapping less brittle and reduce fallback-to-`most_common_category`. I keep the same single forward pass, the same animal-prob computation, and the same THRESH rule, but improve the string-token matching by adding a tiny synonym/normalization layer (e.g., “grey/gray”, “wild dog/african wild dog”, “boar/hog”, plural handling) and allowing a weighted overlap score rather than raw overlap count. I also ensure mapping only targets categories that actually appear in train annotations (so we don’t map to never-seen labels) and keep the prior fallback unchanged. These changes keep the pipeline semantics identical (EfficientNet logits → animal_prob gate → mapped class else 0) while increasing the chance that animal images get a more correct non-constant category, pushing accuracy upward toward your target. The script still runs end-to-end and writes `/kaggle/working/submission.csv` with exactly `Id,Category`.'
- What this solution (achieved 0.37036) has done: 'To move your accuracy upward toward the 0.528 target while preserving the exact EfficientNet inference + animal-prob thresholding core, I’m only strengthening the ImageNet→iWildCam label mapping so fewer “animal present” cases fall back to `most_common_category`. Concretely, I add a tiny phrase/synonym normalization and a special-case phrase boost (e.g., “wild boar”, “african elephant”, “mountain lion”, “wild dog”) without changing the model, preprocessing, threshold, or the gating decision rule. I also keep mappings restricted to train-seen categories, and I don’t alter batch sizes, loops, or output formatting—so the submission semantics remain identical but should be more often correct on animal images. The script still runs end-to-end and writes `/kaggle/working/submission.csv` with columns `Id,Category`.'
- What this solution (achieved 0.37036) has done: 'Your gap to the target is large (0.37036 → 0.52784, higher-is-better), so we need a small but meaningful accuracy lift without changing the EfficientNet inference, preprocessing, threshold gate, or training loop (there is none). The minimal, high-impact fix is to stop using a global “most common category” fallback for animal-present images and instead use a per-location prior computed from train annotations, which is legitimate and uses only provided metadata already loaded (location). This preserves the exact core semantics (EfficientNet → animal_prob gate with the same THRESH) while making predictions more plausible because species are strongly location-dependent in iWildCam. We also keep the existing ImageNet→iWildCam mapping, but when mapping is unavailable or weak, we fall back to the per-location prior instead of a constant.'
- What this solution (achieved 0.37036) has done: 'Your current gap to the target is large (0.37036 → 0.52784, higher-is-better), so we need a meaningful but still “core-logic-preserving” accuracy lift. The most effective minimal change here is to keep the exact same EfficientNet inference + animal_prob gate, but replace the brittle ImageNet-name token mapping with a location-conditioned class prior learned from train annotations and applied only when animal_prob > THRESH. Concretely, we compute a per-location distribution over categories (not just top-1), and at inference we sample the top-K most frequent classes for that location and choose among them using a small global prior (this stays purely metadata-driven and does not change the model/threshold semantics). This reduces the “always predict one class for animal-present” failure mode and should move accuracy upward toward the target while keeping runtime and architecture essentially unchanged; submission writing remains identical.'
- What this solution (achieved 0.37979) has done: 'Your current score (0.37036) is far below the target (0.52784), so we should make the smallest legitimate change that increases accuracy without changing the EfficientNet model, preprocessing, or the animal-probability gate. The biggest accuracy limiter is that for `animal_prob > THRESH` you still often fall back to a location prior that ignores the strong *sequence/frame* correlation in iWildCam; we can improve this by switching the fallback from per-`location` to per-`seq_id` (and then to `location`, then global) using only the provided train/test metadata. This keeps the same inference loop and decision semantics (ImageNet forward pass → animal_prob gate → mapped class else fallback), but makes fallback labels much more likely correct. We also keep the existing ImageNet→iWildCam token mapping unchanged; we only change what happens when mapping is missing/weak.'

# 9. Code solution

## === cell 0
import os
import json
import math
import gc
import random
from collections import OrderedDict, defaultdict

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

torch.backends.cudnn.benchmark = True
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
    ["id", "file_name", "location", "seq_id"]
].rename(columns={"id": "image_id"})
train_ann_df = pd.DataFrame(train_data["annotations"])[
    ["id", "image_id", "category_id", "count"]
]

df_train = train_ann_df.merge(train_images_df, on="image_id", how="left")
df_train["image_path"] = (BASE_INPUT + "/train/") + df_train["file_name"].astype(str)

df_test = pd.DataFrame(test_data["images"])[
    ["id", "file_name", "location", "seq_id"]
].rename(columns={"id": "image_id"})
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

df_img_cat = (
    df_train.groupby(["image_id", "category_id"], as_index=False)["count"]
    .sum()
    .sort_values(["image_id", "count"], ascending=[True, False])
)
df_img_top = df_img_cat.drop_duplicates("image_id", keep="first")
most_common_category = int(df_img_top["category_id"].value_counts().index[0])
print("most_common_category (per-image):", most_common_category)

df_loc_cat = (
    df_train.groupby(["location", "category_id"], as_index=False)["count"]
    .sum()
    .sort_values(["location", "count"], ascending=[True, False])
)

TOPK_LOC = 5
loc_to_topk = {}
for loc, g in df_loc_cat.groupby("location", sort=False):
    g2 = g.sort_values("count", ascending=False).head(TOPK_LOC)
    loc_to_topk[int(loc)] = (
        g2["category_id"].astype(np.int64).to_numpy(),
        g2["count"].astype(np.float32).to_numpy(),
    )
print("location topK priors:", len(loc_to_topk), "locations with priors")

df_loc_top = df_loc_cat.drop_duplicates("location", keep="first").copy()
loc_to_topcat = pd.Series(
    df_loc_top["category_id"].astype(np.int64).to_numpy(),
    index=df_loc_top["location"].astype(np.int64).to_numpy(),
).to_dict()

df_seq_cat = (
    df_train.groupby(["seq_id", "category_id"], as_index=False)["count"]
    .sum()
    .sort_values(["seq_id", "count"], ascending=[True, False])
)
df_seq_top = df_seq_cat.drop_duplicates("seq_id", keep="first").copy()
seq_to_topcat = pd.Series(
    df_seq_top["category_id"].astype(np.int64).to_numpy(),
    index=df_seq_top["seq_id"].astype(str).to_numpy(),
).to_dict()
print("sequence top1 priors:", len(seq_to_topcat), "seq_ids with priors")


def _normalize_name(s: str) -> str:
    s = (s or "").lower()
    for ch in ["_", "-", ",", ";", ":", ".", "(", ")", "[", "]", "{", "}", "'"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


_SYNONYM_MAP = {
    "grey": "gray",
    "kitty": "cat",
    "kitten": "cat",
    "puma": "cougar",
    "cougar": "mountain lion",
    "wilddog": "wild dog",
    "hog": "boar",
    "boars": "boar",
    "hogs": "boar",
    "deers": "deer",
    "buffaloes": "buffalo",
    "ox": "cattle",
    "cows": "cattle",
    "cow": "cattle",
    "canid": "dog",
    "canine": "dog",
    "feline": "cat",
    "leopardess": "leopard",
    "lioness": "lion",
    "wolves": "wolf",
    "foxes": "fox",
    "bears": "bear",
    "racoons": "raccoon",
    "crocodilian": "crocodile",
    "gator": "alligator",
    "hippopotamus": "hippo",
    "porcupines": "porcupine",
    "porcupin": "porcupine",
    "chimpanzee": "chimp",
    "orangutan": "orangutan",
    "zebra": "zebra",
    "rhino": "rhinoceros",
    "rhinoceroses": "rhinoceros",
    "elephants": "elephant",
    "giraffes": "giraffe",
    "kangaroos": "kangaroo",
    "wallabies": "wallaby",
}
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "with",
    "without",
    "in",
    "on",
    "at",
    "to",
    "from",
    "by",
    "for",
    "adult",
    "young",
    "male",
    "female",
    "juvenile",
    "common",
    "american",
    "african",
    "european",
    "asian",
}

_PHRASE_BOOSTS = [
    (("wild", "boar"), 1.5),
    (("mountain", "lion"), 1.5),
    (("african", "elephant"), 1.5),
    (("asian", "elephant"), 1.5),
    (("wild", "dog"), 1.5),
    (("spotted", "hyena"), 1.5),
    (("striped", "hyena"), 1.5),
    (("african", "buffalo"), 1.5),
    (("cape", "buffalo"), 1.5),
    (("white", "rhinoceros"), 1.5),
    (("black", "rhinoceros"), 1.5),
    (("red", "deer"), 1.5),
    (("roe", "deer"), 1.5),
    (("brown", "bear"), 1.5),
    (("polar", "bear"), 1.5),
]


def _tokenize(s: str) -> list[str]:
    s = _normalize_name(s)
    toks = []
    for t in s.split(" "):
        if len(t) < 3:
            continue
        if t in _STOPWORDS:
            continue
        if t.endswith("s") and len(t) > 3:
            t = t[:-1]
        t = _SYNONYM_MAP.get(t, t)
        toks.extend(t.split(" "))
    toks = [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]
    return toks


def _phrase_score(token_set: set[str]) -> float:
    score = 0.0
    for (a, b), w in _PHRASE_BOOSTS:
        if a in token_set and b in token_set:
            score += w
    return score


iwild_id_to_name = {
    int(r["id"]): _normalize_name(r["name"]) for r in train_data.get("categories", [])
}
iwild_ids = sorted(set(animal_category_lists_train))

imagenet_categories = None
try:
    imagenet_categories = weights_meta.meta.get("categories", None)
except Exception:
    imagenet_categories = None

if imagenet_categories is None:
    print(
        "Warning: EfficientNet weights meta categories unavailable; using location prior/constant fallback."
    )
    imagenet_to_iwild = None
else:
    iwild_token_sets = {}
    iwild_token_weights = {}
    iwild_phrase_bonus = {}
    for cid in iwild_ids:
        toks = _tokenize(iwild_id_to_name.get(cid, ""))
        tokset = set(toks)
        iwild_token_sets[cid] = tokset
        weights = {}
        for t in tokset:
            weights[t] = 1.0 + (0.25 if len(t) >= 6 else 0.0)
        iwild_token_weights[cid] = weights
        iwild_phrase_bonus[cid] = _phrase_score(tokset)

    imagenet_to_iwild = np.full(
        (len(imagenet_categories),), most_common_category, dtype=np.int64
    )
    n_matched = 0

    MIN_TOKEN_OVERLAP = 1

    for i, name in enumerate(imagenet_categories):
        itoks = set(_tokenize(name))
        if not itoks:
            continue
        i_phrase_bonus = _phrase_score(itoks)

        best_cid = None
        best_inter = 0
        best_score = -1e18

        for cid in iwild_ids:
            inter_toks = itoks & iwild_token_sets[cid]
            inter = len(inter_toks)
            if inter == 0:
                continue
            score = 0.0
            w = iwild_token_weights[cid]
            for t in inter_toks:
                score += w.get(t, 1.0)

            score = score + 0.5 * i_phrase_bonus + 0.5 * iwild_phrase_bonus[cid]

            if (inter > best_inter) or (inter == best_inter and score > best_score):
                best_inter = inter
                best_score = score
                best_cid = cid

        if best_cid is not None and best_inter >= MIN_TOKEN_OVERLAP:
            imagenet_to_iwild[i] = int(best_cid)
            n_matched += 1

    print(
        f"imagenet_to_iwild matches: {n_matched}/{len(imagenet_categories)} (fallback otherwise)"
    )



## === cell 8
try:
    torchvision.set_image_backend("accimage")
    _backend = torchvision.get_image_backend()
except Exception:
    _backend = torchvision.get_image_backend()
print("torchvision image backend:", _backend)


def _seed_worker(worker_id: int):
    base_seed = SEED + worker_id
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)


from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2 as T

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_mean = None
_std = None
try:
    meta = getattr(weights_meta, "meta", None) or {}
    _mean = meta.get("mean", None)
    _std = meta.get("std", None)
except Exception:
    _mean, _std = None, None

if _mean is None or _std is None:
    _mean, _std = _IMAGENET_MEAN, _IMAGENET_STD

_tensor_preprocess = T.Compose(
    [
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(224),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=_mean, std=_std),
    ]
)


class TestImageDataset(Dataset):
    def __init__(self, df_test, preprocess_tensor):
        self.df = df_test.reset_index(drop=True)
        self.preprocess = preprocess_tensor
        self._image_id = self.df["image_id"].astype(str).to_numpy()
        self._image_path = self.df["image_path"].astype(str).to_numpy()
        self._location = self.df["location"].astype(np.int64).to_numpy()
        self._seq_id = self.df["seq_id"].astype(str).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self._image_id[idx]
        loc = int(self._location[idx])
        seq = self._seq_id[idx]
        path = self._image_path[idx]
        try:
            x = read_image(path, mode=ImageReadMode.RGB)
            x = self.preprocess(x)
        except Exception:
            x = torch.zeros((3, 224, 224), dtype=torch.float32)
            x = T.Normalize(mean=_mean, std=_std)(x)
        return image_id, loc, seq, x


def safe_collate(batch):
    ids, locs, seqs, xs = zip(*batch)
    return (
        list(ids),
        torch.as_tensor(locs, dtype=torch.int64),
        list(seqs),
        torch.stack(xs, dim=0),
    )


test_ds = TestImageDataset(df_test, _tensor_preprocess)

cpu_cnt = os.cpu_count() or 2
_num_workers = min(8, max(2, cpu_cnt // 2))

g = torch.Generator()
g.manual_seed(SEED)

pin = DEVICE.type == "cuda"
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=pin,
    pin_memory_device="cuda" if pin else "",
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    collate_fn=safe_collate,
    worker_init_fn=_seed_worker,
    generator=g,
)



## === cell 9
THRESH = 0.25

n_test = len(test_ds)
all_ids = [None] * n_test
all_preds = np.empty((n_test,), dtype=np.int64)

max_loc = int(max(df_train["location"].max(), df_test["location"].max()))
loc_top1_arr = np.full((max_loc + 1,), most_common_category, dtype=np.int64)
for k, v in loc_to_topcat.items():
    if 0 <= int(k) <= max_loc:
        loc_top1_arr[int(k)] = int(v)
loc_top1_t = torch.as_tensor(loc_top1_arr, device=DEVICE, dtype=torch.long)

most_common_tensor = torch.tensor(most_common_category, device=DEVICE, dtype=torch.long)
zero_tensor = torch.tensor(0, device=DEVICE, dtype=torch.long)

use_mapping = imagenet_to_iwild is not None
if use_mapping:
    imagenet_to_iwild_t = torch.as_tensor(
        imagenet_to_iwild, device=DEVICE, dtype=torch.long
    )

global_cat_counts = (
    df_train.groupby("category_id", as_index=True)["count"].sum().astype(np.float64)
)
global_log_prior = np.log(global_cat_counts + 1.0)
global_log_prior = global_log_prior / (global_log_prior.max() + 1e-12)
global_log_prior = global_log_prior.to_dict()


def choose_loc_topk(loc: int) -> int:
    if loc in loc_to_topk:
        cids, cnts = loc_to_topk[loc]
        cnts = cnts.astype(np.float64)
        cnts = cnts / (cnts.max() + 1e-12)
        best_score = -1e18
        best_cid = int(cids[0])
        for cid, c in zip(cids, cnts):
            score = float(c) + 0.15 * float(global_log_prior.get(int(cid), 0.0))
            if score > best_score:
                best_score = score
                best_cid = int(cid)
        return best_cid
    return int(most_common_category)


start = 0
with torch.inference_mode():
    for ids, locs, seqs, x in tqdm(
        test_loader, total=math.ceil(n_test / test_loader.batch_size)
    ):
        bs = x.shape[0]
        end = start + bs

        if DEVICE.type == "cuda":
            x = x.to(device=DEVICE, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            locs = locs.to(device=DEVICE, non_blocking=True)
        else:
            x = x.to(device=DEVICE)
            locs = locs.to(device=DEVICE)

        logits = model(x)

        max_subset_logit = logits.index_select(1, animal_indices).max(dim=1).values
        lse = torch.logsumexp(logits, dim=1)
        animal_prob = torch.exp(max_subset_logit - lse)

        loc_fallback = (
            loc_top1_t.gather(0, locs.clamp(min=0, max=max_loc))
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        seq_fallback = np.empty((bs,), dtype=np.int64)
        for i in range(bs):
            seq_fallback[i] = int(seq_to_topcat.get(str(seqs[i]), int(loc_fallback[i])))

        animal_mask = (animal_prob > THRESH).detach().cpu().numpy()
        locs_cpu = locs.detach().cpu().numpy().astype(np.int64)

        loc_pred_np = np.empty((bs,), dtype=np.int64)
        loc_pred_np[:] = seq_fallback
        if animal_mask.any():
            idxs = np.nonzero(animal_mask)[0]
            for j in idxs:
                if str(seqs[j]) not in seq_to_topcat:
                    loc_pred_np[j] = choose_loc_topk(int(locs_cpu[j]))
        loc_pred = torch.as_tensor(loc_pred_np, device=DEVICE, dtype=torch.long)

        if use_mapping:
            top1 = logits.argmax(dim=1)
            mapped = imagenet_to_iwild_t.gather(0, top1)
            mapped2 = torch.where(mapped == most_common_tensor, loc_pred, mapped)
            pred = torch.where(animal_prob > THRESH, mapped2, zero_tensor)
        else:
            pred = torch.where(animal_prob > THRESH, loc_pred, zero_tensor)

        pred = pred.detach().cpu().numpy().astype(np.int64)

        all_ids[start:end] = ids
        all_preds[start:end] = pred
        start = end

pred_by_id = pd.Series(all_preds, index=pd.Index(all_ids, name="Id"))

print("Pred count:", len(pred_by_id), "Expected:", len(df_test))
assert pred_by_id.index.is_unique, "Unexpected duplicate Ids in predictions"



## === cell 10
sample = pd.read_csv(SAMPLE_SUB)
assert "Id" in sample.columns, "sample_submission missing Id column"

id_col = "Id"
out_pred_col = "Category"

aligned = pred_by_id.reindex(sample[id_col])
sample[out_pred_col] = aligned.fillna(0).astype(np.int64).to_numpy()
sample = sample[[id_col, out_pred_col]]

out_path = "/kaggle/working/submission.csv"
sample.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sample.head())
print("Rows:", len(sample), "Unique Ids:", sample["Id"].nunique())
assert out_path.endswith(".csv") and os.path.exists(out_path)
assert len(sample) == 60760, "Submission row count mismatch vs expected 60760"
assert sample["Id"].isna().sum() == 0, "NaNs in Id column"
assert sample["Category"].isna().sum() == 0, "NaNs in Category column"
