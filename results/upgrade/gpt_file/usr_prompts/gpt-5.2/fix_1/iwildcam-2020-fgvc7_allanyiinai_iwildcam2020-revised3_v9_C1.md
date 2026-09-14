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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
from IPython import display

## === cell 2
import os
def is_in_ipython():
    "Is the code running in the ipython environment (jupyter including)"
    program_name = os.path.basename(os.getenv('_', ''))

    if ('jupyter-notebook' in program_name or # jupyter-notebook
        'ipython'          in program_name or # ipython
        'jupyter' in program_name or  # jupyter
        'JPY_PARENT_PID'   in os.environ):    # ipython-notebook
        return True
    else:
        return False


def is_in_colab():
    if not is_in_ipython(): return False
    try:
        from google import colab
        return True
    except: return False

def is_in_kaggle_kernal():
    if 'kaggle' in os.environ['PYTHONPATH']:
        return True
    else:
        return False

if is_in_colab():
    from google.colab import drive
    drive.mount('/content/gdrive')

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/3371189574.py in <cell line: 0>()
     31 if is_in_colab():
     32     from google.colab import drive
---> 33     drive.mount('/content/gdrive')

/usr/local/lib/python3.11/dist-packages/google/colab/drive.py in mount(mountpoint, force_remount, timeout_ms, readonly)
     98 def mount(mountpoint, force_remount=False, timeout_ms=120000, readonly=False):
     99   """Mount your Google Drive at the specified mountpoint path."""
--> 100   return _mount(
    101       mountpoint,
    102       force_remount=force_remount,

/usr/local/lib/python3.11/dist-packages/google/colab/drive.py in _mount(mountpoint, force_remount, timeout_ms, ephemeral, readonly)
    116   """Internal helper to mount Google Drive."""
    117   if not _os.path.exists('/var/colab/hostname'):
--> 118     raise NotImplementedError(
    119         'Mounting drive is unsupported in this environment. Use PyDrive2'
    120         ' instead. See examples at'

NotImplementedError: Mounting drive is unsupported in this environment. Use PyDrive2 instead. See examples at https://colab.research.google.com/notebooks/io.ipynb#scrollTo=7taylj9wpsA2.

## === cell 3
os.environ['TRIDENT_BACKEND'] = 'pytorch'

if is_in_kaggle_kernal():
    os.environ['TRIDENT_HOME'] = './trident'
    
elif is_in_colab():
    os.environ['TRIDENT_HOME'] = '/content/gdrive/My Drive/trident'

!pip uninstall tridentx -y
!pip install '../input/trident/tridentx-0.7.5-py3-none-any.whl' --upgrade
import json
import copy
import numpy as np
import trident as T
from trident import *
from trident.models import resnet,efficientnet

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2184896537.py in <cell line: 0>()
     14 import numpy as np
     15 #調用trident api
---> 16 import trident as T
     17 from trident import *
     18 from trident.models import resnet,efficientnet

ModuleNotFoundError: No module named 'trident'

## === cell 4
import pandas as pd

with open('../input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as json_file:
    train_data = json.load(json_file)


with open('../input/iwildcam-2020-fgvc7/iwildcam2020_test_information.json') as test_json_file:
    test_data = json.load(test_json_file)

    
df_train = pd.DataFrame({'id': [item['id'] for item in train_data['annotations']],
                         'category_id': [item['category_id'] for item in train_data['annotations']],
                         'image_id': [item['image_id'] for item in train_data['annotations']],
                         'location': [item['location'] for item in train_data['images']],
                         'file_name': [item['file_name'] for item in train_data['images']]})
df_test = pd.DataFrame({'image_id': [item['id'] for item in train_data['images']],
                         'location': [item['location'] for item in train_data['images']],
                         'file_name': [item['file_name'] for item in train_data['images']]})



df_train

## === cell 5
df_category_train=pd.DataFrame({'id': [item['id'] for item in train_data['categories']],
                         'name': [item['name'] for item in train_data['categories']],
                         'count': [item['count'] for item in train_data['categories']]})

df_category_test=pd.DataFrame({'id': [item['id'] for item in test_data['categories']],
                         'name': [item['name'] for item in test_data['categories']],
                         'count': [item['count'] for item in test_data['categories']]})

df_category_train=df_category_train.sort_values(['count'],ascending=False) 
print(df_category_train)
df_category_test=df_category_test.sort_values(['count'],ascending=False) 
print(df_category_test)

animal_category_lists=list(sorted(set([item['category_id'] for item in train_data['annotations']])))


df_category_train=df_category_train[df_category_train['id'].isin(animal_category_lists)]
print(df_category_train)

df_category_test=df_category_test[df_category_test['count']>0]
print(df_category_test)

animal_category_lists_train=[category_id.item() for category_id in df_category_train[['id']].to_numpy().astype(np.int64)]
print(animal_category_lists_train[:5])

animal_category_lists_test=[category_id.item() for category_id in df_category_test[['id']].to_numpy().astype(np.int64)]
print(animal_category_lists_test[:5])

category_missing_list=[category_id for category_id in animal_category_lists_test if category_id not in animal_category_lists_train]
print(category_missing_list)



## === cell 7
label2category=OrderedDict()
category2label=OrderedDict()
for i in range(len(animal_category_lists_train)):
    category2label[animal_category_lists_train[i]]=i
    label2category[i]=animal_category_lists_train[i]

label_idxes=[category2label[item['category_id']] for item in train_data['annotations']]
image_pathes=['../input/iwildcam-2020-fgvc7/train/'+item['file_name'] for item in train_data['images']]

print('label_idxes',label_idxes[:5])
print('image_pathes',image_pathes[:5])
print('label2category',list(label2category.items())[:5])
print('category2label',list(category2label.items())[:5])

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1152334138.py in <cell line: 0>()
----> 1 label2category=OrderedDict()
      2 category2label=OrderedDict()
      3 #產生能將category_id轉label的字典
      4 for i in range(len(animal_category_lists_train)):
      5     category2label[animal_category_lists_train[i]]=i

NameError: name 'OrderedDict' is not defined

## === cell 9
rare_animals=OrderedDict()
common_animals=OrderedDict()
df_animal_frequency=df_train['category_id'].value_counts()

df_rare_animal_frequency=df_animal_frequency[df_animal_frequency<=10]
df_common_animal_frequency=df_animal_frequency[df_animal_frequency>1000]

for item in  df_rare_animal_frequency.iteritems() :
    rare_animals[item[0]]=item[1]
    
for item in  df_common_animal_frequency.iteritems() :
    common_animals[item[0]]=item[1]
    
print('rare_animals',len(rare_animals))
print('common_animals',len(common_animals))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1403753429.py in <cell line: 0>()
----> 1 rare_animals=OrderedDict()
      2 common_animals=OrderedDict()
      3 df_animal_frequency=df_train['category_id'].value_counts()
      4 
      5 df_rare_animal_frequency=df_animal_frequency[df_animal_frequency<=10]

NameError: name 'OrderedDict' is not defined

## === cell 11
import glob
imgs=glob.glob('../input/iwildcam-2020-fgvc7/train/*.jpg')
print(len(imgs))
print(imgs[:5])

img_pathes=[img.split('/')[-1] for img in imgs]
img_pathes=list(sorted(set(img_pathes)))
print(len(img_pathes))
print(img_pathes[:5])


## === cell 13
rare_images=[]
rare_labels=[]
for i in range(len(image_pathes)):
    img=image_pathes[i]
    label=label_idxes[i]
    if label2category[label] in rare_animals:
        cnt=rare_animals[label2category[label]]
        for n in range(int(20.0/cnt)):
             rare_images.append(img) 
             rare_labels.append(label) 
        
print(len(rare_images))
print(len(rare_labels))
print(rare_images[:5])
print(rare_labels[:5])

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/336283383.py in <cell line: 0>()
      1 rare_images=[]
      2 rare_labels=[]
----> 3 for i in range(len(image_pathes)):
      4     img=image_pathes[i]
      5     label=label_idxes[i]

NameError: name 'image_pathes' is not defined

## === cell 15

img_ds=ImageDataset(image_pathes+rare_images*10,symbol='image')
label_ds=LabelDataset(label_idxes+rare_labels*10,symbol='label')

sample_filter=lambda x:label2category[x[-1]] not in common_animals.key_list or random.random()<(1000.0/common_animals[label2category[x[-1]]])


data_provider=DataProvider(traindata=Iterator(data=img_ds,label=label_ds))

data_provider.image_transform_funcs=[
    Resize((224,224)),
    CLAHE(),
    RandomAdjustGamma(scale=(0.8,1.2)),#調整明暗
    RandomAdjustHue(scale=(-0.2,0.2)),#調整色相
    RandomAdjustSaturation(scale=(0.8,1.2)),#調整飽和度
    SaltPepperNoise(0.005, keep_prob=0.5),#加入胡椒鹽噪音
    RandomErasing(size_range=(0.05, 0.2), transparency_range=(0.4, 0.8), transparancy_ratio=1.0, keep_prob=0.5), #加入隨機擦去
    RandomTransformAffine(rotation_range=45, zoom_range=0.00, shift_range=0.00, shear_range=0.2, random_flip=0.15),#隨機仿射變換
    Normalize(127.5,127.5)] #標準化


data,labels=data_provider.next()
print(data.shape)
print(labels)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/135552824.py in <cell line: 0>()
----> 1 img_ds=ImageDataset(image_pathes+rare_images*10,symbol='image')
      2 label_ds=LabelDataset(label_idxes+rare_labels*10,symbol='label')
      3 
      4 #非常見動物的全部+常見動物基於頻率調整成為1000
      5 sample_filter=lambda x:label2category[x[-1]] not in common_animals.key_list or random.random()<(1000.0/common_animals[label2category[x[-1]]])

NameError: name 'ImageDataset' is not defined

## === cell 16
%%time
data_provider.preview_images()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed eval> in <module>

NameError: name 'data_provider' is not defined

## === cell 17
from trident.models import efficientnet

net1=efficientnet.EfficientNetB0(pretrained=True,include_top=False,classes=len(animal_category_lists_train),input_shape=(3,224,224),freeze_features=True)
net1.model.add_module('last_conv',Conv2d_Block((3,3),num_filters=len(animal_category_lists_train),use_bias=False,activation=None, normalization='l2'))
cam=ShortCut(
    Identity(),
    Sequential(
    GlobalAvgPool2d(),
    Reshape((len(animal_category_lists_train),1,1)),
    Conv2d((1,1),num_filters=len(animal_category_lists_train),use_bias=False,activation=None)
    )
,mode='dot'
)

net1.model.add_module('cam',cam)
net1.model.add_module('aggregate1',Aggregation('sum',axis=2))
net1.model.add_module('aggregate2',Aggregation('sum',axis=3))
net1.model.add_module('reshape',Reshape((len(animal_category_lists_train))))
net1.model.add_module('sigmoid',Sigmoid())
net1.model.add_module('fc',Dense((len(animal_category_lists_train))))
net1.model.add_module('softmax',SoftMax(axis=-1,add_noise=True,noise_intensity=0.12))
net1.summary()

net1.model.block7a.trainable=True

if os.path.exists('./Models/revised3_net1.pth.tar'):
    net1.load_model('./Models/revised3_net1.pth.tar')
elif os.path.exists('../input/iwildcam2020-revised3/Models/revised3_net1.pth.tar'):
    net1.load_model('../input/iwildcam2020-revised3/Models/revised3_net1.pth.tar')


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1277246733.py in <cell line: 0>()
----> 1 from trident.models import efficientnet
      2 
      3 net1=efficientnet.EfficientNetB0(pretrained=True,include_top=False,classes=len(animal_category_lists_train),input_shape=(3,224,224),freeze_features=True)
      4 net1.model.add_module('last_conv',Conv2d_Block((3,3),num_filters=len(animal_category_lists_train),use_bias=False,activation=None, normalization='l2'))
      5 cam=ShortCut(

ModuleNotFoundError: No module named 'trident'

## === cell 19
net2=efficientnet.EfficientNetB0(pretrained=True,include_top=False,classes=len(animal_category_lists_train),input_shape=(3,224,224),freeze_features=True)
net2.model.add_module('output_layer', 
    Sequential(
        Dropout(dropout_rate=0.4),
        Flatten(),
        Dense((512),use_bias=False),
    ))
net2.model.add_module('l2norm',L2Norm())
net2.model.add_module('fc',Dense((len(animal_category_lists_train)),use_bias=False,weights_norm='l2'))
net2.summary()
net2.model.block7a.trainable=True

if os.path.exists('./Models/revised3_net2.pth.tar'):
    net2.load_model('./Models/revised3_net2.pth.tar')
elif os.path.exists('../input/iwildcam2020-revised3/Models/revised3_net2.pth.tar'):
    net2.load_model('../input/iwildcam2020-revised3/Models/revised3_net2.pth.tar')


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/327973009.py in <cell line: 0>()
----> 1 net2=efficientnet.EfficientNetB0(pretrained=True,include_top=False,classes=len(animal_category_lists_train),input_shape=(3,224,224),freeze_features=True)
      2 net2.model.add_module('output_layer', 
      3     Sequential(
      4         Dropout(dropout_rate=0.4),
      5         Flatten(),

NameError: name 'efficientnet' is not defined

## === cell 20
from trident.models import visual_transformer
vit=visual_transformer.VisionTransformer_small(pretrained=True,input_shape=(3,224,224),patch_size=16,num_classes=len(animal_category_lists_train))



if os.path.exists('./Models/revised3_vit.pth.tar'):
    vit.load_model('./Models/revised3_vit.pth.tar')
elif os.path.exists('../input/iwildcam2020-revised3/Models/revised3_vit.pth.tar'):
    vit.load_model('../input/iwildcam2020-revised3/Models/revised3_vit.pth.tar')

vit.model.trainable=True
vit.summary()

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/183294550.py in <cell line: 0>()
----> 1 from trident.models import visual_transformer
      2 vit=visual_transformer.VisionTransformer_small(pretrained=True,input_shape=(3,224,224),patch_size=16,num_classes=len(animal_category_lists_train))
      3 
      4 
      5 

ModuleNotFoundError: No module named 'trident'

## === cell 22
class ArcMarginProductLoss(Layer):
    def __init__(self, scale=32.0, margin=0.50, easy_margin=False, name='ArcMarginProductLoss'):
        super(ArcMarginProductLoss, self).__init__()
        self._name=name
        self.scale = scale
        self.m = margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
        self.base_loss=CrossEntropyLoss(reduction='mean')


    def forward(self, output, target,**kwargs):
        try:
            cosine=output
            sine = sqrt(1.0 - pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = where(cosine > 0, phi, cosine)
            else:
                phi = where((cosine - self.th) > 0, phi, cosine - self.mm)

            one_hot = zeros_like(cosine,requires_grad=True)
            one_hot.scatter(1, target.view(-1, 1), 1)

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.scale
        except Exception as e:
            print(e)
            PrintException()

        loss = self.base_loss(output, target)
        return loss

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2370351309.py in <cell line: 0>()
----> 1 class ArcMarginProductLoss(Layer):
      2     def __init__(self, scale=32.0, margin=0.50, easy_margin=False, name='ArcMarginProductLoss'):
      3         super(ArcMarginProductLoss, self).__init__()
      4         self._name=name
      5         self.scale = scale

NameError: name 'Layer' is not defined

## === cell 23
net1.with_optimizer(optimizer=AdaBelief,lr=5e-4,betas=(0.9, 0.999),gradient_centralization='all')\
.with_loss(CrossEntropyLoss(auto_balance=True,label_smooth=True))\
.with_loss(FocalLoss,loss_weight=0.5)\
.with_metric(accuracy,name='accuracy')\
.with_metric(accuracy,topk=5,name='top5_accuracy',print_only=True)\
.with_regularizer('l2',reg_weight=5e-5)\
.adjust_learning_rate_scheduling(1000,unit='batch',new_value=5e-4)\
.with_accumulate_grads(4)\
.with_callbacks(MixupCallback(alpha= 1,loss_criterion=CrossEntropyLoss,loss_weight=0.5))\
.with_learning_rate_scheduler(StepLR(frequency=1000,unit='batch',gamma=0.5))\
.with_model_save_path('./Models/revised3_net1.pth')\
.with_automatic_mixed_precision_training()


net2.with_optimizer(optimizer=AdaBelief,lr=5e-4,betas=(0.9, 0.999),gradient_centralization='all')\
.with_loss(ArcMarginProductLoss(scale=32.0, margin=0.50, easy_margin=False)) \
.with_metric(accuracy,name='accuracy')\
.with_metric(accuracy,topk=5,name='top5_accuracy',print_only=True)\
.with_regularizer('l2',reg_weight=5e-5)\
.adjust_learning_rate_scheduling(1000,unit='batch',new_value=5e-4)\
.with_accumulate_grads(4)\
.with_grad_clipping(3)\
.with_learning_rate_scheduler(StepLR(frequency=1000,unit='batch',gamma=0.5))\
.with_model_save_path('./Models/revised3_net2.pth')\
.with_automatic_mixed_precision_training()


vit.with_optimizer(optimizer=DiffGrad,lr=5e-5,betas=(0.9, 0.999),gradient_centralization='all')\
.with_loss(ArcMarginProductLoss(scale=32.0, margin=0.80, easy_margin=False)) \
.with_metric(accuracy,name='accuracy')\
.with_metric(accuracy,topk=5,name='top5_accuracy',print_only=True)\
.with_regularizer('l2',reg_weight=5e-5)\
.with_accumulate_grads(5)\
.with_learning_rate_scheduler(StepLR(frequency=1000,unit='batch',gamma=0.5))\
.with_model_save_path('./Models/revised3_vit.pth')\
.with_automatic_mixed_precision_training()


plan=TrainingPlan()\
    .add_training_item(net1,name='net1')\
    .add_training_item(net2,name='net2')\
    .add_training_item(vit,name='vit')\
    .with_data_loader(data_provider)\
    .with_batch_size(32)\
    .print_gradients_scheduling(200,unit='batch') \
    .print_progress_scheduling(10,unit='batch') \
    .display_loss_metric_curve_scheduling(200)\
    .save_model_scheduling(100,unit='batch')


plan.only_steps(num_steps=10000, collect_data_inteval=5)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2345542144.py in <cell line: 0>()
----> 1 net1.with_optimizer(optimizer=AdaBelief,lr=5e-4,betas=(0.9, 0.999),gradient_centralization='all')\
      2 .with_loss(CrossEntropyLoss(auto_balance=True,label_smooth=True))\
      3 .with_loss(FocalLoss,loss_weight=0.5)\
      4 .with_metric(accuracy,name='accuracy')\
      5 .with_metric(accuracy,topk=5,name='top5_accuracy',print_only=True)\

NameError: name 'net1' is not defined

## === cell 24
bad_images=['86994b3e-21bc-11ea-a13a-137349068a90.jpg',
'882a533a-21bc-11ea-a13a-137349068a90.jpg',
'88a28616-21bc-11ea-a13a-137349068a90.jpg',
'88b99aae-21bc-11ea-a13a-137349068a90.jpg',
'89362ed4-21bc-11ea-a13a-137349068a90.jpg',
'8985bb98-21bc-11ea-a13a-137349068a90.jpg',
'89e09b26-21bc-11ea-a13a-137349068a90.jpg',
'8a804608-21bc-11ea-a13a-137349068a90.jpg',
'8b8e02a6-21bc-11ea-a13a-137349068a90.jpg',
'8b91394e-21bc-11ea-a13a-137349068a90.jpg',
'8cc46b6a-21bc-11ea-a13a-137349068a90.jpg',
'8d705d8a-21bc-11ea-a13a-137349068a90.jpg',
'8e930668-21bc-11ea-a13a-137349068a90.jpg',
'8e940310-21bc-11ea-a13a-137349068a90.jpg',
'8ea6a768-21bc-11ea-a13a-137349068a90.jpg',
'8fff9dc2-21bc-11ea-a13a-137349068a90.jpg',
'9044a3b8-21bc-11ea-a13a-137349068a90.jpg',
'920ee4c4-21bc-11ea-a13a-137349068a90.jpg',
'950ed288-21bc-11ea-a13a-137349068a90.jpg',
'9522d4fe-21bc-11ea-a13a-137349068a90.jpg',
'96bacf06-21bc-11ea-a13a-137349068a90.jpg',
'98552f5a-21bc-11ea-a13a-137349068a90.jpg',
'98da656c-21bc-11ea-a13a-137349068a90.jpg',
'9955d012-21bc-11ea-a13a-137349068a90.jpg']



test_imgs=glob.glob('../input/iwildcam-2020-fgvc7/test/*.jpg')
print(len(test_imgs))
test_imgs=[img_path for img_path in test_imgs if img_path.split('/')[-1] not in bad_images]
print(len(test_imgs))


img_ds=ImageDataset(test_imgs,symbol='image')
imgpath_ds=ImageDataset(test_imgs,object_type=ObjectType.image_path,symbol='img_path')

test_data_provider=DataProvider(traindata=Iterator(data=img_ds,label=imgpath_ds,is_shuffle=False))

test_data_provider.image_transform_funcs=[
    Resize((224,224)),
    CLAHE(),
    Normalize(127.5,127.5)] #標準化


data,labels=test_data_provider.next()
print(data.shape)
print(labels)
print(labels[0].item())
test_data_provider.preview_images()

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/221740914.py in <cell line: 0>()
     32 
     33 
---> 34 img_ds=ImageDataset(test_imgs,symbol='image')
     35 #請注意，要設定object_type=ObjectType.image_path，這樣就可以確保輸出為stype=np.string_的numpy array
     36 imgpath_ds=ImageDataset(test_imgs,object_type=ObjectType.image_path,symbol='img_path')

NameError: name 'ImageDataset' is not defined

## === cell 25
f=open('../input/iwildcam-2020-fgvc7/sample_submission.csv','r',encoding='utf-8-sig')
rows=f.readlines()
print(rows[:5])
print(rows[-5:])

submission_dict=OrderedDict()
for row in rows[1:]:
    cols=row.strip().split(',')
    if  cols[0]+'.jpg' not in bad_images:
        submission_dict[cols[0]]=None
print(len(submission_dict))

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2003727101.py in <cell line: 0>()
      5 print(rows[-5:])
      6 
----> 7 submission_dict=OrderedDict()
      8 for row in rows[1:]:
      9     cols=row.strip().split(',')

NameError: name 'OrderedDict' is not defined

## === cell 26
import torch
if is_gpu_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2523063259.py in <cell line: 0>()
      1 #清除gpu快取
      2 import torch
----> 3 if is_gpu_available():
      4     torch.cuda.synchronize()
      5     torch.cuda.empty_cache()

NameError: name 'is_gpu_available' is not defined

## === cell 27
net1.eval()
net2.eval()
vit.eval()
test_data_provider.batch_size=64
test_data_provider.traindata.batch_sampler.reset()


for i,(img,img_path) in tqdm(enumerate(test_data_provider)):
    out2=np.argmax(net2(img),axis=1)
    for k in range(len(out2)):
        _,image_id,_=split_path(img_path[k].item())
        if i==0 and k==0:
            print(image_id)
        category_id=label2category[out2[k]]
        if image_id in submission_dict and image_id+'.jpg' not in bad_images:
            submission_dict[image_id]=category_id
        if None not in submission_dict.value_list:
            break
  

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1455953809.py in <cell line: 0>()
----> 1 net1.eval()
      2 net2.eval()
      3 vit.eval()
      4 #設定批次為64
      5 test_data_provider.batch_size=64

NameError: name 'net1' is not defined

## === cell 28
submission_rows=['Id,Category\n']
for k,v in submission_dict.items():
    submission_rows.append('{0},{1}\n'.format(k,v))
print(len(submission_rows))
make_dir_if_need('./results')
with open('./results/submission.csv','w',encoding='utf-8-sig') as f:
    f.writelines(submission_rows)
fr=open('./results/submission.csv','r',encoding='utf-8-sig')
rows=fr.readlines()
print(rows[:3])

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3933976054.py in <cell line: 0>()
      1 submission_rows=['Id,Category\n']
----> 2 for k,v in submission_dict.items():
      3     submission_rows.append('{0},{1}\n'.format(k,v))
      4 #確認提交檔筆數
      5 print(len(submission_rows))

NameError: name 'submission_dict' is not defined
