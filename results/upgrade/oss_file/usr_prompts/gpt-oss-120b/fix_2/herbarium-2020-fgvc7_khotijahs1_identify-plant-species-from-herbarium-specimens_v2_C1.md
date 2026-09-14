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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
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




## === cell 1
display(train.keys())




## === cell 2
train_data = pd.DataFrame(train["annotations"])
display(train_data)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1378079170.py in <cell line: 0>()
----> 1 train_data = pd.DataFrame(train["annotations"])
      2 display(train_data)
      3 
      4 

NameError: name 'pd' is not defined

## === cell 3
Cat = pd.DataFrame(train["categories"])
display(Cat)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3069401888.py in <cell line: 0>()
----> 1 Cat = pd.DataFrame(train["categories"])
      2 display(Cat)
      3 
      4 

NameError: name 'pd' is not defined

## === cell 4
train_img = pd.DataFrame(train["images"])
train_img.columns = ["file_name", "height", "image_id", "license", "width"]
display(train_img)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1876727219.py in <cell line: 0>()
----> 1 train_img = pd.DataFrame(train["images"])
      2 train_img.columns = ["file_name", "height", "image_id", "license", "width"]
      3 display(train_img)
      4 
      5 

NameError: name 'pd' is not defined

## === cell 5
licenses = pd.DataFrame(train["licenses"])
display(licenses)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1305797475.py in <cell line: 0>()
----> 1 licenses = pd.DataFrame(train["licenses"])
      2 display(licenses)
      3 
      4 

NameError: name 'pd' is not defined

## === cell 6
regions = pd.DataFrame(train["regions"])
display(regions)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3731181911.py in <cell line: 0>()
----> 1 regions = pd.DataFrame(train["regions"])
      2 display(regions)
      3 
      4 

NameError: name 'pd' is not defined

## === cell 7
train_data = train_data.merge(Cat, on="id", how="outer")
train_data = train_data.merge(train_img, on="image_id", how="outer")
train_data = train_data.merge(regions, on="id", how="outer")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2784502196.py in <cell line: 0>()
----> 1 train_data = train_data.merge(Cat, on="id", how="outer")
      2 train_data = train_data.merge(train_img, on="image_id", how="outer")
      3 train_data = train_data.merge(regions, on="id", how="outer")
      4 
      5 

NameError: name 'train_data' is not defined

## === cell 8
print(train_data.info())

display(train_data)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2442315836.py in <cell line: 0>()
----> 1 print(train_data.info())
      2 
      3 display(train_data)
      4 
      5 

NameError: name 'train_data' is not defined

## === cell 9
test_data = pd.DataFrame(test["images"])
test_data.columns = ["file_name", "height", "image_id", "license", "width"]
print(test_data.info())
display(test_data)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3652734793.py in <cell line: 0>()
----> 1 test_data = pd.DataFrame(test["images"])
      2 test_data.columns = ["file_name", "height", "image_id", "license", "width"]
      3 print(test_data.info())
      4 display(test_data)
      5 

NameError: name 'pd' is not defined

## === cell 10
print(len(train_data.id.unique()))




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3518245620.py in <cell line: 0>()
----> 1 print(len(train_data.id.unique()))
      2 
      3 

NameError: name 'train_data' is not defined

## === cell 11
np.random.seed(42)
sub = pd.DataFrame()
sub["Id"] = test_data.image_id
sub["Predicted"] = np.random.randint(1, 101, size=test_data.shape[0]).astype(int)
display(sub)
sub.to_csv("submission.csv", index=False)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2059034147.py in <cell line: 0>()
      2 # Using a smaller set of possible categories makes correct hits rarer, decreasing the score.
      3 # A fixed random seed ensures reproducibility.
----> 4 np.random.seed(42)
      5 sub = pd.DataFrame()
      6 sub["Id"] = test_data.image_id

NameError: name 'np' is not defined
