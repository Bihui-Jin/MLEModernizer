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

albumentations==2.0.8
geopandas==0.14.4
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

0.9498136762796549

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fixed the import errors, added a fallback implementation for the missing `segmentation_models_pytorch` package, imported the missing pandas and dataclass utilities, and re‑ordered the notebook cells into a clean sequential script that loads the sample submission, builds the dataset, loads (or mocks) the models, runs inference, and finally writes a valid `submission.csv` file.'
- What this solution (achieved 0.0) has done: 'I fixed the missing rasterio dependency by replacing it with a simple cv2 loader, added proper imports (including `dataclass`), set a device‑aware handling for CUDA/CPU, corrected the DataLoader collate function so strings are handled, and adjusted the inference loop to work with the new data structures. These changes unblock execution, produce a valid submission.csv and keep the core modeling logic unchanged, moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'I turn off the background‑skip flag and ensure the inference loop always runs over the test rows, so predictions are generated and a proper RLE string is written to the submission file. This small change should move the score from 0.0 toward the target without altering the core model or training logic.'

# 9. Code solution

## === cell 0
BATCH_SIZE = 1
NUM_WORKERS = 0
CROP_SIZE = 1024 * 4  # 4096
STEP = 1024 * 2  # 2048
THR = 0.5
VOTE = 0
SKIP_COMMIT = True
RAMKA = 256

SKIP_BACKGROUND = False

df_sub = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2105755756.py in <cell line: 0>()
     10 SKIP_BACKGROUND = False
     11 
---> 12 df_sub = pd.read_csv("../input/hubmap-kidney-segmentation/sample_submission.csv")
     13 
     14 

NameError: name 'pd' is not defined

## === cell 1
def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    if pixels.size > 0:
        pixels[0] = 0
        pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def valid_transform(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def _check_background(img, crop_size) -> bool:
    s_th = 40  # saturation blanking threshold
    p_th = 1000 * (crop_size // 256) ** 2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, ss, _ = cv2.split(hsv)
    background = False if (ss > s_th).sum() <= p_th or img.sum() <= p_th else True
    return background




## === cell 2
from PIL import Image  # added fallback image loader


class SingleTiffDataset(Dataset):
    def __init__(self, tiff_path, all_img_sizes, crop_size=1024, step=512):
        self.crop_size = crop_size
        self.all_img_sizes = all_img_sizes
        self.step = step

        try:
            img = cv2.imread(tiff_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imread returned None")
        except Exception:
            pil_img = Image.open(tiff_path)
            pil_img = pil_img.convert("RGB")  # ensure 3 channels
            img = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR for consistency

        if img is None:
            raise FileNotFoundError(f"Unable to read image: {tiff_path}")

        self.full_img = img
        self.h, self.w, _ = img.shape

        self.row_count = 1 + math.ceil((self.h - self.crop_size) / self.step)
        self.col_count = 1 + math.ceil((self.w - self.crop_size) / self.step)

    def __len__(self):
        return self.row_count * self.col_count

    def __getitem__(self, idx):
        y = (idx // self.col_count) * self.step
        x = (idx % self.col_count) * self.step
        if x + self.crop_size > self.w:
            x = self.w - self.crop_size
        if y + self.crop_size > self.h:
            y = self.h - self.crop_size

        img_crop = self.full_img[y : y + self.crop_size, x : x + self.crop_size]

        transformed = {
            size: valid_transform(size)(image=img_crop)["image"]
            for size in self.all_img_sizes
        }
        transformed["crop_names"] = f"{x}_{y}"
        transformed["not_background"] = _check_background(img_crop, self.crop_size)
        return transformed




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1677528300.py in <cell line: 0>()
      2 
      3 
----> 4 class SingleTiffDataset(Dataset):
      5     def __init__(self, tiff_path, all_img_sizes, crop_size=1024, step=512):
      6         self.crop_size = crop_size

NameError: name 'Dataset' is not defined

## === cell 3
def inference(data_loader, models_with_config, crop_size):
    img_h, img_w = data_loader.dataset.h, data_loader.dataset.w
    mask_pred = np.zeros((img_h, img_w), dtype=np.uint8)

    for batch in tqdm(data_loader, ncols=70, leave=False):
        if SKIP_BACKGROUND and not batch["not_background"]:
            continue

        with torch.no_grad():
            pred_total = None

            for cfg, model_group in models_with_config:
                image = batch[cfg.img_size].to(device)
                pred_group = None

                for model in model_group:
                    pred = model(image).sigmoid()
                    if cfg.img_size != TRAIN_IMG_SIZE:
                        pred = torch.nn.functional.interpolate(
                            pred,
                            size=TRAIN_IMG_SIZE,
                            mode="bilinear",
                            align_corners=False,
                        )
                    if pred_group is None:
                        pred_group = pred
                    else:
                        pred_group += pred

                pred_group = pred_group.squeeze()
                if pred_group.ndim == 2:
                    pred_group = pred_group.unsqueeze(0)

                pred_group = pred_group.cpu()
                pred_group = cfg.weight_blend * (pred_group / len(model_group))

                if pred_total is None:
                    pred_total = pred_group
                else:
                    pred_total += pred_group

            pred_total = (pred_total.numpy() > THR).astype(np.uint8)

            crop_name = batch["crop_names"]
            x, y = map(int, crop_name.split("_"))

            if crop_size != TRAIN_IMG_SIZE:
                pred_total = np.array(
                    [cv2.resize(pred_total[0], (crop_size, crop_size))]
                )

            predict_single = pred_total[0]
            predict_single[:RAMKA, :] = 0
            predict_single[-RAMKA:, :] = 0
            predict_single[:, :RAMKA] = 0
            predict_single[:, -RAMKA:] = 0

            mask_pred[y : y + crop_size, x : x + crop_size] += predict_single

    mask_pred = (mask_pred > VOTE).astype(np.uint8)
    return rle_encode_less_memory(mask_pred)




## === cell 4
@dataclass
class ModelConfig:
    encoder: str
    weights_path: List[str]
    img_size: int
    weight_blend: float


my_models = [
    ModelConfig(
        "resnet34",
        [
            "./model_1/fold0_avg_0.9238.pth",
            "./model_1/fold1_avg_0.9162.pth",
            "./model_1/fold2_avg_0.9303.pth",
            "./model_1/fold3_avg_0.9336.pth",
            "./model_1/fold4_avg_0.9407.pth",
        ],
        4096,
        0.4,
    ),
    ModelConfig(
        "timm-efficientnet-b3",
        [
            "../input/hubmap5/exp_25_fold_0.pt",
            "../input/hubmap5/exp_24_fold_1.pt",
            "../input/hubmap5/exp_23_fold_2.pt",
            "../input/hubmap5/exp_22_fold_3.pt",
            "../input/hubmap5/exp_21_fold_4.pt",
        ],
        1024,
        0.2,
    ),
    ModelConfig(
        "se_resnext50_32x4d",
        [
            "./model_2/fold0_avg_0.9457.pth",
            "./model_2/fold1_avg_0.9566.pth",
            "./model_2/fold2_avg_0.9379.pth",
            "./model_2/fold3_avg_0.9095.pth",
            "./model_2/fold4_avg_0.9338.pth",
        ],
        2048,
        0.4,
    ),
]

TRAIN_IMG_SIZE = max(cfg.img_size for cfg in my_models)
all_img_sizes = set(cfg.img_size for cfg in my_models)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2543768463.py in <cell line: 0>()
----> 1 @dataclass
      2 class ModelConfig:
      3     encoder: str
      4     weights_path: List[str]
      5     img_size: int

NameError: name 'dataclass' is not defined

## === cell 5
models_with_config = []
for cfg in my_models:
    group = []
    for w_path in cfg.weights_path:
        model = smp.Unet(encoder=cfg.encoder, encoder_weights=None, classes=1).to(
            device
        )
        try:
            state = torch.load(w_path, map_location="cpu")
            model.load_state_dict(state)
        except Exception:
            pass
        model.eval()
        group.append(model)
    models_with_config.append((cfg, group))




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/660512736.py in <cell line: 0>()
      1 models_with_config = []
----> 2 for cfg in my_models:
      3     group = []
      4     for w_path in cfg.weights_path:
      5         model = smp.Unet(encoder=cfg.encoder, encoder_weights=None, classes=1).to(

NameError: name 'my_models' is not defined

## === cell 6
for idx, row in df_sub.iterrows():
    test_ds = SingleTiffDataset(
        tiff_path=f"../input/hubmap-kidney-segmentation/test/{row['id']}.tiff",
        all_img_sizes=all_img_sizes,
        crop_size=CROP_SIZE,
        step=STEP,
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda batch: batch[0],  # return dict directly for batch size 1
    )
    rle = inference(test_loader, models_with_config, CROP_SIZE)
    df_sub.at[idx, "predicted"] = rle




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/4062300377.py in <cell line: 0>()
----> 1 for idx, row in df_sub.iterrows():
      2     test_ds = SingleTiffDataset(
      3         tiff_path=f"../input/hubmap-kidney-segmentation/test/{row['id']}.tiff",
      4         all_img_sizes=all_img_sizes,
      5         crop_size=CROP_SIZE,

NameError: name 'df_sub' is not defined

## === cell 7
for folder in ["model_1", "model_2"]:
    if os.path.isdir(folder):
        try:
            import shutil

            shutil.rmtree(folder)
        except Exception:
            pass

df_sub.to_csv("submission.csv", index=False)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3198848411.py in <cell line: 0>()
      1 for folder in ["model_1", "model_2"]:
----> 2     if os.path.isdir(folder):
      3         try:
      4             import shutil
      5 

NameError: name 'os' is not defined
