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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8791175581746751

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12332) has done: 'I fixed the Albumentations `RandomResizedCrop` usage by providing the size as a tuple (384, 384) which resolves the validation error and allows `tta_transform` to be defined. With this correction the test loop can access `tta_transform`, run the TTA predictions, and generate a proper submission CSV whose length matches the test set.'
- What this solution (achieved 0.16405) has done: 'The changes preload‑resize the images to the network size, batch the TTA augmentations so each model processes a 5‑image batch in a single forward pass, and simplify the DataLoader workers. This keeps the exact same augmentation and averaging logic while cutting the number of forward passes per image from 20 to 4, which brings the total run time well under 600 seconds.'
- What this solution (achieved 0.30344) has done: 'I simplify the ensemble to use only the EfficientNet‑7 model, which was the best‑performing checkpoint (≈0.8799 validation accuracy). By relying on this single strong model we avoid diluting its predictions with weaker models and move the Kaggle score closer to the target. The change is limited to the inference loop and does not alter any model architecture, training logic, or data handling.'
- What this solution (achieved 0.21413) has done: 'The fix adds all required imports, defines missing objects, and corrects the inference loop to use the strongest model (EfficientNet‑7) so the produced predictions are more likely to meet the target accuracy. It also ensures the submission CSV is written correctly.'
- What this solution (achieved 0.09978) has done: 'The fix adds checks for whether the custom EfficientNet‑7 checkpoint is actually loaded; if not, it falls back to using the ResNet‑50 checkpoint (which is present in the input data). The inference loop now averages predictions from any available models, ensuring that at least one pretrained model contributes to the final scores, which should move the Kaggle accuracy much closer to the target.'

# 9. Code solution

## === cell 0
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"



## === cell 1
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1771389829.py in <cell line: 0>()
----> 1 test_df = pd.read_csv(
      2     "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
      3 )
      4 

NameError: name 'pd' is not defined

## === cell 2
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),  # Resize to match EfficientNetV2-S input size
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3766273052.py in <cell line: 0>()
----> 1 efficientnet_transforms = A.Compose(
      2     [
      3         A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
      4         A.Resize(384, 384),  # Resize to match EfficientNetV2-S input size
      5         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

NameError: name 'A' is not defined

## === cell 3
class CassavaTestDataset(Dataset):
    """
    Loads all test images into memory once during initialization to avoid
    per‑sample disk I/O during inference. Images are also resized to the
    model input size (384×384) here, which matches the later transforms and
    reduces CPU time for each TTA augmentation.
    """

    def __init__(self, dataframe, image_dir):
        self.image_dir = image_dir
        self.image_ids = dataframe.iloc[:, 0].tolist()
        self.images = []
        for img_name in tqdm(self.image_ids, desc="Preloading test images"):
            img_path = os.path.join(self.image_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
            self.images.append(img)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        return self.images[idx], self.image_ids[idx]




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2820677875.py in <cell line: 0>()
----> 1 class CassavaTestDataset(Dataset):
      2     """
      3     Loads all test images into memory once during initialization to avoid
      4     per‑sample disk I/O during inference. Images are also resized to the
      5     model input size (384×384) here, which matches the later transforms and

NameError: name 'Dataset' is not defined

## === cell 4
tta_transform = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/985132292.py in <cell line: 0>()
      2 # Removing random crops / flips avoids degrading predictions while
      3 # still matching the training preprocessing.
----> 4 tta_transform = A.Compose(
      5     [
      6         A.Resize(384, 384),

NameError: name 'A' is not defined

## === cell 5
def tta_predict_single_model(model, image, tta_transform, device, n_tta=5):
    """
    Perform TTA inference using a single batched forward pass.
    The function still averages over ``n_tta`` augmentations, preserving
    the original semantics, but runs the model only once per image
    (batch size = n_tta), which dramatically reduces GPU kernel launch
    overhead without altering predictions.
    """
    model.eval()
    if image.shape[-1] != 3:
        raise ValueError("Image must have 3 channels (H, W, 3)")

    aug_tensors = [tta_transform(image=image)["image"] for _ in range(n_tta)]
    batch = torch.stack(aug_tensors).to(device)  # shape: (n_tta, C, H, W)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(batch)  # (n_tta, num_classes)
            probs = F.softmax(outputs, dim=1)  # (n_tta, num_classes)

    avg_probs = probs.mean(dim=0, keepdim=True)  # (1, num_classes)
    return avg_probs




## === cell 6
def identity_collate(batch):
    return batch


test_dataset = CassavaTestDataset(
    test_df,
    test_image_dir,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,  # no extra workers needed after preloading
    pin_memory=True,
    collate_fn=identity_collate,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3797337517.py in <cell line: 0>()
      3 
      4 
----> 5 test_dataset = CassavaTestDataset(
      6     test_df,
      7     test_image_dir,

NameError: name 'CassavaTestDataset' is not defined

## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/780918693.py in <cell line: 0>()
----> 1 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      2 

NameError: name 'torch' is not defined

## === cell 8
resnet_model = models.resnet50(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

try:
    resnet_state = torch.load(
        "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
        map_location=device,
    )
    resnet_model.load_state_dict(resnet_state, strict=False)
    resnet_loaded = True
except Exception as e:
    print(f"Custom ResNet weights not found ({e}); using ImageNet pretrained weights.")
    resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)
    resnet_loaded = False

resnet_model = resnet_model.to(device)
resnet_model.eval()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3649727761.py in <cell line: 0>()
----> 1 resnet_model = models.resnet50(pretrained=False)
      2 num_ftrs = resnet_model.fc.in_features
      3 resnet_model.fc = nn.Linear(num_ftrs, 5)
      4 
      5 try:

NameError: name 'models' is not defined

## === cell 9
efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
try:
    efficientnet_state = torch.load(
        "/kaggle/input/eff-7-last/pytorch/default/1/Eff7_0.8799.pth",
        map_location=device,
    )
    efficientnet_model_7.load_state_dict(efficientnet_state, strict=False)
    efficientnet_loaded = True
except Exception as e:
    print(
        f"Custom EfficientNet-7 weights not found ({e}); using ImageNet pretrained weights."
    )
    efficientnet_model_7 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    efficientnet_model_7.classifier[1] = nn.Linear(
        efficientnet_model_7.classifier[1].in_features, 5
    )
    efficientnet_loaded = False
efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/951095411.py in <cell line: 0>()
----> 1 efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
      2 num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
      3 efficientnet_model_7.classifier = nn.Sequential(
      4     nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
      5 )

NameError: name 'models' is not defined

## === cell 10
efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)
efficientnet_model_1.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
try:
    efficientnet_model_1.load_state_dict(
        torch.load(
            "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth", map_location=device
        ),
        strict=False,
    )
    efficientnet1_loaded = True
except Exception as e:
    print(
        f"Custom EfficientNet-1 weights not found ({e}); using ImageNet pretrained weights."
    )
    efficientnet_model_1 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    efficientnet_model_1.classifier[1] = nn.Linear(
        efficientnet_model_1.classifier[1].in_features, 5
    )
    efficientnet1_loaded = False
efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1244126236.py in <cell line: 0>()
----> 1 efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
      2 num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
      3 efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)
      4 efficientnet_model_1.classifier = nn.Sequential(
      5     nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)

NameError: name 'models' is not defined

## === cell 11
ensemble_predictions = []
image_names = []

with torch.no_grad():
    for batch in test_loader:
        image, img_name = batch[0]  # batch size = 1
        if image is None:
            continue

        if efficientnet_loaded:
            probs = tta_predict_single_model(
                efficientnet_model_7, image, tta_transform, device, n_tta=num_tta
            )
        else:
            probs = tta_predict_single_model(
                resnet_model, image, tta_transform, device, n_tta=num_tta
            )

        final_pred = probs.argmax(dim=1).cpu().item()
        ensemble_predictions.append(final_pred)
        image_names.append(img_name)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1868925365.py in <cell line: 0>()
      2 image_names = []
      3 
----> 4 with torch.no_grad():
      5     for batch in test_loader:
      6         image, img_name = batch[0]  # batch size = 1

NameError: name 'torch' is not defined

## === cell 12
submission_df = pd.DataFrame(
    {
        "image_id": image_names,
        "label": ensemble_predictions,
    }
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file saved as '{submission_path}'")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2719054269.py in <cell line: 0>()
----> 1 submission_df = pd.DataFrame(
      2     {
      3         "image_id": image_names,
      4         "label": ensemble_predictions,
      5     }

NameError: name 'pd' is not defined
