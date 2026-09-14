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

3.12

# 3. Installed packages

albumentations==2.0.8
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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

0.898458748866727

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.50262) has done: 'I fixed the Albumentations augmentation definitions that were causing validation errors (the `RandomResizedCrop` API changed). I replaced it with `RandomCrop`, which matches the current library schema, so both training and test pipelines compile. This also restores the `test_augs` variable, allowing the inference loop to run and generate a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"
DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [
    torch.device("cpu")
]
OUT_FEATURES = 5
NUM_EPOCHS = 5  # keep original number of epochs
BATCH_SIZE = 64  # larger batch to reduce iteration overhead
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 8



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2904644595.py in <cell line: 0>()
      6 RESNEXT_PATH = "1022_res50.pth"
      7 B4_PATH = "1022_b4ns.pth"
----> 8 DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [
      9     torch.device("cpu")
     10 ]

NameError: name 'torch' is not defined

## === cell 1
train_dataset = CassavaDataset(TRAIN_CSV_PATH, TRAIN_IMAGE_PATH, train_augs)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,  # more workers for parallel loading
    pin_memory=True,
)

my_model_1 = load_or_pretrained(
    my_model_1, os.path.join(INPUT_PATH, RESNEXT_PATH), model_name1
)
my_model_2 = load_or_pretrained(
    my_model_2, os.path.join(INPUT_PATH, B4_PATH), model_name2
)

my_model_1 = nn.DataParallel(my_model_1).to(DEVICES[0])
my_model_2 = nn.DataParallel(my_model_2).to(DEVICES[0])

criterion = nn.CrossEntropyLoss()
optimizer_2 = OPTIMIZER(my_model_2.parameters(), lr=LR_MAX)

scaler = torch.cuda.amp.GradScaler()

my_model_2.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        imgs = imgs.to(DEVICES[0])
        labels = labels.to(DEVICES[0])
        optimizer_2.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = my_model_2(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer_2)
        scaler.update()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} – loss: {epoch_loss/len(train_loader):.4f}")

my_model_1.eval()
my_model_2.eval()
torch.cuda.empty_cache()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1705994107.py in <cell line: 0>()
      1 # increase workers for faster I/O and use mixed‑precision training
----> 2 train_dataset = CassavaDataset(TRAIN_CSV_PATH, TRAIN_IMAGE_PATH, train_augs)
      3 train_loader = DataLoader(
      4     train_dataset,
      5     batch_size=BATCH_SIZE,

NameError: name 'CassavaDataset' is not defined

## === cell 2
preds_1 = []
with torch.no_grad():
    for batch_names in tqdm(test_loader, desc="Model1 inference"):
        imgs = []
        for name in batch_names:
            img = Image.open(os.path.join(TEST_IMAGE_PATH, name)).convert("RGB")
            aug = test_augs(image=np.array(img))["image"]
            imgs.append(aug)
        batch_tensor = torch.stack(imgs).to(DEVICES[0])
        out = my_model_1(batch_tensor)
        preds_1.append(out.cpu())
predictions_1 = torch.cat(preds_1, dim=0)
prob_pred_1 = F.softmax(predictions_1, dim=1)

accum_predictions = torch.zeros(len(test_dataset), OUT_FEATURES)

for tta_step in range(TTA):
    preds_step = []
    with torch.no_grad():
        for batch_names in tqdm(
            test_loader, desc=f"Model2 TTA {tta_step+1}/{TTA}", leave=False
        ):
            raw_imgs = [
                np.array(Image.open(os.path.join(TEST_IMAGE_PATH, name)).convert("RGB"))
                for name in batch_names
            ]
            aug_imgs = [test_augs(image=raw)["image"] for raw in raw_imgs]
            batch_tensor = torch.stack(aug_imgs).to(DEVICES[0])
            out = my_model_2(batch_tensor)
            preds_step.append(out.cpu())
    step_tensor = torch.cat(preds_step, dim=0)
    accum_predictions += step_tensor

prob_pred_2 = F.softmax(accum_predictions / TTA, dim=1)

final_pred = (prob_pred_1 * 0.44) + (prob_pred_2 * 0.56)
label = final_pred.argmax(dim=1).numpy()
df_submission = pd.DataFrame({"image_id": test_dataset.file_names, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3185439568.py in <cell line: 0>()
      1 # Optimize test inference: read each image once per batch and reuse the raw array for all TTA steps
      2 preds_1 = []
----> 3 with torch.no_grad():
      4     for batch_names in tqdm(test_loader, desc="Model1 inference"):
      5         imgs = []

NameError: name 'torch' is not defined
