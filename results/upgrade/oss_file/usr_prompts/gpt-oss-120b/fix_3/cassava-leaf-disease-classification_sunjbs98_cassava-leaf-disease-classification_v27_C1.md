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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.8224539135690541

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.10762) has done: 'I fix the runtime errors that prevent a submission file from being created and keep the model logic unchanged.  
- Guard the pre‑trained weight loading with a file‑existence check and fall back to random initialization if the file is missing.  
- Correct the misuse of `torch.cuda.is_available` (add parentheses) when moving tensors/weights to GPU.  
- Add a short comment explaining each change so the intent (producing a valid CSV and moving the score toward the target) is clear.'

# 9. Code solution

## === cell 0
is_submission = False  # Train the model first; later we will generate predictions.
data_path = "/kaggle/input/cassava-leaf-disease-classification"
csv_file_name = "train.csv" if not is_submission else "sample_submission.csv"
train_batch_size, val_batch_size, sub_batch_size = 4, 8, 8
train_rate = 0.7  # rate for train dataset
num_workers = 2
mean, std = 0.5, 0.5

use_balanced_sample = False
use_class_weight = not use_balanced_sample
mul_weight = torch.tensor([3.0, 1.5, 1.0, 1.0, 4.5])
learning_rate = 0.001
weight_decay = 0.00002
efficient_net_version = 3  # (1.4, 1.2)

train_epoch = (0, 20)
debug = not is_submission

use_pre_trained_weight = True
pre_trained_weight_path = (
    "../input/cassava-leaf-disease-classification-weight/weight.pth"
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3309444360.py in <cell line: 0>()
      9 use_balanced_sample = False
     10 use_class_weight = not use_balanced_sample
---> 11 mul_weight = torch.tensor([3.0, 1.5, 1.0, 1.0, 4.5])
     12 learning_rate = 0.001
     13 weight_decay = 0.00002

NameError: name 'torch' is not defined

## === cell 1
if not is_submission and use_class_weight:

    def class_weight(label_count):
        total = sum(label_count)
        weight = [total / (2 * count) if count > 0 else 0.0 for count in label_count]
        return torch.tensor(weight, dtype=torch.float32)

    label_count = [0 for _ in range(5)]

    for i in range(len(dataset)):
        label = dataset.cassava_leaf_disease.label[i]
        label_count[label] += 1
    print(label_count)

    loss_weight = class_weight(label_count)
    loss_weight *= mul_weight
    if torch.cuda.is_available():
        loss_weight = loss_weight.cuda()
        print(loss_weight)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2844409719.py in <cell line: 0>()
      9     label_count = [0 for _ in range(5)]
     10 
---> 11     for i in range(len(dataset)):
     12         label = dataset.cassava_leaf_disease.label[i]
     13         label_count[label] += 1

NameError: name 'dataset' is not defined

## === cell 2
if not is_submission:
    if use_class_weight:
        criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
    elif mul_weight is not None:
        if torch.cuda.is_available():
            mul_weight = mul_weight.cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=mul_weight)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_accuracy = []
    train_losses = []
    val_accuracy = []
    best_accuracy = 0.0 if len(val_accuracy) == 0 else min(val_accuracy)
    best_model = deepcopy(model.state_dict())



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1858407083.py in <cell line: 0>()
      1 if not is_submission:
      2     if use_class_weight:
----> 3         criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
      4     elif mul_weight is not None:
      5         if torch.cuda.is_available():

NameError: name 'torch' is not defined

## === cell 3
model.eval()
if os.path.isfile("weight.pth"):
    best_state = torch.load("weight.pth", map_location="cpu")
    model.load_state_dict(best_state)

test_csv_path = os.path.join(data_path, "test.csv")
if not os.path.isfile(test_csv_path):
    test_images_dir = os.path.join(data_path, "test_images")
    test_filenames = sorted(
        [f for f in os.listdir(test_images_dir) if f.lower().endswith((".jpg", ".png"))]
    )
    test_df = pd.DataFrame({"image_id": test_filenames})
    test_df.to_csv("generated_test.csv", index=False)
    test_csv_path = "generated_test.csv"


class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "test_images", self.df.iloc[idx, 0])
        img = io.imread(img_name)
        if self.transform:
            img = self.transform(img)
        return img


test_dataset = TestDataset(
    csv_file=test_csv_path, root_dir=data_path, transform=transform
)
test_loader = DataLoader(
    test_dataset, batch_size=sub_batch_size, shuffle=False, num_workers=num_workers
)

predictions = []
with torch.no_grad():
    for inputs in test_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

submission = pd.DataFrame(
    {"image_id": pd.read_csv(test_csv_path)["image_id"], "label": predictions}
)
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created.")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3448422576.py in <cell line: 0>()
      1 # After training, generate predictions for the test set and create submission.csv
----> 2 model.eval()
      3 # load best weights saved during training
      4 if os.path.isfile("weight.pth"):
      5     best_state = torch.load("weight.pth", map_location="cpu")

NameError: name 'model' is not defined
