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

0.8981565427621638

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59043) has done: 'I fix the two runtime errors: replace the unavailable `A.Cutout` with a supported augmentation ( `A.GridDropout` ) and ensure the test augmentation pipeline is defined, so inference runs and a proper `submission.csv` is written.'
- What this solution (achieved 0.49738) has done: 'I keep the overall model loading and inference pipeline unchanged but improve the prediction post‑processing, which is the main factor affecting the validation accuracy.  
Instead of L2‑normalising raw logits, I convert them to proper class probabilities with a soft‑max, then average the two model’s probabilities using the same ensemble weights. This aligns the scoring with the classification‑accuracy metric and should raise the score toward the target. I also increase TTA slightly (from 8 to 12) for the EfficientNet model to get a more stable prediction without altering the core logic.'
- What this solution (achieved 0.52093) has done: 'The changes batch the TTA augmentations, read each image only once, and perform a single forward pass per model for the whole TTA batch. This removes the inner Python loop and dramatically cuts overhead while preserving the exact same augmentations, model architecture, and averaging logic, so the predictions remain unchanged. The rest of the pipeline is left intact.'
- What this solution (achieved 0.53513) has done: 'The change makes the test‑time augmentation deterministic (same as the validation pipeline) and reduces the number of TTA copies, which aligns predictions with the accuracy metric and should raise the validation score toward the target while keeping the core model logic unchanged.'
- What this solution (achieved 0.52728) has done: 'I increase the test‑time augmentation to be truly stochastic (so TTA actually diversifies predictions) and raise the number of TTA copies from 4 to 8. This keeps the model architecture and training logic untouched, but provides richer averaged predictions that better match the validation‑style augmentations, which should raise the accuracy toward the target score.'

# 9. Code solution

## === cell 0
INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 32  # batch size for validation / inference
IMAGE_SIZE = 512
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 8  # reduced TTA to fit GPU memory
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3503083319.py in <cell line: 0>()
     17 SEED = 42
     18 
---> 19 DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NameError: name 'torch' is not defined

## === cell 1
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=True)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)


def load_checkpoint(model, ckpt_path):
    full_path = os.path.join(INPUT_PATH, ckpt_path)
    if os.path.isfile(full_path):
        ckpt = torch.load(full_path, map_location=DEVICE)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        state_dict = {
            k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()
        }
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Checkpoint not found, using pretrained model: {ckpt_path}")


load_checkpoint(my_model_1, RESNEXT_PATH)
load_checkpoint(my_model_2, B4_PATH)

if torch.cuda.device_count() > 0:
    my_model_1 = nn.DataParallel(my_model_1).to(DEVICE)
    my_model_2 = nn.DataParallel(my_model_2).to(DEVICE)
else:
    my_model_1 = my_model_1.to(DEVICE)
    my_model_2 = my_model_2.to(DEVICE)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3349963394.py in <cell line: 0>()
      1 model_name1 = "resnext50_32x4d"
----> 2 my_model_1 = timm.create_model(model_name1, pretrained=True)
      3 my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
      4 nn.init.xavier_uniform_(my_model_1.fc.weight)
      5 if my_model_1.fc.bias is not None:

NameError: name 'timm' is not defined

## === cell 2
class SimpleImageDataset(Dataset):
    def __init__(self, df, img_dir, transforms):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        aug = self.transforms(image=img_np)
        return aug["image"], row["label"]


train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_df = train_df[:val_size]

val_dataset = SimpleImageDataset(val_df, TRAIN_IMAGE_PATH, valid_augs)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

my_model_1.eval()
my_model_2.eval()

torch.cuda.empty_cache()


def compute_accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            correct += (preds == labels.numpy()).sum()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


acc1 = compute_accuracy(my_model_1, val_loader)
acc2 = compute_accuracy(my_model_2, val_loader)

weight_sum = acc1 + acc2 if (acc1 + acc2) > 0 else 2.0
w1 = acc1 / weight_sum
w2 = acc2 / weight_sum
print(f"Validation accuracies -> Model1: {acc1:.4f}, Model2: {acc2:.4f}")
print(f"Ensemble weights -> w1: {w1:.4f}, w2: {w2:.4f}")


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1255729263.py in <cell line: 0>()
----> 1 class SimpleImageDataset(Dataset):
      2     def __init__(self, df, img_dir, transforms):
      3         self.df = df.reset_index(drop=True)
      4         self.img_dir = img_dir
      5         self.transforms = transforms

NameError: name 'Dataset' is not defined

## === cell 3
test_image_names = sorted(
    [f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(".jpg")]
)


class TestFilenameDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.filenames[idx]


test_dataset = TestFilenameDataset(test_image_names)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda x: x,
)


def tta_inference(model, loader, description):
    preds = []
    for batch_names in tqdm(loader, desc=description):
        aug_tensors = []
        for img_name in batch_names:
            img_path = os.path.join(TEST_IMAGE_PATH, img_name)
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            aug_imgs = [test_augs(image=img_np)["image"] for _ in range(TTA)]
            aug_tensors.extend(aug_imgs)
        batch_tensor = torch.stack(aug_tensors).to(DEVICE)
        with torch.no_grad():
            out = model(batch_tensor)  # (B*TTA, OUT_FEATURES)
        out = out.view(len(batch_names), TTA, -1).mean(dim=1)  # (B, OUT_FEATURES)
        preds.append(out.cpu())
    return torch.cat(preds, dim=0)  # (N, OUT_FEATURES)


preds_1 = tta_inference(my_model_1, test_loader, "Model1 inference (TTA)")
preds_2 = tta_inference(my_model_2, test_loader, "Model2 inference (TTA)")

prob_1 = F.softmax(preds_1, dim=1)
prob_2 = F.softmax(preds_2, dim=1)

final_prob = prob_1 * w1 + prob_2 * w2
labels = final_prob.argmax(dim=1).numpy()

submission_df = pd.DataFrame({"image_id": test_image_names, "label": labels})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3877478011.py in <cell line: 0>()
      1 test_image_names = sorted(
----> 2     [f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(".jpg")]
      3 )
      4 
      5 

NameError: name 'os' is not defined
