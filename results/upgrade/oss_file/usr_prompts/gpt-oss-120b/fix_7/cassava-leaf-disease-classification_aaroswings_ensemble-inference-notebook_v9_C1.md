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

albumentations==2.0.8
geopandas==0.14.4
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

0.8750377757630704

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The fix filters out sub‑directories when building the test dataset, sorts the filenames for deterministic ordering, and ensures the DataLoader is created only after the dataset is successfully built. This resolves the `IsADirectoryError`, defines `test_loader` correctly, and allows the submission CSV to be written with matching lengths.'
- What this solution (achieved 0.05531) has done: 'I adjust the loading of the pretrained checkpoints so that they are applied to a proper ResNeXt‑101 architecture (instead of falling back to the dummy model) and fix the image normalization to use ImageNet statistics. These minimal changes keep the overall pipeline identical while allowing the ensemble to produce meaningful predictions, moving the validation accuracy much closer to the target score.'
- What this solution (achieved 0.05531) has done: 'I adjust the preprocessing size to match the ResNeXt‑101 expected input (224 × 224) and make the checkpoint loading more robust: if the loaded object is a dict containing a `model` or `state_dict` entry we extract the actual weights; otherwise we fall back to the dummy model. These minimal changes keep the overall architecture and inference pipeline unchanged while allowing the pretrained weights to be applied correctly, which should raise the validation accuracy substantially toward the target score.'
- What this solution (achieved 0.05531) has done: 'I enable ImageNet‑pretrained weights for the ResNeXt‑101 models (changing `pretrained=False` to `pretrained=True`). This keeps the architecture unchanged but gives the models meaningful feature extractors, which should raise the validation accuracy substantially toward the target score. The rest of the pipeline remains identical.'

# 9. Code solution

## === cell 0
batch_size = 32
valid_input_size = 224
test_img_path = "../input/cassava-leaf-disease-classification/test_images"



## === cell 1
import torchvision.models as tv_models

model_fnames = [
    "../input/cassava-notebook-15-resnext-models/resnext101wsl_epoch_6.pickle",
    "../input/cassava-notebook-16-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-19-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-20-models/resnext101wsl_epoch_6.pickle",
]

models = []
for fname in model_fnames:
    try:
        ckpt = torch.load(fname, map_location=device)

        if isinstance(ckpt, nn.Module):
            m = ckpt.to(device).eval()
        else:
            if isinstance(ckpt, dict):
                if "model" in ckpt:  # some notebooks save under 'model'
                    state_dict = ckpt["model"]
                elif "state_dict" in ckpt:  # standard PyTorch checkpoint format
                    state_dict = ckpt["state_dict"]
                else:
                    state_dict = ckpt  # fallback: treat dict as state_dict
            else:
                state_dict = None

            base_model = tv_models.resnext101_32x8d(pretrained=False)
            base_model.fc = nn.Linear(base_model.fc.in_features, 5)

            if state_dict is not None:
                base_model.load_state_dict(state_dict, strict=False)

            m = base_model.to(device).eval()
    except Exception:
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                batch_sz = x.shape[0]
                return torch.zeros(batch_sz, 5, device=x.device)

        m = DummyModel().to(device).eval()
    models.append(m)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2223254314.py in <cell line: 0>()
     12     try:
---> 13         ckpt = torch.load(fname, map_location=device)
     14 

NameError: name 'torch' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2223254314.py in <cell line: 0>()
     38     except Exception:
     39         # Fallback dummy model that returns zero logits
---> 40         class DummyModel(nn.Module):
     41             def __init__(self):
     42                 super().__init__()

NameError: name 'nn' is not defined

## === cell 2
test_tfms = Compose(
    [
        Resize(valid_input_size, valid_input_size, always_apply=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


class TestDataset(Dataset):
    def __init__(self, path, tfms):
        super(TestDataset, self).__init__()
        files = [f for f in Path(path).iterdir() if f.is_file()]
        files.sort(key=lambda x: x.name)
        self.data = [(file.name, open(file, "rb").read()) for file in files]
        self.tfms = tfms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, img_bytes = self.data[idx]
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        img = self.tfms(image=img)["image"]
        return filename, img


test_ds = TestDataset(test_img_path, test_tfms)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0, drop_last=False)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/959513478.py in <cell line: 0>()
----> 1 test_tfms = Compose(
      2     [
      3         Resize(valid_input_size, valid_input_size, always_apply=True),
      4         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
      5         ToTensorV2(),

NameError: name 'Compose' is not defined

## === cell 3
class EnsemblePredictor:
    def __init__(self, models):
        super(EnsemblePredictor, self).__init__()
        self.models = models

    def predict_on_loader(self, loader):
        predictions = []
        filenames = []

        for file_batch, img_batch in loader:
            img_batch = img_batch.to(device)
            batch_pred = None
            for model in self.models:
                with torch.no_grad():
                    out = model(img_batch).cpu().numpy()
                if batch_pred is None:
                    batch_pred = out
                else:
                    batch_pred += out
            batch_preds = [int(np.argmax(x)) for x in batch_pred]
            predictions.extend(batch_preds)
            filenames.extend(list(file_batch))

        return predictions, filenames




## === cell 4
predictor = EnsemblePredictor(models)
predictions, filenames = predictor.predict_on_loader(test_loader)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/774291346.py in <cell line: 0>()
      1 predictor = EnsemblePredictor(models)
----> 2 predictions, filenames = predictor.predict_on_loader(test_loader)
      3 

NameError: name 'test_loader' is not defined

## === cell 5
with open("submission.csv", "w") as submission:
    submission.write("image_id,label\n")
    for filename, pred in zip(filenames, predictions):
        submission.write(f"{filename},{pred}\n")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3128773338.py in <cell line: 0>()
      1 with open("submission.csv", "w") as submission:
      2     submission.write("image_id,label\n")
----> 3     for filename, pred in zip(filenames, predictions):
      4         submission.write(f"{filename},{pred}\n")

NameError: name 'filenames' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
