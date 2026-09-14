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

0.8381686310063463

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.1719) has done: 'Implemented fixes to correctly locate image files and generate a valid submission CSV. The script now skips directories, processes only image files (jpg/jpeg/png), and ensures the output matches the expected format and length, preventing earlier `IsADirectoryError` and submission size mismatches.'
- What this solution (achieved 0.25897) has done: 'The fix adds logic to automatically load a fine‑tuned checkpoint (if it exists) into the selected model, so that inference uses learned cassava‑leaf disease weights instead of raw ImageNet weights. This small change keeps the original architecture and inference flow unchanged but can raise accuracy dramatically, moving the score toward the target. If no checkpoint is found the script falls back to the original pretrained model, preserving functionality.'
- What this solution (achieved 0.21786) has done: 'I add a lightweight fine‑tuning step that runs only when the expected checkpoint is not found. After loading the backbone (ViT or EfficientNet), the script checks if a fine‑tuned weight file was loaded; if not, it quickly trains the model’s final linear head on the original training set for a couple of epochs (freezing the backbone). This modest training improves the classification accuracy from near‑random without changing the core architecture, keeping the original inference flow unchanged and still producing a valid `submission.csv`.'
- What this solution (achieved 0.47945) has done: 'Implemented robust handling for model head parameter selection to avoid attribute errors when the fallback `Sequential` model is used, and extended lightweight fine‑tuning to 5 epochs for better accuracy. The head is now detected generically (`heads.head`, `classifier[1]`, or the second module in a `Sequential`) before unfreezing its weights and bias.'

# 9. Code solution

## === cell 0
def invert_square_pad(img):
    width, height = img.size

    center_width, center_height = width // 2, height // 2
    top_left = img.crop((0, 0, center_width, center_height))
    top_right = img.crop((center_width, 0, width, center_height))
    bottom_left = img.crop((0, center_height, center_width, height))
    bottom_right = img.crop((center_width, center_height, width, height))

    top_combined = Image.new("RGB", (width, center_height))
    top_combined.paste(bottom_right, (0, 0))
    top_combined.paste(bottom_left, (center_width, 0))

    bottom_combined = Image.new("RGB", (width, center_height))
    bottom_combined.paste(top_right, (0, 0))
    bottom_combined.paste(top_left, (center_width, 0))

    flipped_img = Image.new("RGB", (width, height))
    flipped_img.paste(top_combined, (0, 0))
    flipped_img.paste(bottom_combined, (0, center_height))

    img = flipped_img.copy()

    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,  # left
        (max_side - height) // 2,  # top
        (max_side - width) - (max_side - width) // 2,  # right
        (max_side - height) - (max_side - height) // 2,  # bottom
    )

    padded_img = transforms.functional.pad(img, padding, padding_mode="reflect")
    return padded_img




## === cell 1
val_transforms = transforms.Compose(
    [
        transforms.Resize((model_image_size, model_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(model_image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4097881510.py in <cell line: 0>()
----> 1 val_transforms = transforms.Compose(
      2     [
      3         transforms.Resize((model_image_size, model_image_size)),
      4         transforms.ToTensor(),
      5         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

NameError: name 'transforms' is not defined

## === cell 2
checkpoint_loaded = False

if model_select == "vit":
    try:
        from torchvision import models as tv_models

        vit_model = tv_models.vit_h_14(
            weights=tv_models.ViT_H_14_Weights.IMAGENET1K_V1, image_size=518
        )
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        checkpoint_path = (
            "/kaggle/input/cassava-leaf-disease-classification/vit_finetuned.pth"
        )
        if not os.path.isfile(checkpoint_path):
            checkpoint_path = "/kaggle/working/vit_finetuned.pth"
        if os.path.isfile(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=device)
            vit_model.load_state_dict(state, strict=False)
            checkpoint_loaded = True
        vit_model.to(device)
        vit_model.eval()
    except Exception as e:
        vit_model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * model_image_size * model_image_size, num_classes),
        )
        vit_model.to(device)
        vit_model.eval()
elif model_select == "en":
    try:
        from torchvision import models as tv_models

        en_model = tv_models.efficientnet_v2_l(
            weights=tv_models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        )
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )
        checkpoint_path = (
            "/kaggle/input/cassava-leaf-disease-classification/en_finetuned.pth"
        )
        if not os.path.isfile(checkpoint_path):
            checkpoint_path = "/kaggle/working/en_finetuned.pth"
        if os.path.isfile(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=device)
            en_model.load_state_dict(state, strict=False)
            checkpoint_loaded = True
        en_model.to(device)
        en_model.eval()
    except Exception as e:
        en_model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * model_image_size * model_image_size, num_classes),
        )
        en_model.to(device)
        en_model.eval()
else:
    raise ValueError("Unsupported model_select value")

model = vit_model if model_select == "vit" else en_model




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/581915622.py in <cell line: 0>()
      1 checkpoint_loaded = False
      2 
----> 3 if model_select == "vit":
      4     try:
      5         from torchvision import models as tv_models

NameError: name 'model_select' is not defined

## === cell 3
if not checkpoint_loaded:
    print("Checkpoint not found – starting lightweight fine‑tuning on training data.")
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    if not os.path.isfile(train_csv_path):
        train_csv_path = "/kaggle/working/train.csv"
    train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    if not os.path.isdir(train_img_dir):
        train_img_dir = "/kaggle/working/train_images"

    class CassavaDataset(torch.utils.data.Dataset):
        def __init__(self, csv_path, img_dir, transform):
            self.df = pd.read_csv(csv_path)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["image_id"])
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            label = int(row["label"])
            return image, label

    train_dataset = CassavaDataset(train_csv_path, train_img_dir, train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
    )

    for name, param in model.named_parameters():
        param.requires_grad = False

    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        head = model.heads.head
    elif hasattr(model, "classifier") and isinstance(
        model.classifier, torch.nn.ModuleList
    ):
        head = model.classifier[1]
    elif isinstance(model, torch.nn.Sequential):
        head = model[1]
    else:
        raise RuntimeError("Unable to locate model head for fine‑tuning.")

    head.weight.requires_grad = True
    head.bias.requires_grad = True

    if model_select == "vit":
        for name, param in model.named_parameters():
            if "encoder.layers.11" in name:  # last transformer block
                param.requires_grad = True
    elif model_select == "en":
        for name, param in model.named_parameters():
            if "features.7" in name:  # approximate last stage identifier
                param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4
    )

    model.train()
    epochs = 12  # extended training for better convergence
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Fine‑tuning epoch {epoch+1}/{epochs} – loss: {epoch_loss:.4f}")

    model.eval()
    print("Lightweight fine‑tuning completed.")
else:
    print("Fine‑tuned checkpoint loaded – skipping additional training.")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1794035534.py in <cell line: 0>()
      2     print("Checkpoint not found – starting lightweight fine‑tuning on training data.")
      3     train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
----> 4     if not os.path.isfile(train_csv_path):
      5         train_csv_path = "/kaggle/working/train.csv"
      6     train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"

NameError: name 'os' is not defined

## === cell 4
supported_ext = (".jpg", ".jpeg", ".png")
image_paths = []
for root, _, files in os.walk(test_data_directory):
    for f in files:
        if f.lower().endswith(supported_ext):
            image_paths.append(os.path.join(root, f))

image_paths.sort()

predictions = []
image_ids = []

for image_path in tqdm(image_paths, desc="Test"):
    image_name = os.path.basename(image_path)
    image = Image.open(image_path).convert("RGB")
    transformed_image = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed_image)
        _, predicted_class = torch.max(output, 1)

    predictions.append(int(predicted_class.item()))
    image_ids.append(image_name)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3790733049.py in <cell line: 0>()
      1 supported_ext = (".jpg", ".jpeg", ".png")
      2 image_paths = []
----> 3 for root, _, files in os.walk(test_data_directory):
      4     for f in files:
      5         if f.lower().endswith(supported_ext):

NameError: name 'os' is not defined

## === cell 5
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_df = submission_df.sort_values("image_id").reset_index(drop=True)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/199389284.py in <cell line: 0>()
----> 1 submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
      2 submission_df = submission_df.sort_values("image_id").reset_index(drop=True)
      3 
      4 submission_path = "submission.csv"
      5 submission_df.to_csv(submission_path, index=False)

NameError: name 'pd' is not defined
