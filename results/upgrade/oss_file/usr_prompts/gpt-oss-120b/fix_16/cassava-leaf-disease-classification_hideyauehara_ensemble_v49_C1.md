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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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

0.8943789664551224

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.58558) has done: 'I make the test‑image path detection robust (search any “test_images” folder and fall back to a sensible default) and filter the files to real images so the DataLoader never crashes. Afterwards I keep the existing inference logic unchanged and ensure the final CSV contains only the required two columns with integer labels, guaranteeing a valid submission file.'
- What this solution (achieved 0.10762) has done: 'I correct the paths used to locate the pretrained model checkpoints so that the script actually loads the high‑performing models provided in the Kaggle input directory instead of falling back to a simple pretrained ResNet‑18. By fixing the glob patterns to point to “/kaggle/input/…”, the ensemble include the stronger pretrained checkpoints, which should raise the validation accuracy toward the target score while keeping the original model architecture and inference pipeline unchanged.'
- What this solution (achieved 0.58296) has done: 'Added missing imports (`numpy` and `glob`) to resolve NameError issues, enabling the script to locate model files and perform array operations. No other logic changes were made, preserving the original modeling and inference pipeline while ensuring a valid `submission.csv` is generated.'
- What this solution (achieved 0.58034) has done: 'I make the model‑loading step more robust by searching recursively for any *.pth checkpoint under /kaggle/input, instead of relying on hard‑coded folder names that may not exist. This allows the script to actually load the provided pretrained weights (e.g., DenseNet‑201 or EfficientNet‑B7) rather than falling back to a generic ResNet‑18, leading to a higher validation accuracy and moving the score closer to the target.'
- What this solution (achieved 0.56726) has done: 'I limit the test‑time augmentations to only the first two robust transforms (center‑crop and horizontal‑flip) so that noisy augmentations no longer drag down the averaged predictions. This small change keeps the core modeling untouched while expected to raise the validation accuracy toward the target.'
- What this solution (achieved 0.58483) has done: 'I keep the overall pipeline unchanged and only adjust the test‑time augmentation handling. The original code limits the TTA transforms to the first two, which reduces the diversity of predictions and hurts accuracy. By removing that slicing line we use the full set of five augmentations, giving a richer ensemble and moving the validation accuracy upward toward the target score. No other logic or model architecture is altered.'
- What this solution (achieved 0.56726) has done: 'I limit the test‑time augmentation to the first two robust transforms (center‑crop and horizontal‑flip) instead of using all five augmentations, which can introduce noisy predictions and lower accuracy. This small change keeps the core model and inference pipeline unchanged while expected to raise the validation accuracy toward the target score. The modification is applied in the loop that iterates over the transforms during inference.'
- What this solution (achieved 0.58109) has done: 'I broaden the test‑time augmentation by using all five defined transforms instead of only the first two. This modest change keeps the original model loading and inference pipeline intact while likely improving the averaged predictions, moving the validation accuracy closer to the target score.'
- What this solution (achieved 0.58221) has done: 'I switch the torchvision models to load their ImageNet‑pretrained weights (still using the same architecture and checkpoint loading logic). This small change keeps the core pipeline unchanged while giving the networks a better starting point, which should raise the validation accuracy toward the target.'
- What this solution (achieved 0.05531) has done: 'I broaden the search for pretrained *.pth checkpoints to include both the usual /kaggle/input directory **and** the working directory, guaranteeing that the script actually finds and loads the provided models instead of falling back to a single ResNet‑18. I also simplify the checkpoint‑loading logic so every supported model (including EfficientNet‑B7) loads its state dict directly, avoiding the previous conditional that could skip loading. These minimal adjustments keep the original pipeline intact while enabling the stronger pretrained ensembles, which should move the validation accuracy much closer to the target score.'

# 9. Code solution

## === cell 0
import sys

try:
    sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
    from efficientnet_pytorch import EfficientNet

    EFFICIENTNET_AVAILABLE = True
except Exception:
    EFFICIENTNET_AVAILABLE = False




## === cell 1
SIZE = 512  # image size
num_classes = 5




## === cell 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")


def find_test_path() -> str:
    candidates = list(Path(".").rglob("test_images"))
    for cand in candidates:
        if cand.is_dir():
            return str(cand.resolve())
    fallback = Path("/kaggle/input/cassava-leaf-disease-classification/test_images")
    if fallback.is_dir():
        return str(fallback)
    raise FileNotFoundError("Test images directory not found.")


TEST_PATH = find_test_path()
test_files = [f for f in os.listdir(TEST_PATH) if f.lower().endswith(".jpg")]
print(f"Number of test images found: {len(test_files)}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3946226664.py in <cell line: 0>()
----> 1 device = "cuda" if torch.cuda.is_available() else "cpu"
      2 print(f"使用デバイス: {device}")
      3 
      4 
      5 def find_test_path() -> str:

NameError: name 'torch' is not defined

## === cell 3
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1  # placeholder, will be overwritten later




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/123932543.py in <cell line: 0>()
----> 1 df_test = pd.DataFrame(test_files, columns=["image_id"])
      2 df_test["label"] = 1  # placeholder, will be overwritten later
      3 
      4 

NameError: name 'pd' is not defined

## === cell 4
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2383576632.py in <cell line: 0>()
----> 1 if len(df_test) == 1:
      2     df_test.loc[1] = df_test.loc[0]
      3     print(df_test)
      4 
      5 

NameError: name 'df_test' is not defined

## === cell 5
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2172022482.py in <cell line: 0>()
      4 transform = {
      5     "test": [
----> 6         Compose(
      7             [
      8                 A.CenterCrop(height=SIZE, width=SIZE),

NameError: name 'Compose' is not defined

## === cell 6
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)
        labels_a = labels
        labels_b = labels[index]
        loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(
            outputs, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3608567986.py in <cell line: 0>()
----> 1 class FinalLayerMixupModel(nn.Module):
      2     def __init__(self, model, criterion, num_classes, alpha):
      3         super(FinalLayerMixupModel, self).__init__()
      4         self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
      5         num_ftrs = model.fc.in_features

NameError: name 'nn' is not defined

## === cell 7
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])
        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)
        labels_a = labels
        labels_b = labels[index]
        loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(
            outputs, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1452739864.py in <cell line: 0>()
----> 1 class FinalLayerMixupModelDenseNet(nn.Module):
      2     def __init__(self, model, criterion, num_classes, alpha):
      3         super(FinalLayerMixupModelDenseNet, self).__init__()
      4         self.convlayer = model.features
      5         self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

NameError: name 'nn' is not defined

## === cell 8
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)
        self.model = model
        self.criterion = criterion

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self.model(inputs)
            return outputs

        print("ここにきてはいけない")
        sys.exit()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1312263886.py in <cell line: 0>()
----> 1 class FinalLayerMixupModelEN(nn.Module):
      2     def __init__(self, model, criterion, num_classes, alpha):
      3         super(FinalLayerMixupModelEN, self).__init__()
      4         num_ftrs = model._fc.in_features
      5         model._fc = nn.Linear(num_ftrs, num_classes)

NameError: name 'nn' is not defined

## === cell 9
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = f"{TEST_PATH}/{image_id}"
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1678751526.py in <cell line: 0>()
----> 1 class TestDataset(data.Dataset):
      2     def __init__(self, df, transform=None):
      3         super().__init__()
      4         self.image_ids = df.image_id.tolist()
      5         self.transform = transform

NameError: name 'data' is not defined

## === cell 10
def predict_model(basename, net, dataloader):
    model_start_time = time.time()
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []
    for inputs, image_ids in tqdm(dataloader["test"], desc=f"{basename}: "):
        inputs = inputs.to(device)
        outputs = net(inputs, False, "test")
        probability.append(torch.softmax(outputs, dim=1).cpu().numpy())
    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability)




## === cell 11
INPUT_ROOT = "/kaggle/input"
pretrained_models = glob.glob(os.path.join(INPUT_ROOT, "**", "*.pth"), recursive=True)

if not pretrained_models:
    pretrained_models = glob.glob(
        os.path.join("/kaggle/working", "**", "*.pth"), recursive=True
    )

print(f"{len(pretrained_models)} models found.")
print("\n".join(np.sort(pretrained_models)))




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3871553160.py in <cell line: 0>()
      1 INPUT_ROOT = "/kaggle/input"
----> 2 pretrained_models = glob.glob(os.path.join(INPUT_ROOT, "**", "*.pth"), recursive=True)
      3 
      4 if not pretrained_models:
      5     pretrained_models = glob.glob(

NameError: name 'glob' is not defined

## === cell 12
if not pretrained_models:
    print(
        "No pretrained .pth models found. Falling back to torchvision's pretrained ResNet18."
    )
    pretrained_models = ["fallback_resnet18"]

probability = []
start_time = time.time()

for pretrained_model in pretrained_models:
    basename = os.path.splitext(os.path.basename(pretrained_model))[0]
    criterion = nn.CrossEntropyLoss()

    if "fallback_resnet18" == basename:
        MODEL_NAME = "resnet18_fallback"
        net = models.resnet18(pretrained=True)  # use ImageNet pretrained weights
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 64
    elif "resnet18" in basename:
        MODEL_NAME = "resnet18"
        net = models.resnet18(pretrained=True)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 64
    elif "resnet50" in basename:
        MODEL_NAME = "resnet50"
        net = models.resnet50(pretrained=True)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 32
    elif "resnet152" in basename:
        MODEL_NAME = "resnet152"
        net = models.resnet152(pretrained=True)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 16
    elif "resnext101" in basename:
        MODEL_NAME = "resnext101"
        net = models.resnext101_32x8d(pretrained=True)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 12
    elif "densenet201" in basename:
        MODEL_NAME = "densenet201"
        net = models.densenet201(pretrained=True)
        net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
        BATCH_SIZE = 12
    elif "efficientnet-b7" in basename and EFFICIENTNET_AVAILABLE:
        MODEL_NAME = "efficientnet-b7"
        net = EfficientNet.from_name(MODEL_NAME)
        net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
        BATCH_SIZE = 10
    else:
        print(f"{basename} is not supported or EfficientNet not available.")
        continue

    print(f"{basename}: {MODEL_NAME}")

    try:
        net.load_state_dict(torch.load(pretrained_model, map_location=device))
    except Exception as e:
        print(f"Failed to load state dict for {basename}: {e}")
        continue

    for param in net.parameters():
        param.requires_grad = False

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        dataset = TestDataset(df_test, transform=transform_)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        proba = predict_model(basename, net, {"test": dataloader})
        probability.append(proba)

    del net
    torch.cuda.empty_cache()

if probability:
    probs_array = np.array(
        probability
    )  # (n_transforms * n_models, n_samples, n_classes)
    mean_probs = probs_array.mean(axis=0)  # (n_samples, n_classes)
    df_test["mean"] = mean_probs.argmax(axis=1)
else:
    possible_paths = [
        "./train.csv",
        "/kaggle/input/cassava-leaf-disease-classification/train.csv",
        "/kaggle/input/train.csv",
    ]
    train_path = None
    for p in possible_paths:
        if os.path.exists(p):
            train_path = p
            break
    if train_path is not None:
        train_df = pd.read_csv(train_path)
        most_common_label = int(train_df["label"].mode()[0])
        print(f"Using most common label {most_common_label} as fallback prediction.")
        df_test["mean"] = most_common_label
    else:
        print("Training CSV not found; defaulting to label 0.")
        df_test["mean"] = 0

print(f"total time: {time.time() - start_time:.2f}[sec]")




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3077026782.py in <cell line: 0>()
----> 1 if not pretrained_models:
      2     print(
      3         "No pretrained .pth models found. Falling back to torchvision's pretrained ResNet18."
      4     )
      5     pretrained_models = ["fallback_resnet18"]

NameError: name 'pretrained_models' is not defined

## === cell 13
df_test["label"] = df_test["mean"].astype(int)
df_test = df_test[["image_id", "label"]]




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1470096187.py in <cell line: 0>()
----> 1 df_test["label"] = df_test["mean"].astype(int)
      2 df_test = df_test[["image_id", "label"]]
      3 
      4 

NameError: name 'df_test' is not defined

## === cell 14
df_test.head()




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/533082530.py in <cell line: 0>()
----> 1 df_test.head()
      2 
      3 

NameError: name 'df_test' is not defined

## === cell 15
df_test.to_csv("submission.csv", index=False)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3957235323.py in <cell line: 0>()
----> 1 df_test.to_csv("submission.csv", index=False)

NameError: name 'df_test' is not defined
