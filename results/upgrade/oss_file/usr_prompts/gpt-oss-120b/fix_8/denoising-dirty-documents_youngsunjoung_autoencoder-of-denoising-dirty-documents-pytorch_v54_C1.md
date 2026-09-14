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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.13

# 3. Installed packages

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.45275

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
def maybe_unzip(zip_path, extract_to):
    if not os.path.isdir(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        try:
            subprocess.check_call(["unzip", "-q", zip_path, "-d", extract_to])
        except Exception:
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)


base_extract_dir = "/kaggle/working/denoising_data"

maybe_unzip("/kaggle/input/denoising-dirty-documents/train.zip", base_extract_dir)
maybe_unzip("/kaggle/input/denoising-dirty-documents/test.zip", base_extract_dir)
maybe_unzip(
    "/kaggle/input/denoising-dirty-documents/train_cleaned.zip",
    base_extract_dir,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3948559105.py in <cell line: 0>()
     13 base_extract_dir = "/kaggle/working/denoising_data"
     14 
---> 15 maybe_unzip("/kaggle/input/denoising-dirty-documents/train.zip", base_extract_dir)
     16 maybe_unzip("/kaggle/input/denoising-dirty-documents/test.zip", base_extract_dir)
     17 maybe_unzip(

/tmp/ipykernel_55/3948559105.py in maybe_unzip(zip_path, extract_to)
      1 def maybe_unzip(zip_path, extract_to):
----> 2     if not os.path.isdir(extract_to):
      3         os.makedirs(extract_to, exist_ok=True)
      4         try:
      5             subprocess.check_call(["unzip", "-q", zip_path, "-d", extract_to])

NameError: name 'os' is not defined

## === cell 1
search_root = base_extract_dir


def find_dir(name_substr):
    matches = []
    for root, dirs, _ in os.walk(search_root):
        for d in dirs:
            if name_substr.lower() in d.lower():
                full_path = os.path.join(root, d)
                if any(f.lower().endswith(".png") for f in os.listdir(full_path)):
                    matches.append(full_path)
    if not matches:
        raise FileNotFoundError(
            f"Unable to locate directory containing '{name_substr}'"
        )
    return sorted(matches, key=lambda x: len(x.split(os.sep)))[0]


train_dir = find_dir("train")
train_cleaned_dir = find_dir("train_cleaned")
test_dir = find_dir("test")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3035235886.py in <cell line: 0>()
     20 
     21 
---> 22 train_dir = find_dir("train")
     23 train_cleaned_dir = find_dir("train_cleaned")
     24 test_dir = find_dir("test")

/tmp/ipykernel_55/3035235886.py in find_dir(name_substr)
      5 def find_dir(name_substr):
      6     matches = []
----> 7     for root, dirs, _ in os.walk(search_root):
      8         for d in dirs:
      9             if name_substr.lower() in d.lower():

NameError: name 'os' is not defined

## === cell 2
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images




## === cell 3
train_images = load_images_from_folder(train_dir)
train_cleaned_images = load_images_from_folder(train_cleaned_dir)
test_images = load_images_from_folder(test_dir)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/148537737.py in <cell line: 0>()
----> 1 train_images = load_images_from_folder(train_dir)
      2 train_cleaned_images = load_images_from_folder(train_cleaned_dir)
      3 test_images = load_images_from_folder(test_dir)
      4 

NameError: name 'train_dir' is not defined

## === cell 4
print(f"train_images: {len(train_images)}")
print(f"train_cleaned_images: {len(train_cleaned_images)}")
print(f"test_images: {len(test_images)}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1788669715.py in <cell line: 0>()
----> 1 print(f"train_images: {len(train_images)}")
      2 print(f"train_cleaned_images: {len(train_cleaned_images)}")
      3 print(f"test_images: {len(test_images)}")
      4 
      5 

NameError: name 'train_images' is not defined

## === cell 5
class PadToSize:
    def __init__(self, target_size):
        self.target_size = target_size  # (height, width)

    def __call__(self, img):
        _, height, width = img.shape
        target_height, target_width = self.target_size
        pad_top = (target_height - height) // 2
        pad_bottom = target_height - height - pad_top
        pad_left = (target_width - width) // 2
        pad_right = target_width - width - pad_left
        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)




## === cell 6
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        grayscale_img = pil_img.convert("L")
        return TF.to_tensor(grayscale_img)




## === cell 7
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),
        v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

val_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

test_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2346121324.py in <cell line: 0>()
----> 1 train_transforms = v2.Compose(
      2     [
      3         v2.ToImage(),
      4         PadToSize((420, 540)),
      5         v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),

NameError: name 'v2' is not defined

## === cell 8
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".png")
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3805309422.py in <cell line: 0>()
----> 1 class ImageDataset(Dataset):
      2     def __init__(self, data_dir, transform=None):
      3         self.data_dir = data_dir
      4         self.transform = transform
      5         self.image_files = sorted(

NameError: name 'Dataset' is not defined

## === cell 9
train_files = sorted(
    [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".png")]
)
cleaned_files = sorted(
    [
        os.path.join(train_cleaned_dir, f)
        for f in os.listdir(train_cleaned_dir)
        if f.endswith(".png")
    ]
)

train_files, val_files, cleaned_train, cleaned_val = train_test_split(
    train_files, cleaned_files, test_size=2 / 9, random_state=42
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/508094791.py in <cell line: 0>()
      1 train_files = sorted(
----> 2     [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".png")]
      3 )
      4 cleaned_files = sorted(
      5     [

NameError: name 'os' is not defined

## === cell 10
class PairedImageDataset(Dataset):
    def __init__(self, train_files, cleaned_files, transform=None):
        self.train_files = train_files
        self.cleaned_files = cleaned_files
        self.transform = transform

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, idx):
        train_img = Image.open(self.train_files[idx]).convert("RGB")
        cleaned_img = Image.open(self.cleaned_files[idx]).convert("RGB")
        if self.transform:
            train_img = self.transform(train_img)
            cleaned_img = self.transform(cleaned_img)
        return train_img, cleaned_img




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2476192371.py in <cell line: 0>()
----> 1 class PairedImageDataset(Dataset):
      2     def __init__(self, train_files, cleaned_files, transform=None):
      3         self.train_files = train_files
      4         self.cleaned_files = cleaned_files
      5         self.transform = transform

NameError: name 'Dataset' is not defined

## === cell 11
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1443139496.py in <cell line: 0>()
----> 1 train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
      2 val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
      3 test_dataset = ImageDataset(test_dir, test_transforms)
      4 
      5 train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

NameError: name 'PairedImageDataset' is not defined

## === cell 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3093293998.py in <cell line: 0>()
----> 1 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      2 print("Using device:", device)
      3 
      4 

NameError: name 'torch' is not defined

## === cell 13
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(1, 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(8, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(
                16,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=16,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.ConvTranspose2d(16, 8, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(
                8,
                8,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=8,
                bias=False,
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.ConvTranspose2d(8, 1, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(
                1, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        resized = F.interpolate(
            decoded, size=(420, 540), mode="bilinear", align_corners=False
        )
        outputs = torch.clamp(resized + x, min=0.001, max=0.999)
        return outputs


model = DenoisingAutoencoder().to(device)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1331497623.py in <cell line: 0>()
----> 1 class DenoisingAutoencoder(nn.Module):
      2     def __init__(self):
      3         super(DenoisingAutoencoder, self).__init__()
      4         self.encoder = nn.Sequential(
      5             nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),

NameError: name 'nn' is not defined

## === cell 14
try:
    from torchinfo import summary
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])
    from torchinfo import summary

summary(model, input_size=(16, 1, 420, 540), device=device)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4068536208.py in <cell line: 0>()
      5     from torchinfo import summary
      6 
----> 7 summary(model, input_size=(16, 1, 420, 540), device=device)
      8 
      9 

NameError: name 'model' is not defined

## === cell 15
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3903342212.py in <cell line: 0>()
----> 1 class RMSELoss(nn.Module):
      2     def forward(self, pred, target):
      3         return torch.sqrt(F.mse_loss(pred, target))
      4 
      5 

NameError: name 'nn' is not defined

## === cell 16
class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=0.8, lambda_l1=0.2):
        super().__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse = RMSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse(pred, target) + self.lambda_l1 * self.l1(
            pred, target
        )


criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10, eta_min=1e-5
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1499497582.py in <cell line: 0>()
----> 1 class HybridLoss(nn.Module):
      2     def __init__(self, lambda_rmse=0.8, lambda_l1=0.2):
      3         super().__init__()
      4         self.lambda_rmse = lambda_rmse
      5         self.lambda_l1 = lambda_l1

NameError: name 'nn' is not defined

## === cell 17
epochs = 20  # reduced for quick execution
patience = 5
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

epoch = 0
while epoch < epochs:
    model.train()
    train_loss = 0.0
    train_rmse = 0.0
    for train_imgs, clean_imgs in train_loader:
        train_imgs = train_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        optimizer.zero_grad()
        outputs = model(train_imgs)
        loss = criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_rmse += RMSELoss()(outputs, clean_imgs).item()
    train_loss /= len(train_loader)
    train_rmse /= len(train_loader)
    print(
        f"Epoch [{epoch+1}/{epochs}] Train loss: {train_loss:.4f} RMSE: {train_rmse:.4f}"
    )

    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    with torch.no_grad():
        for val_imgs, val_clean in val_loader:
            val_imgs = val_imgs.to(device)
            val_clean = val_clean.to(device)
            outputs = model(val_imgs)
            loss = criterion(outputs, val_clean)
            val_loss += loss.item()
            val_rmse += RMSELoss()(outputs, val_clean).item()
    val_loss /= len(val_loader)
    val_rmse /= len(val_loader)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print(f"  New best val loss: {val_loss:.4f} (RMSE {val_rmse:.4f})")
    else:
        early_stop_counter += 1
        print(f"  No improvement. Early stop cnt: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered")
        break
    epoch += 1

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print("Best model saved.")
else:
    print("No model saved.")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2104578179.py in <cell line: 0>()
      7 epoch = 0
      8 while epoch < epochs:
----> 9     model.train()
     10     train_loss = 0.0
     11     train_rmse = 0.0

NameError: name 'model' is not defined

## === cell 18
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outs = model(batch)  # (B,1,420,540)
        all_outputs.append(outs.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # (N,1,420,540)
print("All outputs shape:", all_outputs.shape)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/652653879.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load("best_model.pth", map_location=device))
      2 model.eval()
      3 all_outputs = []
      4 with torch.no_grad():
      5     for batch in test_loader:

NameError: name 'model' is not defined

## === cell 19
import csv
from PIL import Image

target_size = (420, 540)  # (height, width)


def compute_padding(orig_size, target_size):
    orig_h, orig_w = orig_size
    tgt_h, tgt_w = target_size
    pad_top = (tgt_h - orig_h) // 2 if tgt_h > orig_h else 0
    pad_bottom = tgt_h - orig_h - pad_top if tgt_h > orig_h else 0
    pad_left = (tgt_w - orig_w) // 2 if tgt_w > orig_w else 0
    pad_right = tgt_w - orig_w - pad_left if tgt_w > orig_w else 0
    return pad_top, pad_bottom, pad_left, pad_right


def remove_padding(pred, orig_size, target_size):
    orig_h, orig_w = orig_size
    pad_top, _, pad_left, _ = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


test_file_paths = sorted(
    [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

submission_rows = []
for idx, path in enumerate(test_file_paths):
    img_id = os.path.splitext(os.path.basename(path))[0]
    orig_img = Image.open(path).convert("L")
    orig_w, orig_h = orig_img.size
    orig_size = (orig_h, orig_w)

    pred = all_outputs[idx]  # (1,420,540)
    cropped = remove_padding(pred, orig_size, target_size).squeeze(0).numpy()
    for r in range(orig_h):
        for c in range(orig_w):
            pixel_id = f"{img_id}_{r+1}_{c+1}"
            pixel_val = float(cropped[r, c])
            submission_rows.append((pixel_id, pixel_val))

submission_path = "submission.csv"
with open(submission_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_rows)

print(f"Submission file written to {submission_path}")

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/979545329.py in <cell line: 0>()
     22 
     23 test_file_paths = sorted(
---> 24     [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")],
     25     key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
     26 )

NameError: name 'os' is not defined
