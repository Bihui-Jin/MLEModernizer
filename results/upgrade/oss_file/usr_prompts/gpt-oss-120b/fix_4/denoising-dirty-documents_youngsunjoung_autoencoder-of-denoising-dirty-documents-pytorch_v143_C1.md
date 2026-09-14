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
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
        input/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
            test/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
            train/
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

# 5. Target score

0.27246

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
try:
    from torchinfo import summary
except ImportError:
    import subprocess, sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])
    from torchinfo import summary




## === cell 1
import zipfile


def unzip_to_dir(zip_path, target_dir):
    if os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)


base_input = "/kaggle/input/denoising-dirty-documents"
dest_dir = base_input  # No unzip needed; keep path simple




## === cell 2
train_dir = os.path.join(base_input, "train")
train_cleaned_dir = os.path.join(base_input, "train_cleaned")
test_dir = os.path.join(base_input, "test")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2265193686.py in <cell line: 0>()
----> 1 train_dir = os.path.join(base_input, "train")
      2 train_cleaned_dir = os.path.join(base_input, "train_cleaned")
      3 test_dir = os.path.join(base_input, "test")
      4 
      5 

NameError: name 'os' is not defined

## === cell 3
def load_images_from_folder(folder):
    """Load all PNG images from a folder. Returns an empty list if the folder does not exist."""
    images = []
    if not os.path.isdir(folder):
        return images
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    return images




## === cell 4
train_images = load_images_from_folder(train_dir)
train_cleaned_images = load_images_from_folder(train_cleaned_dir)
test_images = load_images_from_folder(test_dir)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2899237143.py in <cell line: 0>()
----> 1 train_images = load_images_from_folder(train_dir)
      2 train_cleaned_images = load_images_from_folder(train_cleaned_dir)
      3 test_images = load_images_from_folder(test_dir)
      4 
      5 

NameError: name 'train_dir' is not defined

## === cell 5
print(f"train_images: {len(train_images)}")
print(f"train_cleaned_images: {len(train_cleaned_images)}")
print(f"test_images: {len(test_images)}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1788669715.py in <cell line: 0>()
----> 1 print(f"train_images: {len(train_images)}")
      2 print(f"train_cleaned_images: {len(train_cleaned_images)}")
      3 print(f"test_images: {len(test_images)}")
      4 
      5 

NameError: name 'train_images' is not defined

## === cell 6
if train_images:
    train_images[0]




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2728390984.py in <cell line: 0>()
      1 # Optional sanity check – will not raise if lists are empty
----> 2 if train_images:
      3     train_images[0]
      4 
      5 

NameError: name 'train_images' is not defined

## === cell 7
if train_cleaned_images:
    train_cleaned_images[0]




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4114036845.py in <cell line: 0>()
----> 1 if train_cleaned_images:
      2     train_cleaned_images[0]
      3 
      4 

NameError: name 'train_cleaned_images' is not defined

## === cell 8
if test_images:
    test_images[0]




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4056466273.py in <cell line: 0>()
----> 1 if test_images:
      2     test_images[0]
      3 
      4 

NameError: name 'test_images' is not defined

## === cell 9
if train_images:
    train_sizes = [img.shape[:2] for img in train_images]
if train_cleaned_images:
    train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
if test_images:
    test_sizes = [img.shape[:2] for img in test_images]




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1216360106.py in <cell line: 0>()
----> 1 if train_images:
      2     train_sizes = [img.shape[:2] for img in train_images]
      3 if train_cleaned_images:
      4     train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
      5 if test_images:

NameError: name 'train_images' is not defined

## === cell 10
if train_images:
    unique_train_sizes = np.unique(train_sizes, axis=0)
if train_cleaned_images:
    unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
if test_images:
    unique_test_sizes = np.unique(test_sizes, axis=0)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4279423986.py in <cell line: 0>()
----> 1 if train_images:
      2     unique_train_sizes = np.unique(train_sizes, axis=0)
      3 if train_cleaned_images:
      4     unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
      5 if test_images:

NameError: name 'train_images' is not defined

## === cell 11
if train_images:
    print(f"train_images:\n {unique_train_sizes}")
if train_cleaned_images:
    print(f"train_cleaned_images:\n {unique_train_cleaned_sizes}")
if test_images:
    print(f"test_images:\n {unique_test_sizes}")




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2095181778.py in <cell line: 0>()
----> 1 if train_images:
      2     print(f"train_images:\n {unique_train_sizes}")
      3 if train_cleaned_images:
      4     print(f"train_cleaned_images:\n {unique_train_cleaned_sizes}")
      5 if test_images:

NameError: name 'train_images' is not defined

## === cell 12
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




## === cell 13
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        grayscale_img = pil_img.convert("L")
        return TF.to_tensor(grayscale_img)




## === cell 14
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




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2346121324.py in <cell line: 0>()
----> 1 train_transforms = v2.Compose(
      2     [
      3         v2.ToImage(),
      4         PadToSize((420, 540)),
      5         v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),

NameError: name 'v2' is not defined

## === cell 15
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.lower().endswith(".png")
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




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2583785652.py in <cell line: 0>()
----> 1 class ImageDataset(Dataset):
      2     def __init__(self, data_dir, transform=None):
      3         self.data_dir = data_dir
      4         self.transform = transform
      5         self.image_files = sorted(

NameError: name 'Dataset' is not defined

## === cell 16
train_files = sorted(
    [
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.lower().endswith(".png")
    ]
)
cleaned_files = sorted(
    [
        os.path.join(train_cleaned_dir, f)
        for f in os.listdir(train_cleaned_dir)
        if f.lower().endswith(".png")
    ]
)

train_files, val_files, cleaned_train, cleaned_val = train_test_split(
    train_files, cleaned_files, test_size=2 / 9, random_state=42
)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1128120163.py in <cell line: 0>()
      2     [
      3         os.path.join(train_dir, f)
----> 4         for f in os.listdir(train_dir)
      5         if f.lower().endswith(".png")
      6     ]

NameError: name 'os' is not defined

## === cell 17
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




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2476192371.py in <cell line: 0>()
----> 1 class PairedImageDataset(Dataset):
      2     def __init__(self, train_files, cleaned_files, transform=None):
      3         self.train_files = train_files
      4         self.cleaned_files = cleaned_files
      5         self.transform = transform

NameError: name 'Dataset' is not defined

## === cell 18
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/561502297.py in <cell line: 0>()
----> 1 train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
      2 val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
      3 test_dataset = ImageDataset(test_dir, test_transforms)
      4 
      5 train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

NameError: name 'PairedImageDataset' is not defined

## === cell 19
def visualize_paired_dataset(paired_loader, num_images=5):
    for train_images, cleaned_images in paired_loader:
        fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 3))
        for i in range(num_images):
            axes[i, 0].imshow(
                train_images[i].permute(1, 2, 0).cpu().numpy(), cmap="gray"
            )
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(
                cleaned_images[i].permute(1, 2, 0).cpu().numpy(), cmap="gray"
            )
            axes[i, 1].set_title(f"Cleaned {i+1}")
            axes[i, 1].axis("off")
        plt.tight_layout()
        plt.show()
        break






## === cell 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3093293998.py in <cell line: 0>()
----> 1 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      2 print("Using device:", device)
      3 
      4 

NameError: name 'torch' is not defined

## === cell 21
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(1, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=1, bias=False),
            nn.ConvTranspose2d(
                64,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=64,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=1, bias=False),
            nn.ConvTranspose2d(
                32,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=32,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=1, bias=False),
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
            nn.Dropout(0.2),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=1, bias=False),
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
            nn.Dropout(0.1),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=1, bias=False),
            nn.ConvTranspose2d(
                1,
                1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=1, bias=False)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec5 = F.interpolate(dec5, size=(26, 33), mode="bilinear", align_corners=False)
        dec5 = torch.cat([dec5, enc4], dim=1)
        dec5 = self.conv5(dec5)

        dec4 = self.dec4(enc4)
        dec4 = F.interpolate(dec4, size=(52, 67), mode="bilinear", align_corners=False)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.conv4(dec4)

        dec3 = self.dec3(dec4)
        dec3 = F.interpolate(
            dec3, size=(105, 135), mode="bilinear", align_corners=False
        )

        dec2 = self.dec2(dec3)
        dec2 = F.interpolate(
            dec2, size=(210, 270), mode="bilinear", align_corners=False
        )

        dec1 = self.dec1(dec2)
        dec1 = F.interpolate(
            dec1, size=(420, 540), mode="bilinear", align_corners=False
        )

        return torch.clamp(dec1, 0.001, 0.999)


model = DenoisingAutoencoder().to(device)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/458130967.py in <cell line: 0>()
----> 1 class DenoisingAutoencoder(nn.Module):
      2     def __init__(self):
      3         super(DenoisingAutoencoder, self).__init__()
      4         self.enc1 = nn.Sequential(
      5             nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),

NameError: name 'nn' is not defined

## === cell 22
summary(model, input_size=(1, 1, 420, 540), device=device)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2487729042.py in <cell line: 0>()
----> 1 summary(model, input_size=(1, 1, 420, 540), device=device)
      2 
      3 

NameError: name 'model' is not defined

## === cell 23
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3903342212.py in <cell line: 0>()
----> 1 class RMSELoss(nn.Module):
      2     def forward(self, pred, target):
      3         return torch.sqrt(F.mse_loss(pred, target))
      4 
      5 

NameError: name 'nn' is not defined

## === cell 24
class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=0.8, lambda_l1=0.2):
        super().__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse_loss(
            pred, target
        ) + self.lambda_l1 * self.l1_loss(pred, target)




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/708863712.py in <cell line: 0>()
----> 1 class HybridLoss(nn.Module):
      2     def __init__(self, lambda_rmse=0.8, lambda_l1=0.2):
      3         super().__init__()
      4         self.lambda_rmse = lambda_rmse
      5         self.lambda_l1 = lambda_l1

NameError: name 'nn' is not defined

## === cell 25
criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3167067772.py in <cell line: 0>()
----> 1 criterion = HybridLoss()
      2 optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
      3 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      4     optimizer, mode="min", factor=0.5, patience=2
      5 )

NameError: name 'HybridLoss' is not defined

## === cell 26
epochs = 15  # modest number of epochs for quick training
patience = 5
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_rmse = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        out = model(noisy)
        loss = criterion(out, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_rmse += RMSELoss()(out, clean).item()
    train_loss /= len(train_loader)
    train_rmse /= len(train_loader)

    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            loss = criterion(out, clean)
            val_loss += loss.item()
            val_rmse += RMSELoss()(out, clean).item()
    val_loss /= len(val_loader)
    val_rmse /= len(val_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} RMSE: {train_rmse:.4f} | "
        f"Val Loss: {val_loss:.4f} RMSE: {val_rmse:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print("  ** New best model saved **")
    else:
        early_stop_counter += 1
        print(f"  No improvement. Early stop counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print("Best model checkpoint saved.")




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2109349184.py in <cell line: 0>()
      6 
      7 for epoch in range(epochs):
----> 8     model.train()
      9     train_loss = 0.0
     10     train_rmse = 0.0

NameError: name 'model' is not defined

## === cell 27
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)
        break  # just to verify forward pass works




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2680891536.py in <cell line: 0>()
      1 best_model_path = "best_model.pth"
----> 2 model.load_state_dict(torch.load(best_model_path, map_location=device))
      3 model.eval()
      4 with torch.no_grad():
      5     for batch in test_loader:

NameError: name 'model' is not defined

## === cell 28
def visualize_images_and_outputs(images, outputs):
    num_images = images.size(0)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))
    for i in range(num_images):
        axes[i, 0].imshow(images[i].cpu().squeeze().numpy(), cmap="gray")
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(outputs[i].cpu().squeeze().numpy(), cmap="gray")
        axes[i, 1].set_title(f"Output {i+1}")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()






## === cell 29
print("Test directory:", test_dir)




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2447717360.py in <cell line: 0>()
----> 1 print("Test directory:", test_dir)
      2 
      3 

NameError: name 'test_dir' is not defined

## === cell 30
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outs = model(batch)
        all_outputs.append(outs.cpu())
all_outputs = torch.cat(all_outputs, dim=0)
print("All outputs shape:", all_outputs.shape)




## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2464221689.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load(best_model_path, map_location=device))
      2 model.eval()
      3 all_outputs = []
      4 with torch.no_grad():
      5     for batch in test_loader:

NameError: name 'model' is not defined

## === cell 31
import csv
from PIL import Image

target_size = (420, 540)  # (height, width)


def compute_padding(orig_size, target_size):
    orig_h, orig_w = orig_size
    tgt_h, tgt_w = target_size
    pad_top = (tgt_h - orig_h) // 2 if tgt_h > orig_h else 0
    pad_bottom = (tgt_h - orig_h - pad_top) if tgt_h > orig_h else 0
    pad_left = (tgt_w - orig_w) // 2 if tgt_w > orig_w else 0
    pad_right = (tgt_w - orig_w - pad_left) if tgt_w > orig_w else 0
    return pad_top, pad_bottom, pad_left, pad_right


def remove_padding(pred, orig_size, target_size):
    orig_h, orig_w = orig_size
    pad_top, _, pad_left, _ = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


test_file_paths = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

submission_data = []
for idx, file_path in enumerate(test_file_paths):
    image_id = os.path.splitext(os.path.basename(file_path))[0]

    orig_img = Image.open(file_path).convert("L")
    orig_w, orig_h = orig_img.size
    orig_size = (orig_h, orig_w)

    pred = all_outputs[idx]  # (1, 420, 540)
    cropped = remove_padding(pred, orig_size, target_size)
    pred_np = cropped.squeeze(0).numpy()

    for r in range(orig_h):
        for c in range(orig_w):
            pixel_id = f"{image_id}_{r+1}_{c+1}"
            pixel_value = float(pred_np[r, c])
            submission_data.append((pixel_id, pixel_value))

submission_file = "submission.csv"
with open(submission_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_data)

print(f"Submission file '{submission_file}' created.")

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1059059473.py in <cell line: 0>()
     24     [
     25         os.path.join(test_dir, f)
---> 26         for f in os.listdir(test_dir)
     27         if f.lower().endswith(".png")
     28     ],

NameError: name 'os' is not defined
