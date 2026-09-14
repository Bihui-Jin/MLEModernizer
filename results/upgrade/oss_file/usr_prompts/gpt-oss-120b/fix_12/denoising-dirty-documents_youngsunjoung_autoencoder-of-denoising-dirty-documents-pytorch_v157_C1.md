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

0.3091

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.40939) has done: 'The fix removes the non‑code description cell that caused a syntax error, replaces notebook‑style `!` commands with standard Python calls, points the data directories to the already‑extracted Kaggle input paths, and ensures the submission CSV is written to the working directory with the correct `.csv` extension. These minimal changes let the script run from start to finish and produce a valid `submission.csv` while keeping the original model and training logic untouched.'
- What this solution (achieved 0.47111) has done: 'Implemented fixes:
- Replaced the nonexistent `v2.ToDtype` with `v2.ConvertImageDtype(torch.float32)` in the transform pipelines.
- Added a safe‑load guard for the best model to avoid a crash when the checkpoint file is missing.
- Adjusted cell numbering to start at 1 while preserving the original order and logic.'
- What this solution (achieved 0.42418) has done: 'I increase the training duration so the model can converge better: the `epochs` variable is changed from 5 to 20 while keeping the early‑stopping logic unchanged. This modest change lets the network see more updates without altering its architecture or loss formulation, which should reduce the validation RMSE and move the score closer to the target.'
- What this solution (achieved 0.42262) has done: 'I slightly increase training length, add a small L‑1 regularisation term, and lower the weight‑decay so the model can keep improving a bit more without altering its architecture. These changes keep the original pipeline intact while moving the RMSE toward the target.'
- What this solution (achieved 0.39115) has done: 'I tighten the training pipeline: remove the random augmentations from the training transform (they add unnecessary variability), increase the L1 regularisation weight to guide smoother outputs, lower weight‑decay so the model can fit the data better, and give the model a few more epochs with a slightly longer early‑stopping patience. These modest tweaks keep the original architecture and loss semantics while nudging the validation RMSE down toward the target.'
- What this solution (achieved 0.42761) has done: 'I remove the L1 regularisation term and its weight, set the optimizer’s weight decay to 0, and give the training loop a few more epochs with a longer early‑stopping patience. These small tweaks keep the model architecture unchanged while allowing it to fit the data more closely, which should lower the RMSE toward the target value.'

# 9. Code solution

## === cell 0
train_dir = "/kaggle/input/denoising-dirty-documents/train"
train_cleaned_dir = "/kaggle/input/denoising-dirty-documents/train_cleaned"
test_dir = "/kaggle/input/denoising-dirty-documents/test"




## === cell 1
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images




## === cell 2
train_images = load_images_from_folder(train_dir)
train_cleaned_images = load_images_from_folder(train_cleaned_dir)
test_images = load_images_from_folder(test_dir)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/148537737.py in <cell line: 0>()
----> 1 train_images = load_images_from_folder(train_dir)
      2 train_cleaned_images = load_images_from_folder(train_cleaned_dir)
      3 test_images = load_images_from_folder(test_dir)
      4 

/tmp/ipykernel_55/1014508565.py in load_images_from_folder(folder)
      1 def load_images_from_folder(folder):
      2     images = []
----> 3     for filename in os.listdir(folder):
      4         if filename.lower().endswith(".png"):
      5             img_path = os.path.join(folder, filename)

NameError: name 'os' is not defined

## === cell 3
print(f"train_images: {len(train_images)}")
print(f"train_cleaned_images: {len(train_cleaned_images)}")
print(f"test_images: {len(test_images)}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1348431945.py in <cell line: 0>()
----> 1 print(f"train_images: {len(train_images)}")
      2 print(f"train_cleaned_images: {len(train_cleaned_images)}")
      3 print(f"test_images: {len(test_images)}")
      4 

NameError: name 'train_images' is not defined

## === cell 4
train_sizes = [img.shape[:2] for img in train_images]
train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
test_sizes = [img.shape[:2] for img in test_images]



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3322215904.py in <cell line: 0>()
----> 1 train_sizes = [img.shape[:2] for img in train_images]
      2 train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
      3 test_sizes = [img.shape[:2] for img in test_images]
      4 

NameError: name 'train_images' is not defined

## === cell 5
unique_train_sizes = np.unique(train_sizes, axis=0)
unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
unique_test_sizes = np.unique(test_sizes, axis=0)

print(f"train_images:\n {unique_train_sizes}")
print(f"train_cleaned_images:\n {unique_train_cleaned_sizes}")
print(f"test_images:\n {unique_test_sizes}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/786564925.py in <cell line: 0>()
----> 1 unique_train_sizes = np.unique(train_sizes, axis=0)
      2 unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
      3 unique_test_sizes = np.unique(test_sizes, axis=0)
      4 
      5 print(f"train_images:\n {unique_train_sizes}")

NameError: name 'np' is not defined

## === cell 6
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




## === cell 7
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        grayscale_img = pil_img.convert("L")
        return TF.to_tensor(grayscale_img)




## === cell 8
train_transforms = v2.Compose(
    [
        v2.ToTensor(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ConvertImageDtype(torch.float32),
    ]
)

val_transforms = v2.Compose(
    [
        v2.ToTensor(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ConvertImageDtype(torch.float32),
    ]
)

test_transforms = v2.Compose(
    [
        v2.ToTensor(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ConvertImageDtype(torch.float32),
    ]
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1956742993.py in <cell line: 0>()
----> 1 train_transforms = v2.Compose(
      2     [
      3         v2.ToTensor(),
      4         PadToSize((420, 540)),
      5         Grayscale(),

NameError: name 'v2' is not defined

## === cell 9
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




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2583785652.py in <cell line: 0>()
----> 1 class ImageDataset(Dataset):
      2     def __init__(self, data_dir, transform=None):
      3         self.data_dir = data_dir
      4         self.transform = transform
      5         self.image_files = sorted(

NameError: name 'Dataset' is not defined

## === cell 10
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




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1128120163.py in <cell line: 0>()
      2     [
      3         os.path.join(train_dir, f)
----> 4         for f in os.listdir(train_dir)
      5         if f.lower().endswith(".png")
      6     ]

NameError: name 'os' is not defined

## === cell 11
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




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2476192371.py in <cell line: 0>()
----> 1 class PairedImageDataset(Dataset):
      2     def __init__(self, train_files, cleaned_files, transform=None):
      3         self.train_files = train_files
      4         self.cleaned_files = cleaned_files
      5         self.transform = transform

NameError: name 'Dataset' is not defined

## === cell 12
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/561502297.py in <cell line: 0>()
----> 1 train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
      2 val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
      3 test_dataset = ImageDataset(test_dir, test_transforms)
      4 
      5 train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

NameError: name 'PairedImageDataset' is not defined

## === cell 13
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




## === cell 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3093293998.py in <cell line: 0>()
----> 1 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      2 print("Using device:", device)
      3 
      4 

NameError: name 'torch' is not defined

## === cell 15
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(1, 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(8, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
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
            nn.Dropout(p=0.3),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=1, stride=1, bias=False),
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
            nn.Dropout(p=0.3),
        )
        self.dec3 = nn.Sequential(
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
            nn.Dropout(p=0.2),
        )
        self.dec2 = nn.Sequential(
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
            nn.Dropout(p=0.1),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=1, stride=1, bias=False),
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
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, stride=1, bias=False)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)

        dec5_out = self.dec5(enc5_out)
        dec5_out = F.interpolate(
            dec5_out, size=(26, 33), mode="bilinear", align_corners=False
        )
        dec5_out = torch.cat([dec5_out, enc4_out], dim=1)
        dec5_out = self.conv5(dec5_out)

        dec4_out = self.dec4(enc4_out)
        dec4_out = F.interpolate(
            dec4_out, size=(52, 67), mode="bilinear", align_corners=False
        )
        dec4_out = torch.cat([dec4_out, enc3_out], dim=1)
        dec4_out = self.conv4(dec4_out)

        dec3_out = self.dec3(dec4_out)
        dec3_out = F.interpolate(
            dec3_out, size=(105, 135), mode="bilinear", align_corners=False
        )

        dec2_out = self.dec2(dec3_out)
        dec2_out = F.interpolate(
            dec2_out, size=(210, 270), mode="bilinear", align_corners=False
        )

        dec1_out = self.dec1(dec2_out)
        dec1_out = F.interpolate(
            dec1_out, size=(420, 540), mode="bilinear", align_corners=False
        )

        return torch.clamp(dec1_out, min=0.001, max=0.999)


model = DenoisingAutoencoder().to(device)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1695100182.py in <cell line: 0>()
----> 1 class DenoisingAutoencoder(nn.Module):
      2     def __init__(self):
      3         super(DenoisingAutoencoder, self).__init__()
      4         self.enc1 = nn.Sequential(
      5             nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),

NameError: name 'nn' is not defined

## === cell 16
from torchinfo import summary

summary(model, input_size=(16, 1, 420, 540), device=device)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1764675256.py in <cell line: 0>()
      1 from torchinfo import summary
      2 
----> 3 summary(model, input_size=(16, 1, 420, 540), device=device)
      4 
      5 

NameError: name 'model' is not defined

## === cell 17
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3903342212.py in <cell line: 0>()
----> 1 class RMSELoss(nn.Module):
      2     def forward(self, pred, target):
      3         return torch.sqrt(F.mse_loss(pred, target))
      4 
      5 

NameError: name 'nn' is not defined

## === cell 18
class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=1.0, lambda_l1=0.05):
        super(HybridLoss, self).__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse_loss(
            pred, target
        ) + self.lambda_l1 * self.l1_loss(pred, target)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1935175925.py in <cell line: 0>()
----> 1 class HybridLoss(nn.Module):
      2     # Added a small L1 component (lambda_l1=0.05) to encourage smoother outputs
      3     def __init__(self, lambda_rmse=1.0, lambda_l1=0.05):
      4         super(HybridLoss, self).__init__()
      5         self.lambda_rmse = lambda_rmse

NameError: name 'nn' is not defined

## === cell 19
criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/611087876.py in <cell line: 0>()
----> 1 criterion = HybridLoss()
      2 # Introduced a tiny weight decay (1e-5) for better generalisation
      3 optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
      4 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      5     optimizer, mode="min", factor=0.5, patience=3

NameError: name 'HybridLoss' is not defined

## === cell 20
epochs = 80
patience = 15
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

epoch = 0
while epoch < epochs:
    model.train()
    train_loss = 0.0
    train_rmse_loss = 0.0
    for train_imgs, train_cleaned_imgs in train_loader:
        train_imgs = train_imgs.to(device)
        train_cleaned_imgs = train_cleaned_imgs.to(device)
        optimizer.zero_grad()
        outputs = model(train_imgs)
        loss = criterion(outputs, train_cleaned_imgs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_rmse_loss += RMSELoss()(outputs, train_cleaned_imgs).item()
    train_loss /= len(train_loader)
    train_rmse_loss /= len(train_loader)
    print(
        f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, RMSE: {train_rmse_loss:.4f}"
    )

    model.eval()
    val_loss = 0.0
    val_rmse_loss = 0.0
    with torch.no_grad():
        for val_imgs, val_cleaned_imgs in val_loader:
            val_imgs = val_imgs.to(device)
            val_cleaned_imgs = val_cleaned_imgs.to(device)
            outputs = model(val_imgs)
            loss = criterion(outputs, val_cleaned_imgs)
            val_loss += loss.item()
            val_rmse_loss += RMSELoss()(outputs, val_cleaned_imgs).item()
    val_loss /= len(val_loader)
    val_rmse_loss /= len(val_loader)

    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    if current_lr != prev_lr:
        print(f"Learning Rate updated: {current_lr:.6f}")

    if val_loss < best_val_loss:
        print(f"New best val loss: {val_loss:.4f}, RMSE: {val_rmse_loss:.4f}")
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"Val loss did not improve (counter {early_stop_counter}/{patience})")

    if early_stop_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        model.load_state_dict(best_model_state)
        break

    epoch += 1

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print(f"Best model saved (val loss = {best_val_loss:.4f})")
else:
    print("No model saved.")



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2283688890.py in <cell line: 0>()
      9 epoch = 0
     10 while epoch < epochs:
---> 11     model.train()
     12     train_loss = 0.0
     13     train_rmse_loss = 0.0

NameError: name 'model' is not defined

## === cell 21
best_model_path = "best_model.pth"
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    print("Loaded best model for inference.")
else:
    print("Best model file not found; using current model weights.")



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2794510944.py in <cell line: 0>()
      1 best_model_path = "best_model.pth"
----> 2 if os.path.exists(best_model_path):
      3     model.load_state_dict(torch.load(best_model_path, map_location=device))
      4     model.eval()
      5     print("Loaded best model for inference.")

NameError: name 'os' is not defined

## === cell 22
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)  # (B, 1, 420, 540)
        all_outputs.append(outputs.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # (N_test, 1, 420, 540)
print("All outputs shape:", all_outputs.shape)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1744883854.py in <cell line: 0>()
      1 all_outputs = []
----> 2 with torch.no_grad():
      3     for batch in test_loader:
      4         batch = batch.to(device)
      5         outputs = model(batch)  # (B, 1, 420, 540)

NameError: name 'torch' is not defined

## === cell 23
target_size = (420, 540)  # (height, width)


def compute_padding(orig_size, target_size):
    oh, ow = orig_size
    th, tw = target_size
    pad_top = (th - oh) // 2 if th > oh else 0
    pad_bottom = th - oh - pad_top if th > oh else 0
    pad_left = (tw - ow) // 2 if tw > ow else 0
    pad_right = tw - ow - pad_left if tw > ow else 0
    return pad_top, pad_bottom, pad_left, pad_right


def remove_padding(pred, orig_size, target_size):
    oh, ow = orig_size
    pad_top, _, pad_left, _ = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + oh, pad_left : pad_left + ow]


test_file_paths = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

submission_rows = []
for idx, file_path in enumerate(test_file_paths):
    image_id = os.path.splitext(os.path.basename(file_path))[0]

    with Image.open(file_path) as img:
        ow, oh = img.size
    orig_size = (oh, ow)

    pred = all_outputs[idx]  # (1, 420, 540)
    cropped = remove_padding(pred, orig_size, target_size)  # (1, oh, ow)
    pred_np = cropped.squeeze(0).numpy()  # (oh, ow)

    for row in range(oh):
        for col in range(ow):
            pixel_id = f"{image_id}_{row+1}_{col+1}"
            pixel_value = float(pred_np[row, col])
            submission_rows.append((pixel_id, pixel_value))

submission_path = os.path.join("/kaggle/working", "submission.csv")
with open(submission_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_rows)

print(f"Submission file written to {submission_path}")

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3418729757.py in <cell line: 0>()
     21     [
     22         os.path.join(test_dir, f)
---> 23         for f in os.listdir(test_dir)
     24         if f.lower().endswith(".png")
     25     ],

NameError: name 'os' is not defined
