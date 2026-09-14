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

0.37181

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.43231) has done: 'The changes remove the stray non‑code text and markdown backticks that caused syntax errors, replace shell‑based unzip and pip commands with pure‑Python equivalents, and ensure the submission‑generation cell runs without extra backticks. No core modeling logic is altered, preserving the original architecture while fixing runtime failures so a valid `submission.csv` is produced.'

# 9. Code solution

## === cell 0
try:
    from torchinfo import summary
except ImportError:
    import subprocess, sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])
    from torchinfo import summary




## === cell 1
base_input = "/kaggle/input/denoising-dirty-documents"
extract_to = "/kaggle/working/denoising_data"
os.makedirs(extract_to, exist_ok=True)

zip_names = ["train.zip", "test.zip", "train_cleaned.zip"]
for zname in zip_names:
    zip_path = os.path.join(base_input, zname)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3423672125.py in <cell line: 0>()
      1 base_input = "/kaggle/input/denoising-dirty-documents"
      2 extract_to = "/kaggle/working/denoising_data"
----> 3 os.makedirs(extract_to, exist_ok=True)
      4 
      5 zip_names = ["train.zip", "test.zip", "train_cleaned.zip"]

NameError: name 'os' is not defined

## === cell 2
train_dir = os.path.join(extract_to, "train")
train_cleaned_dir = os.path.join(extract_to, "train_cleaned")
test_dir = os.path.join(extract_to, "test")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2200178768.py in <cell line: 0>()
----> 1 train_dir = os.path.join(extract_to, "train")
      2 train_cleaned_dir = os.path.join(extract_to, "train_cleaned")
      3 test_dir = os.path.join(extract_to, "test")
      4 
      5 

NameError: name 'os' is not defined

## === cell 3
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
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
/tmp/ipykernel_55/2899237143.py in <cell line: 0>()
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
/tmp/ipykernel_55/1788669715.py in <cell line: 0>()
----> 1 print(f"train_images: {len(train_images)}")
      2 print(f"train_cleaned_images: {len(train_cleaned_images)}")
      3 print(f"test_images: {len(test_images)}")
      4 
      5 

NameError: name 'train_images' is not defined

## === cell 6
train_sizes = [img.shape[:2] for img in train_images]
train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
test_sizes = [img.shape[:2] for img in test_images]




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2428099884.py in <cell line: 0>()
----> 1 train_sizes = [img.shape[:2] for img in train_images]
      2 train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
      3 test_sizes = [img.shape[:2] for img in test_images]
      4 
      5 

NameError: name 'train_images' is not defined

## === cell 7
unique_train_sizes = np.unique(train_sizes, axis=0)
unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
unique_test_sizes = np.unique(test_sizes, axis=0)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/487120744.py in <cell line: 0>()
----> 1 unique_train_sizes = np.unique(train_sizes, axis=0)
      2 unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
      3 unique_test_sizes = np.unique(test_sizes, axis=0)
      4 
      5 

NameError: name 'np' is not defined

## === cell 8
print(f"train_images unique sizes:\n {unique_train_sizes}")
print(f"train_cleaned_images unique sizes:\n {unique_train_cleaned_sizes}")
print(f"test_images unique sizes:\n {unique_test_sizes}")




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2143426994.py in <cell line: 0>()
----> 1 print(f"train_images unique sizes:\n {unique_train_sizes}")
      2 print(f"train_cleaned_images unique sizes:\n {unique_train_cleaned_sizes}")
      3 print(f"test_images unique sizes:\n {unique_test_sizes}")
      4 
      5 

NameError: name 'unique_train_sizes' is not defined

## === cell 9
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




## === cell 10
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        grayscale_img = pil_img.convert("L")
        return TF.to_tensor(grayscale_img)




## === cell 11
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),
        v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.3),
        v2.RandomApply([v2.RandomErasing(p=0.3)], p=0.3),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomApply([v2.GaussianNoise(mean=0, sigma=0.05)], p=0.3),
    ]
)

val_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomApply([v2.GaussianNoise(mean=0, sigma=0.02)], p=0.1),
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




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3435871280.py in <cell line: 0>()
----> 1 train_transforms = v2.Compose(
      2     [
      3         v2.ToImage(),
      4         PadToSize((420, 540)),
      5         v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),

NameError: name 'v2' is not defined

## === cell 12
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




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2583785652.py in <cell line: 0>()
----> 1 class ImageDataset(Dataset):
      2     def __init__(self, data_dir, transform=None):
      3         self.data_dir = data_dir
      4         self.transform = transform
      5         self.image_files = sorted(

NameError: name 'Dataset' is not defined

## === cell 13
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




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1128120163.py in <cell line: 0>()
      2     [
      3         os.path.join(train_dir, f)
----> 4         for f in os.listdir(train_dir)
      5         if f.lower().endswith(".png")
      6     ]

NameError: name 'os' is not defined

## === cell 14
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




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2476192371.py in <cell line: 0>()
----> 1 class PairedImageDataset(Dataset):
      2     def __init__(self, train_files, cleaned_files, transform=None):
      3         self.train_files = train_files
      4         self.cleaned_files = cleaned_files
      5         self.transform = transform

NameError: name 'Dataset' is not defined

## === cell 15
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/561502297.py in <cell line: 0>()
----> 1 train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
      2 val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
      3 test_dataset = ImageDataset(test_dir, test_transforms)
      4 
      5 train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

NameError: name 'PairedImageDataset' is not defined

## === cell 16
def visualize_paired_dataset(paired_loader, num_images=5):
    for train_imgs, cleaned_imgs in paired_loader:
        fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 3))
        for i in range(num_images):
            axes[i, 0].imshow(train_imgs[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(
                cleaned_imgs[i].permute(1, 2, 0).cpu().numpy(), cmap="gray"
            )
            axes[i, 1].set_title(f"Cleaned {i+1}")
            axes[i, 1].axis("off")
        plt.tight_layout()
        plt.show()
        break


visualize_paired_dataset(train_loader, num_images=3)
visualize_paired_dataset(val_loader, num_images=2)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3093423134.py in <cell line: 0>()
     16 
     17 
---> 18 visualize_paired_dataset(train_loader, num_images=3)
     19 visualize_paired_dataset(val_loader, num_images=2)
     20 

NameError: name 'train_loader' is not defined

## === cell 17
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3093293998.py in <cell line: 0>()
----> 1 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      2 print("Using device:", device)
      3 
      4 

NameError: name 'torch' is not defined

## === cell 18
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

        outputs = torch.clamp(dec1_out, min=0.001, max=0.999)
        return outputs


model = DenoisingAutoencoder().to(device)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1261060156.py in <cell line: 0>()
----> 1 class DenoisingAutoencoder(nn.Module):
      2     def __init__(self):
      3         super(DenoisingAutoencoder, self).__init__()
      4 
      5         self.enc1 = nn.Sequential(

NameError: name 'nn' is not defined

## === cell 19
summary(model, input_size=(16, 1, 420, 540), device=device)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/34055172.py in <cell line: 0>()
----> 1 summary(model, input_size=(16, 1, 420, 540), device=device)
      2 
      3 

NameError: name 'model' is not defined

## === cell 20
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3903342212.py in <cell line: 0>()
----> 1 class RMSELoss(nn.Module):
      2     def forward(self, pred, target):
      3         return torch.sqrt(F.mse_loss(pred, target))
      4 
      5 

NameError: name 'nn' is not defined

## === cell 21
class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=0.9, lambda_l1=0.1):
        super(HybridLoss, self).__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse_loss(
            pred, target
        ) + self.lambda_l1 * self.l1_loss(pred, target)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1427731047.py in <cell line: 0>()
      1 # Give a higher weight to the RMSE component (more directly aligned with the competition metric)
----> 2 class HybridLoss(nn.Module):
      3     def __init__(self, lambda_rmse=0.9, lambda_l1=0.1):
      4         super(HybridLoss, self).__init__()
      5         self.lambda_rmse = lambda_rmse

NameError: name 'nn' is not defined

## === cell 22
criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=5
)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3363464799.py in <cell line: 0>()
----> 1 criterion = HybridLoss()
      2 # Reduce learning rate for finer convergence
      3 optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
      4 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      5     optimizer, mode="min", factor=0.1, patience=5

NameError: name 'HybridLoss' is not defined

## === cell 23
epochs = 1500
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
    cur_lr = optimizer.param_groups[0]["lr"]
    if cur_lr != prev_lr:
        print(f"Learning Rate updated: {cur_lr:.6f}")

    if val_loss < best_val_loss:
        print(
            f"New best val loss: {val_loss:.4f} (prev {best_val_loss:.4f}), RMSE: {val_rmse_loss:.4f}"
        )
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(
            f"Val loss did not improve. Early stop counter: {early_stop_counter}/{patience}"
        )

    if early_stop_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        model.load_state_dict(best_model_state)
        break

    epoch += 1

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print(f"Best model saved (val loss = {best_val_loss:.4f})")
else:
    print("No model was saved.")




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2447900824.py in <cell line: 0>()
      8 epoch = 0
      9 while epoch < epochs:
---> 10     model.train()
     11     train_loss = 0.0
     12     train_rmse_loss = 0.0

NameError: name 'model' is not defined

## === cell 24
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path, weights_only=True))
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        _ = model(batch)  # run one batch to verify forward pass
        break




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1875491403.py in <cell line: 0>()
      1 best_model_path = "best_model.pth"
----> 2 model.load_state_dict(torch.load(best_model_path, weights_only=True))
      3 model.eval()
      4 with torch.no_grad():
      5     for batch in test_loader:

NameError: name 'model' is not defined

## === cell 25
def visualize_images_and_outputs(images, outputs):
    n = images.size(0)
    fig, axes = plt.subplots(n, 2, figsize=(10, n * 3))
    for i in range(n):
        axes[i, 0].imshow(images[i].cpu().numpy().squeeze(), cmap="gray")
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(outputs[i].cpu().detach().numpy().squeeze(), cmap="gray")
        axes[i, 1].set_title(f"Output {i+1}")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch)
        visualize_images_and_outputs(batch, preds)
        break




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3726777761.py in <cell line: 0>()
     13 
     14 
---> 15 with torch.no_grad():
     16     for batch in test_loader:
     17         batch = batch.to(device)

NameError: name 'torch' is not defined

## === cell 26
print("Test directory:", test_dir)




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2447717360.py in <cell line: 0>()
----> 1 print("Test directory:", test_dir)
      2 
      3 

NameError: name 'test_dir' is not defined

## === cell 27
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path, weights_only=True))
model.eval()
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)
        all_outputs.append(outputs.cpu())
all_outputs = torch.cat(all_outputs, dim=0)
print("All outputs shape:", all_outputs.shape)




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3944334304.py in <cell line: 0>()
      1 best_model_path = "best_model.pth"
----> 2 model.load_state_dict(torch.load(best_model_path, weights_only=True))
      3 model.eval()
      4 all_outputs = []
      5 with torch.no_grad():

NameError: name 'model' is not defined

## === cell 28
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

    orig_img = Image.open(file_path).convert("L")
    orig_w, orig_h = orig_img.size
    orig_size = (orig_h, orig_w)

    pred = all_outputs[idx]  # shape (1, 420, 540)

    cropped = remove_padding(pred, orig_size, target_size)  # (1, H, W)

    pred_np = cropped.squeeze(0).cpu().numpy()

    for row in range(orig_h):
        for col in range(orig_w):
            pixel_id = f"{image_id}_{row+1}_{col+1}"
            pixel_value = float(pred_np[row, col])
            submission_rows.append((pixel_id, pixel_value))

submission_path = "submission.csv"
with open(submission_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_rows)

print(f"Submission file '{submission_path}' created with {len(submission_rows)} rows.")

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3132167081.py in <cell line: 0>()
     24     [
     25         os.path.join(test_dir, f)
---> 26         for f in os.listdir(test_dir)
     27         if f.lower().endswith(".png")
     28     ],

NameError: name 'os' is not defined
