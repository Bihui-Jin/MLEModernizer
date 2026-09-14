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
    if not os.path.isfile(zip_path):
        return
    if not os.path.isdir(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        try:
            subprocess.check_call(["unzip", "-q", zip_path, "-d", extract_to])
        except Exception:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)


base_extract_dir = "/kaggle/working/denoising-dirty-documents"

if not (
    os.path.isdir(os.path.join(base_extract_dir, "train"))
    and os.path.isdir(os.path.join(base_extract_dir, "train_cleaned"))
    and os.path.isdir(os.path.join(base_extract_dir, "test"))
):
    maybe_unzip("/kaggle/input/denoising-dirty-documents/train.zip", base_extract_dir)
    maybe_unzip("/kaggle/input/denoising-dirty-documents/test.zip", base_extract_dir)
    maybe_unzip(
        "/kaggle/input/denoising-dirty-documents/train_cleaned.zip",
        base_extract_dir,
    )


def find_dir(keyword):
    """
    Walk the extraction directory and return the first sub‑folder whose name contains `keyword`
    (case‑insensitive) and that holds at least one PNG file.
    """
    for root, dirs, _ in os.walk(base_extract_dir):
        for d in dirs:
            if keyword.lower() in d.lower():
                candidate = os.path.join(root, d)
                if any(f.lower().endswith(".png") for f in os.listdir(candidate)):
                    return candidate
    fallback = os.path.join(base_extract_dir, "denoising-dirty-documents", keyword)
    if os.path.isdir(fallback) and any(
        f.lower().endswith(".png") for f in os.listdir(fallback)
    ):
        return fallback
    raise FileNotFoundError(f"Unable to locate directory containing '{keyword}'")


train_dir = find_dir("train")
train_cleaned_dir = find_dir("train_cleaned")
test_dir = find_dir("test")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3556941673.py in <cell line: 0>()
     17 # If for some reason the expected folders are missing, fall back to unzipping the originals.
     18 if not (
---> 19     os.path.isdir(os.path.join(base_extract_dir, "train"))
     20     and os.path.isdir(os.path.join(base_extract_dir, "train_cleaned"))
     21     and os.path.isdir(os.path.join(base_extract_dir, "test"))

NameError: name 'os' is not defined

## === cell 1
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




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1128120163.py in <cell line: 0>()
      2     [
      3         os.path.join(train_dir, f)
----> 4         for f in os.listdir(train_dir)
      5         if f.lower().endswith(".png")
      6     ]

NameError: name 'os' is not defined

## === cell 2
epochs = 5
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

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4135868330.py in <cell line: 0>()
      8 epoch = 0
      9 while epoch < epochs:
---> 10     model.train()
     11     train_loss = 0.0
     12     train_rmse = 0.0

NameError: name 'model' is not defined
