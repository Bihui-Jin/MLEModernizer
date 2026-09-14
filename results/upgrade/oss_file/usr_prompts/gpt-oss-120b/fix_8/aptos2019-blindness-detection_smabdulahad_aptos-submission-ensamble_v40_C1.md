# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.5828832716725559

# 6. Current score

-0.15151

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09672) has done: 'I fix the Pillow ANTIALIAS error, make the optional image‑resize step safe, and change model loading so that if the checkpoint files are missing the script falls back to using ImageNet‑pretrained weights (preserving the original architecture). I also point the test dataset to the original images directory so the DataLoader can find the files, and ensure the submission CSV is written correctly.'
- What this solution (achieved -0.17999) has done: 'I adjust the post‑processing of the model outputs so that predictions are based on the expected value of the class probabilities (rounded to the nearest integer) instead of a simple argmax. This often yields higher quadratic weighted kappa because it better respects the ordinal nature of the labels. The change is limited to the inference cells, preserving the model architecture and training logic unchanged.'
- What this solution (achieved 0.03465) has done: 'I load the training labels to compute the overall mean diagnosis and blend this simple baseline with the model’s expected‑value predictions. By averaging the model output with the global mean (weighted 0.5 each) and then rounding, the predictions are pulled toward a more realistic distribution, which should raise the quadratic weighted kappa from the current negative value toward the target. The rest of the pipeline and model logic remain unchanged.'
- What this solution (achieved -0.03743) has done: 'I reduce the excessive blending with the global mean label, because averaging the model’s expected‑value predictions with a constant pulls the outputs toward the overall average and harms the quadratic weighted kappa. By weighting the model predictions much more (e.g., 90 % model, 10 % mean) or eliminating the mean term entirely, the predictions stay closer to what the model estimates, which is expected to raise the score toward the target. The change is limited to the inference cell and keeps all other logic intact.'
- What this solution (achieved 0.01199) has done: 'I remove the unnecessary blending with the global mean, letting the model’s own expected‑value predictions dominate (model_weight = 1.0, mean_weight = 0.0). This small change keeps the overall pipeline intact while eliminating the bias that was pulling predictions toward the overall average and is expected to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.16863) has done: 'I adjust the inference step to use a simple but effective test‑time augmentation (horizontal flip) and switch the final decision from the expected‑value rounding to an argmax over the averaged softmax probabilities. This keeps the model architecture unchanged, adds only a lightweight augmentation, and is expected to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.15151) has done: 'I replace the argmax‑based prediction with an expected‑value calculation (probability‑weighted sum of class indices rounded to the nearest integer). This respects the ordinal nature of the diagnosis labels and usually yields a higher quadratic weighted kappa, moving the score from the current negative value toward the target while keeping the model architecture and inference pipeline unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import warnings
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from multiprocessing import Pool
import cv2




## === cell 1
import shutil

try:
    shutil.rmtree("/kaggle/working/train")

    model_file_to_delete = "/kaggle/working/models"

    if os.path.isfile(model_file_to_delete):
        os.remove(model_file_to_delete)

except:
    print("No such directories")




## === cell 2
def load_data(data_dir):
    test_csv = os.path.join(data_dir, "test.csv")

    test = pd.read_csv(test_csv)

    test_dir = os.path.join(data_dir, "test_images/")

    test["file_path"] = test["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )

    test["file_name"] = test["id_code"] + ".png"

    return test




## === cell 3
data_dir = "/kaggle/input/aptos2019-blindness-detection/"
test_df = load_data(data_dir)




## === cell 4
def crop_img(img, percentage):

    img_arr = np.array(img)
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

    threshold = img_gray > 0.1 * np.mean(img_gray[img_gray != 0])
    row_sums = np.sum(threshold, axis=1)
    col_sums = np.sum(threshold, axis=0)

    rows = np.where(row_sums > img_arr.shape[1] * percentage)[0]
    cols = np.where(col_sums > img_arr.shape[0] * percentage)[0]

    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)

    crop_img = img_arr[min_row : max_row + 1, min_col : max_col + 1]

    return Image.fromarray(crop_img)




## === cell 5
def resize_maintain_aspect(img, desired_size):
    old_width, old_height = img.size
    aspect_ratio = old_width / old_height

    if aspect_ratio > 1:
        new_width = desired_size
        new_height = int(desired_size / aspect_ratio)
    else:
        new_height = desired_size
        new_width = int(desired_size * aspect_ratio)

    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    padded_image = Image.new("RGB", (desired_size, desired_size))
    x_offset = (desired_size - new_width) // 2
    y_offset = (desired_size - new_height) // 2
    padded_image.paste(resized_img, (x_offset, y_offset))

    return padded_image




## === cell 6
def save_single(args):
    image_path, output_path_folder, percentage, output_size = args
    image = Image.open(image_path)

    croped_img = crop_img(image, percentage)
    image_resized = resize_maintain_aspect(croped_img, desired_size=output_size[0])

    output_image_path = os.path.basename(image_path)
    output_file_path = os.path.join(output_path_folder, output_image_path)
    image_resized.save(output_file_path)




## === cell 7
def fast_image_resize(df, output_path_folder, percentage, output_size=None):
    """Uses multiprocessing to make it fast"""
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=100")
        return

    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)

    jobs = []
    for df_item in range(len(df)):
        image_path = df.file_path.iloc[df_item]
        job = (image_path, output_path_folder, percentage, output_size)
        jobs.append(job)

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))




## === cell 8
percentage = 0.01
try:
    fast_image_resize(
        test_df,
        "/kaggle/working/test/images_resized/",
        percentage,
        output_size=(100, 100),
    )
except Exception as e:
    print(f"Image preprocessing failed ({e}), will use original images.")




## === cell 9
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 10
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 11
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images/"
if not os.path.isdir(test_root_dir):
    test_root_dir = "/kaggle/working/test/images_resized/"
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## === cell 12
model_paths = {
    "efficientnet_b5": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/3/efficientnet_b5.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}




## === cell 13
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []

for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights for {model_key} from {path}")
    except Exception as e:
        print(
            f"Could not load checkpoint for {model_key} ({e}); using ImageNet pretrained weights."
        )
        model = timm.create_model(model_name, pretrained=True, num_classes=5)
    model.to(device)
    model.eval()
    models_list.append(model)




## === cell 14
validation_scores = {
    "efficientnet_b5": 0.952,
}




## === cell 15
total_score = sum(validation_scores.values())
weights = {k: v / total_score for k, v in validation_scores.items()}


train_csv_path = os.path.join(data_dir, "train.csv")
train_df = pd.read_csv(train_csv_path)
global_mean_label = train_df["diagnosis"].mean()  # keep as float for blending

all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device)

        outputs_orig = [
            weights[model_key]
            * nn.functional.softmax(model(images), dim=1).unsqueeze(0)
            for model_key, model in zip(model_paths.keys(), models_list)
        ]
        images_flipped = torch.flip(images, dims=[3])  # flip width dimension
        outputs_flip = [
            weights[model_key]
            * nn.functional.softmax(model(images_flipped), dim=1).unsqueeze(0)
            for model_key, model in zip(model_paths.keys(), models_list)
        ]

        outputs = torch.cat(outputs_orig) + torch.cat(outputs_flip)
        outputs = outputs / 2.0  # shape: (num_models, batch, classes)

        weighted_outputs = torch.sum(outputs, dim=0)  # shape: (batch, classes)
        all_outputs.extend(weighted_outputs.cpu().numpy())

all_outputs = np.array(all_outputs)

class_indices = np.arange(5)  # [0,1,2,3,4]
expected_vals = (all_outputs * class_indices).sum(axis=1)
final_predictions = np.rint(expected_vals).astype(int)

final_predictions = np.clip(final_predictions, 0, 4)




## === cell 16
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)
submission_df.to_csv("submission.csv", index=False)
