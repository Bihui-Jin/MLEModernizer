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

0.6738812898573387

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.01612) has done: 'I fix the Pillow deprecation error by using the proper resampling enum, add missing imports, make the model loading robust by falling back to a pretrained model when the checkpoint file is absent, and ensure all paths and variables are correctly defined so the script runs end‑to‑end and writes a valid `submission.csv` file. The core logic and architecture remain unchanged.'
- What this solution (achieved 0.0) has done: 'I replace the model‑based predictions with a simple baseline that predicts the most frequent diagnosis from the training set. This requires loading `train.csv`, computing the mode of the `diagnosis` column, and overwriting `final_predictions` with that constant class. The change is confined to the inference cell, preserves the rest of the pipeline, and is expected to raise the Quadratic Weighted Kappa from the current negative value toward the target positive score.'
- What this solution (achieved 0.13964) has done: 'I replace the constant‑majority‑label fallback with predictions derived from the model’s soft‑max outputs. After aggregating the (weighted) model probabilities, I take the arg‑max across the five classes to produce a per‑image diagnosis, which should raise the quadratic weighted kappa toward the target. The rest of the pipeline and file writing remain unchanged.'
- What this solution (achieved 0.04114) has done: 'I keep the existing data loading, preprocessing, and model‑inference pipeline unchanged, but after obtaining the averaged soft‑max outputs I rescale the class probabilities so that their overall distribution matches the label distribution in the training set. This simple calibration often raises the Quadratic Weighted Kappa without altering the core model architecture or training procedure, moving the score closer to the target.'
- What this solution (achieved -0.02807) has done: 'I replace the post‑processing in **cell 15** that rescales the ensemble probabilities to match the training label distribution, because that scaling reduced the score (it gave 0.041). Instead I take the arg‑max of the already‑weighted soft‑max outputs to produce the final class predictions, which previously achieved a higher QWK (~0.14). This small change keeps the core model, ensemble weighting, and data pipeline unchanged while moving the metric toward the target.'
- What this solution (achieved 0.00095) has done: 'I adjust the post‑processing of the model outputs (cell 15).  
Instead of taking the class with the highest soft‑max probability (argmax), I compute the expected class value from the probability distribution and round it to the nearest integer (clipping to 0‑4). This small change often aligns the predictions better with the ordinal nature of the diagnosis and should raise the quadratic weighted kappa toward the target while leaving the model architecture and training untouched.'
- What this solution (achieved -0.19817) has done: 'I adjust the post‑processing in **cell 15** to use the class with the highest weighted probability (argmax) instead of rounding the expected value. This simple change aligns predictions with the ordinal nature of the labels and typically yields a higher Quadratic Weighted Kappa, moving the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.0) has done: 'I replace the argmax post‑processing in **cell 15** with an expected‑value rounding approach, which respects the ordinal nature of the diagnosis levels. This change keeps the model loading and inference unchanged, but computes `pred = round( Σ p_i * i )` and clamps it to 0‑4, yielding predictions that are typically more aligned with the quadratic weighted kappa metric and should move the score from the current negative value toward the positive target.'
- What this solution (achieved -0.04177) has done: 'I replace the expected‑value rounding in **cell 15** with a simple argmax over the weighted soft‑max outputs. Using the class with the highest probability aligns better with the ordinal quadratic weighted‑kappa metric and should move the score upward toward the target while keeping the overall architecture and workflow unchanged.'
- What this solution (achieved 0.0) has done: 'I replace the argmax‑based prediction in cell 15 with an expected‑value rounding approach, which better respects the ordinal nature of the diagnosis labels and typically raises the Quadratic Weighted Kappa from a negative value toward the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved -0.20563) has done: 'I modify the post‑processing step (cell 15) to use the class with the highest soft‑max probability (argmax) instead of rounding the expected value. This small change usually raises the Quadratic Weighted Kappa from 0.0 toward the target while preserving the rest of the pipeline.'
- What this solution (achieved 0.0) has done: 'I adjust the post‑processing step (cell 15) to use the expected class value from the soft‑max probabilities instead of a hard argmax. Computing a weighted average of class indices and rounding better respects the ordinal nature of the diagnosis labels, which should raise the quadratic weighted kappa toward the target while keeping the rest of the pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from multiprocessing import Pool
import cv2
import shutil



## === cell 1
try:
    shutil.rmtree("/kaggle/working/train")
    model_file_to_delete = "/kaggle/working/models"
    if os.path.isfile(model_file_to_delete):
        os.remove(model_file_to_delete)
except Exception:
    print("No such directories or files to clean.")




## === cell 2
def load_data(data_dir):
    test_csv = os.path.join(data_dir, "test.csv")
    test = pd.read_csv(test_csv)
    test["file_path"] = test["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )
    test["file_name"] = test["id_code"] + ".png"
    return test




## === cell 3
data_dir = "/kaggle/input/aptos2019-blindness-detection/"
test_df = load_data(data_dir)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1929230581.py in <cell line: 0>()
      1 data_dir = "/kaggle/input/aptos2019-blindness-detection/"
----> 2 test_df = load_data(data_dir)
      3 
      4 

/tmp/ipykernel_55/2603374764.py in load_data(data_dir)
      2     test_csv = os.path.join(data_dir, "test.csv")
      3     test = pd.read_csv(test_csv)
----> 4     test["file_path"] = test["id_code"].map(
      5         lambda x: os.path.join(test_dir, f"{x}.png")
      6     )

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in map(self, arg, na_action)
   4698         dtype: object
   4699         """
-> 4700         new_values = self._map_values(arg, na_action=na_action)
   4701         return self._constructor(new_values, index=self.index, copy=False).__finalize__(
   4702             self, method="map"

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_55/2603374764.py in <lambda>(x)
      3     test = pd.read_csv(test_csv)
      4     test["file_path"] = test["id_code"].map(
----> 5         lambda x: os.path.join(test_dir, f"{x}.png")
      6     )
      7     test["file_name"] = test["id_code"] + ".png"

NameError: name 'test_dir' is not defined

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
    resample = getattr(Image, "Resampling", Image).LANCZOS
    resized_img = img.resize((new_width, new_height), resample=resample)
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
    """Uses multiprocessing to resize images quickly."""
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=100")
        return
    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder, exist_ok=True)
    jobs = []
    for i in range(len(df)):
        image_path = df.file_path.iloc[i]
        jobs.append((image_path, output_path_folder, percentage, output_size))
    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))




## === cell 8
percentage = 0.01
fast_image_resize(
    test_df, "/kaggle/working/test/images_resized/", percentage, output_size=(100, 100)
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/111339431.py in <cell line: 0>()
      1 percentage = 0.01
      2 fast_image_resize(
----> 3     test_df, "/kaggle/working/test/images_resized/", percentage, output_size=(100, 100)
      4 )
      5 

NameError: name 'test_df' is not defined

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
test_root_dir = "/kaggle/working/test/images_resized/"
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



## === cell 12
model_paths = {
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/3/inception_resnet_v2.pth",
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
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    try:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights for {model_key} from {path}")
    except Exception as e:
        print(
            f"Could not load checkpoint for {model_key} ({e}); using pretrained weights."
        )
        model = timm.create_model(model_name, pretrained=True, num_classes=5)
    model.to(device)
    model.eval()
    models_list.append(model)



## === cell 14
validation_scores = {
    "inception_resnet_v2": 0.822,
}
total_score = sum(validation_scores.values())
weights = {k: v / total_score for k, v in validation_scores.items()}

all_outputs = []
class_indices = torch.arange(5, device=device, dtype=torch.float)  # 0‑4
with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device)
        outputs = [
            weights[model_key]
            * nn.functional.softmax(model(images), dim=1).unsqueeze(0)
            for model_key, model in zip(model_paths.keys(), models_list)
        ]
        outputs = torch.cat(outputs)  # (n_models, batch, 5)
        weighted_outputs = torch.sum(outputs, dim=0)  # (batch, 5)

        preds = torch.argmax(weighted_outputs, dim=1).long()
        all_outputs.extend(preds.cpu().numpy())
final_predictions = np.array(all_outputs, dtype=int)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3076836573.py in <cell line: 0>()
      8 class_indices = torch.arange(5, device=device, dtype=torch.float)  # 0‑4
      9 with torch.no_grad():
---> 10     for images in tqdm(test_loader):
     11         images = images.to(device)
     12         outputs = [

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/4262975723.py in __getitem__(self, idx)
     11     def __getitem__(self, idx):
     12         img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
---> 13         image = Image.open(img_name).convert("RGB")
     14         if self.transform:
     15             image = self.transform(image)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/test/images_resized/b460ca9fa26f.png'

## === cell 15
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)
submission_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1898926955.py in <cell line: 0>()
      1 submission_df = pd.DataFrame(
----> 2     {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
      3 )
      4 submission_df.to_csv("submission.csv", index=False)

NameError: name 'final_predictions' is not defined
