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
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
simpleitk==2.5.2
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
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Target score

1.009775180305983

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from tqdm import tqdm
from os.path import join
from os import getcwd
from pandas import read_csv, DataFrame
import SimpleITK as sitk
from torch import cuda, device
from torch import Size, Tensor, no_grad, cat, tensor, float, save, load
from torch.nn.functional import interpolate
import torch.nn as nn
import torch.jit as jit
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
from torchvision.models.densenet import densenet121
from torchvision.transforms import RandomVerticalFlip

## === cell 1
KAGGLE_DATA = r'../input/rsna-2022-cervical-spine-fracture-detection'
N_SPLITS = 5
KERNEL_TYPE = 'densenet121'
RESIZE_H = 150
RESIZE_W = 150
RESIZE_C = 50
BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
LR = 1e-5
OUT_DIM = 8
EPOCHS = 100

test_df = read_csv(join(KAGGLE_DATA, "test.csv"))
test_img_path = join(KAGGLE_DATA, "test_images")

print('test shape:', test_df.shape)

## === cell 2
class CustomDenseNet(nn.Module):
    def __init__(self, inp_c, count_labels):
        super(CustomDenseNet, self).__init__()
        self.prep_layer = nn.Conv2d(in_channels=inp_c, out_channels=3, kernel_size=3, padding='same')
        self.backbone = densenet121(num_classes=count_labels)
        self.out = nn.Softmax()

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.backbone(x)
        x = self.out(x)
        return x

class RSNADataset:
    def load_data(self):
        return self.data_path

    def __init__(self, csv_data, img_path, mode, transform=None, target_cols=None, device='cpu'):
        self.target_cols = target_cols
        self.img_path = img_path
        self.device = device
        self.dataset = csv_data
        self.mode = mode
        self.transform = transform
        self.reader = sitk.ImageSeriesReader()
        self.client_ids = self.dataset['StudyInstanceUID'].tolist()

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        self.reader.SetFileNames(self.reader.GetGDCMSeriesFileNames(join(self.img_path, self.client_ids[index])))
        client_imgs = sitk.GetArrayFromImage(
            sitk.Cast(
                sitk.RescaleIntensity(self.reader.Execute(), 0, 255),
                sitk.sitkUInt8
            )
        )
        client_imgs = Tensor(client_imgs).to(self.device)

        if self.transform:
            client_imgs = self.transform(client_imgs)

        if self.mode == "test":
            return client_imgs, self.dataset.iloc[index, :].to_dict()
        if self.mode == "train":
            return client_imgs, torch.tensor(self.dataset.iloc[index, :][self.target_cols]).float().to(self.device)

class TorchDeviceManager:
    def __init__(self):
        self.GPU_DEVICES = {}
        self.CPU_DEVICE = device('cpu')
        if cuda.is_available():
            print('TORCH: CUDA IS AVAILABLE\nGPU DEVICES:')
            for i in range(cuda.device_count()):
                self.GPU_DEVICES[i] = f"cuda:{i}"
                print(f"    {i}: {cuda.get_device_name(self.GPU_DEVICES[i])}")
        else:
            print('TORCH: CUDA IS NOT AVAILABLE')

class ValidationTransforms(jit.ScriptModule):
    def __init__(self, c, h, w):
        super().__init__()
        self.window = Size([c, h, w])

    @jit.script_method
    def forward(self, x):
        x = interpolate(x[None, None, :], size=self.window).clamp(min=0, max=255)[0, 0]
        x /= 255
        return x

class PipelineRSNA:
    def __init__(self, test_labels_path=None, test_img_path=None, test_transform=None):
        self.target_cols = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'patient_overall']
        self.test_cols = ['row_id', 'StudyInstanceUID', 'prediction_type']
        self.test_transform = test_transform
        self.target_map = {self.target_cols[i]: i for i in range(len(self.target_cols))}
        print(f"Load data from: {test_labels_path}")
        self.test_set = read_csv(test_labels_path)
        self.test_img_path = test_img_path

    def test(self, device, model_path=None):
        print(f"Test: {model_path}")
        
        model = CustomDenseNet(RESIZE_C, OUT_DIM).eval().to(device)
        model.load_state_dict(load(model_path, map_location=device))
        
        test_set = RSNADataset(
            csv_data=self.test_set,
            img_path=self.test_img_path,
            mode='test',
            transform=self.test_transform,
            target_cols=self.test_cols,
            device=device
        )
        test_loader = DataLoader(
            test_set,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
            drop_last=False
        )

        props = []
        r_names = []
        with no_grad():
            for img_batch, meta in tqdm(test_loader):
                predict = model(img_batch)
                for b_i, r_name, p in zip(range(img_batch.shape[0]), meta['row_id'], meta['prediction_type']):
                    r_names.append(r_name)
                    props.append(predict[b_i, self.target_map[p]].cpu().numpy())
        DataFrame({'row_id': r_names, 'fractured': props}).to_csv('submission.csv', index=False)


## === cell 3
transform = ValidationTransforms(RESIZE_C, RESIZE_H, RESIZE_W)

pipe = PipelineRSNA(
        test_labels_path=join(KAGGLE_DATA, 'test.csv'),
        test_img_path=join(KAGGLE_DATA, 'test_images'),
        test_transform=transform
    )

pipe.test(
    device('cuda' if cuda.is_available() else 'cpu'),
    model_path=r'../input/baseline-pretrained/densenet121_e4.pt'
)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2798166787.py in <cell line: 0>()
      8     )
      9 
---> 10 pipe.test(
     11     device('cuda' if cuda.is_available() else 'cpu'),
     12     model_path=r'../input/baseline-pretrained/densenet121_e4.pt'

/tmp/ipykernel_11/681163655.py in test(self, device, model_path)
     85         model = CustomDenseNet(RESIZE_C, OUT_DIM).eval().to(device)
     86 #         model.load_state_dict(load(model_path))
---> 87         model.load_state_dict(load(model_path, map_location=device))
     88 
     89         test_set = RSNADataset(

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '../input/baseline-pretrained/densenet121_e4.pt'
