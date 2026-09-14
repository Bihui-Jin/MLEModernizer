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
Classify and localize common thoracic lung diseases and critical findings.

For each test image, you will be predicting a bounding box and class for all findings. If you predict that there are no findings, you should create a prediction of "14 1 0 0 1 1" (14 is the class ID for no finding, and this provides a one-pixel bounding box with a confidence of 1.0).

## Metric
PASCAL VOC 2010 [mean Average Precision (mAP)](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/devkit_doc_08-May-2010.pdf) at IoU > 0.4.

## Submission Format
Images in the test set may contain more than one object. For each object in a given test image, you must predict a class ID, `confidence` score, and bounding box in format `xmin ymin xmax ymax`. If you predict that there are NO objects in a given image, you should predict `14 1.0 0 0 1 1`, where `14` is the class ID for "No finding", 1.0 is the confidence, and `0 0 1 1` is a one-pixel bounding box.

The submission file should contain a header and have the following format:

```
ID,TARGET
004f33259ee4aef671c2b95d54e4be68,14 1 0 0 1 1
004f33259ee4aef671c2b95d54e4be69,11 0.5 100 100 200 200 13 0.7 10 10 20 20
etc.
```

## Dataset
The dataset comprises postero-anterior (PA) CXR scans in DICOM format.

All images were labeled for the presence of 14 critical radiographic findings as listed below:

```
0 - Aortic enlargement
1 - Atelectasis
2 - Calcification
3 - Cardiomegaly
4 - Consolidation
5 - ILD
6 - Infiltration
7 - Lung Opacity
8 - Nodule/Mass
9 - Other lesion
10 - Pleural effusion
11 - Pleural thickening
12 - Pneumothorax
13 - Pulmonary fibrosis
```

The "No finding" observation (`14`) was intended to capture the absence of all findings above.

### Files
- **train.csv** - the train set metadata, with one row for each object, including a class and a bounding box. Some images in both test and train have multiple objects.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_id` - unique image identifier
- `class_name` - the name of the class of detected object (or "No finding")
- `class_id` - the ID of the class of detected object
- `rad_id` - the ID of the radiologist that made the observation
- `x_min` - minimum X coordinate of the object's bounding box
- `y_min` - minimum Y coordinate of the object's bounding box
- `x_max` - maximum X coordinate of the object's bounding box
- `y_max` - maximum Y coordinate of the object's bounding box

# 2. Python version

3.9

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
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyYAML==6.0.3
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
            description.md (132 lines)
            sample_submission.csv (1501 lines)
            sample_submission.csv.zip (30.9 kB)
            test.zip (12.7 GB)
            train.csv (61172 lines)
            train.csv.zip (1.7 MB)
            train.zip (114.6 GB)
            test/
                00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                ... and 1498 other files
                test/
            train/
                000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                ... and 13498 other files
                train/
            vinbigdata-chest-xray-abnormalities-detection/
                description.md (132 lines)
                sample_submission.csv (1501 lines)
                ... and 5 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
                vinbigdata-chest-xray-abnormalities-detection/
        input/
            description.md (132 lines)
            sample_submission.csv (1501 lines)
            sample_submission.csv.zip (30.9 kB)
            test.zip (12.7 GB)
            train.csv (61172 lines)
            train.csv.zip (1.7 MB)
            train.zip (114.6 GB)
            test/
                00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                ... and 1498 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
            train/
                000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                ... and 13498 other files
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
            vinbigdata-chest-xray-abnormalities-detection/
                description.md (132 lines)
                sample_submission.csv (1501 lines)
                ... and 5 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
                vinbigdata-chest-xray-abnormalities-detection/
        working/
            vinbigdata-chest-xray-abnormalities-detection/
                description.md (132 lines)
                sample_submission.csv (1501 lines)
                ... and 5 other files
                test/
                    00575e3846ebd05a909d97ba59c53d30.dicom (12.9 MB)
                    0059d21bef1793fa9522e4ec8cae1a1a.dicom (9.3 MB)
                    ... and 1498 other files
                    test/
                train/
                    000434271f63a053c4128a0ba6352c7f.dicom (13.3 MB)
                    00053190460d56c53cc3e57321387478.dicom (9.7 MB)
                    ... and 13498 other files
                    train/
                vinbigdata-chest-xray-abnormalities-detection/
```

-> data/sample_submission.csv has 1500 rows and 2 columns.
The columns are: image_id, PredictionString

-> data/train.csv has 61171 rows and 8 columns.
The columns are: image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max

-> data/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv has 1500 rows and 2 columns.
The columns are: image_id, PredictionString

-> data/vinbigdata-chest-xray-abnormalities-detection/train.csv has 61171 rows and 8 columns.
The columns are: image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max

-> input/sample_submission.csv has 1500 rows and 2 columns.
The columns are: image_id, PredictionString

-> input/train.csv has 61171 rows and 8 columns.
The columns are: image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max

-> (stopped after 10 files for performance)

# 5. Target score

0.2031557757398293

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
df = pd.read_csv("../input/results-vinbin/results/tmp_debug/submission.csv")
df2 = pd.read_csv("../input/241solution/submission.csv")
df2

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2665266492.py in <cell line: 0>()
      2 import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
      3 import os
----> 4 df = pd.read_csv("../input/results-vinbin/results/tmp_debug/submission.csv")
      5 df2 = pd.read_csv("../input/241solution/submission.csv")
      6 df2

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/results-vinbin/results/tmp_debug/submission.csv'

## === cell 2

import os
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter
from typing import Any, Dict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut





def read_xray(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    im = Image.fromarray(array)
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    return im


def draw_bboxes(img, tl, br, rgb, score, label="", label_location="tl", opacity=0.1, line_thickness=0, font_scale=0.2,
                font_thickness=1):
    """ Draw bounding boxes in an image"""
    box = np.uint8(np.ones((br[1] - tl[1], br[0] - tl[0], 3)) * rgb)
    sub_combo = cv2.addWeighted(img[tl[1]:br[1], tl[0]:br[0], :], 1 - opacity, box, opacity, 1.0)
    img[tl[1]:br[1], tl[0]:br[0], :] = sub_combo
    if line_thickness > 0:
        img = cv2.rectangle(img, tuple(tl), tuple(br), rgb, line_thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_line_type = cv2.LINE_AA
        label = label.upper()
        text_width, text_height = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        label_origin = {"tl": tl, "br": br, "tr": (br[0], tl[1]), "bl": (tl[0], br[1])}[label_location]
        label_offset = {
            "tl": np.array([0, -10]), "br": np.array([-text_width, text_height + 10]),
            "tr": np.array([-text_width, -10]), "bl": np.array([0, text_height + 10])
        }[label_location]
        img = cv2.putText(img, label + "(" + str(round(score, 2)) + ")", tuple(label_origin + label_offset), font,
                          font_scale, rgb, font_thickness, font_line_type)
    return img


def show_xray(*img, title: list or str = "", axis: bool = False, size: tuple = (20, 13)):
    """ Show with mathlab plot as many X-rays than passed as argument"""
    plt.figure(figsize=size)
    for n in range(len(img)):
        plt.subplot(len(img), 2, n + 1)
        plt.axis(axis)
        plt.imshow(img[n], cmap="gray")
        if len(title) == n and type(title) == "array" :
            titre = title[n]
        elif type(title) == str:
            titre = title
        else:
            raise ValueError("number of titles don't match with the number of Xray")
        plt.title(titre)
    plt.show()


def predict_bbox(image_to_predict, predictor, resized_width=256, resized_height=256):
    """ Return predictions with labels, scores and bboxes"""
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        inputs_list = []
        img = image_to_predict.copy()
        if predictor.input_format == "RGB":
            img = img[:, :, ::-1]
        height, width = img.shape[:2]
        inputs = {"image": image, "height": height, "width": width}
        inputs_list.append(inputs)
        predictions = predictor.model(inputs_list)
    instances = predictions[0]["instances"]
    if len(instances) == 0:
        pred_classes = 14
        pred_boxes = [0, 0, 1, 1]
        pred_scores = 1.0
    else:
        fields: Dict[str, Any] = instances.get_fields()
        pred_classes, pred_scores, pred_boxes = fields["pred_classes"], fields["scores"], fields["pred_boxes"].tensor
        h_ratio, w_ratio = height / resized_height, width / resized_width
        pred_boxes[:, [0, 2]] *= w_ratio
        pred_boxes[:, [1, 3]] *= h_ratio
        pred_classes, pred_boxes, pred_scores = pred_classes.cpu().numpy(), pred_boxes.cpu().numpy(), pred_scores.cpu().numpy()
    return pred_classes, pred_boxes, pred_scores





class Xray:

    def __init__(self, path: str = "", folder: str = "", name: str = "", extension: str = "",
                 th: float = 0.25, palette: str = "icefire", predictor=False):
        if extension == "":
            self.extension = path
        else:
            self.extension = extension
        if name == "":
            self.name = path
        else:
            self.name = name
        if folder == "":
            self.folder = path
        else:
            self.folder = folder
        self.path = path
        self.image = self.extension
        self.shape = self.image.shape
        self.height, self.width = self.shape[0], self.shape[1]
        self.th = th
        self.palette = [tuple([int(x) for x in np.array(c) * (255, 255, 255)]) for c in sns.color_palette(palette, 15)]
        self.predictor = predictor
    
    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self,ext):
        if ext == 'png':
            self._image = cv2.imread(self.path)
        elif ext == "dicom":
            self._image = read_xray(self.path)
        else :
            raise ValueError("extention is not possible")
            
    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if value == "":
            if self.folder != "" and self.name != "" and self.extension != "":
                self._path = self.folder + "/" + self.name + "." + self.extension
            else:
                raise ValueError("Please provide a complete path or name, folder and extension values")
        else:
            self._path = value

    @property
    def extension(self):
        return self._extension

    @extension.setter
    def extension(self, value):
        possible_extensions = ["png", "dicom","jpg"]
        possible_extensions_txt = 'png, dicom, jpg'
        ext = value.split(".")
        if len(ext) > 1:
            ext = ext[-1]
        if ext in possible_extensions:
            self._extension = ext
        else:
            raise ValueError("Please enter a valid extension (possible extensions : " + possible_extensions_txt)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        split_value = value.split(".")
        if len(split_value) > 1:
            name = split_value[-2]
            self._name = name.split("/")[-1]
        else:
            self._name = split_value

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        split = value.split(".")
        split = "".join(split[:(len(split) - 1)])  # getting rid of the extension
        split = split.split("/")
        fold = "/".join(split[:(len(split) - 1)])  # getting rid of the name
        self._folder = fold

    @property
    def predictor(self):
        return self._predictor

    @predictor.setter
    def predictor(self, value):
        self._predictor = value
        return self._predictor

    def show(self):
        show_xray(self.image, title=self.name)

    def predict_bbox(self, resized_width: int = 256, resized_height: int = 256):
        if not self.predictor:
            raise ValueError('Predictor is missing. Please provide one using Xray.predictor = predictor')
        return predict_bbox(self.image, self.predictor, resized_width, resized_height)

    def process_prediction(self, th=False):
        if not th:
            th = self.th
        labels, boxes, scores = self.predict_bbox()
        processed_scores, processed_labels, processed_boxes = [], [], []
        if len(labels) > 1:
            count_dict = Counter(labels.tolist())
        for score, box, label in zip(scores, boxes, labels):
            score_i = score
            if int(label) == 0 and count_dict[label] != 1:
                best_score = np.max(scores[np.where(labels == label)])  # best score for aortic enlargement
                if score < best_score:
                    score_i = 0
            if int(label) == 3:
                score_i = score / 2
                if np.any(labels == 10):  # cardiomegaly + pleuresie => pas de cardiomégalie
                    score_i = 0
            if int(label) == 9:  # other lesion
                score_i = score / 1.3
            print(label + " : " + str(score) + ' ( ' + str(score_i) + ' ) ')
            processed_scores.append(score_i)
            processed_labels.append(label)
            processed_boxes.append(box)
        return processed_labels, processed_boxes, processed_scores

        def predicted_image(self, labels, boxes, scores):
            predicted_img = self.image.copy()
            nb_box = 0
            for label, box, score in labels, boxes, scores:
                if score_i > self.th:
                    predicted_img = draw_bboxes(predicted_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                                self.palette[label], score, label=self.mapping[label],
                                                label_location="tr",
                                                opacity=0.2, line_thickness=1)
                nb_box += 1
                if nb_box == 0:
                    predicted_img = draw_bboxes(predicted_img, (40, self.height - 40), (self.width - 40, self.height),
                                                self.palette[14],
                                                1 - scores[0], label="NORMAL", label_location="tl", opacity=0.2,
                                                line_thickness=1)

        return predicted_img


    
class Xray_dataset:
    
    def __init__(self,files):
        self.files = [Xray(file) for file in files]

        
    
    
    
    
    
    
    
    
    
from typing import Any
import yaml

def save_yaml(filepath: str, content: Any, width: int = 120):
    with open(filepath, "w") as f:
        yaml.dump(content, f, width=width)
    
    
    
    
    
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Union, List


@dataclass
class Flags:
    debug: bool = True
    outdir: str = "results/det"
    device: str = "cuda:0"

    imgdir_name: str = "vinbigdata-chest-xray-resized-png-256x256"
    seed: int = 111
    target_fold: int = 0  # 0~4
    label_smoothing: float = 0.0
    model_name: str = "resnet18"
    model_mode: str = "normal"  # normal, cnn_fixed supported
    epoch: int = 20
    batchsize: int = 8
    valid_batchsize: int = 16
    num_workers: int = 4
    snapshot_freq: int = 5
    ema_decay: float = 0.999  # negative value is to inactivate ema.
    scheduler_type: str = ""
    scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    scheduler_trigger: List[Union[int, str]] = field(default_factory=lambda: [1, "iteration"])
    aug_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})
    mixup_prob: float = -1.0  # Apply mixup augmentation when positive value is set.

    def update(self, param_dict: Dict) -> "Flags":
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self
    
    
    


if __name__ == "__main__":
    print('tests went good')

## === cell 3
n=5
name=df2.loc[n,"image_id"]
pred = df2.loc[n,"PredictionString"]
xray = Xray("../input/vinbigdata-chest-xray-abnormalities-detection/test/"+name+".dicom")
xray.show()

splited = pred.split(" ")
result = {"label" : [],"score" : [],'xmin' : [],"ymin" : [],"xmax" : [], "ymax" : []}

for n in range(len(splited)//6):
    result["label"].append(splited[n*6])
    result["score"].append(splited[n*6+1])
    result["xmin"].append(splited[n*6+2])
    result["ymin"].append(splited[n*6+3])
    result["xmax"].append(splited[n*6+4])
    result["ymax"].append(splited[n*6+5])
    

result_df = pd.DataFrame(result)
result_df

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2473827188.py in <cell line: 0>()
      1 n=5
----> 2 name=df2.loc[n,"image_id"]
      3 pred = df2.loc[n,"PredictionString"]
      4 xray = Xray("../input/vinbigdata-chest-xray-abnormalities-detection/test/"+name+".dicom")
      5 xray.show()

NameError: name 'df2' is not defined

## === cell 4
import seaborn as sns
palette = "icefire"
palette = [tuple([int(x) for x in np.array(c) * (255, 255, 255)]) for c in sns.color_palette(palette, 15)]
predicted = xray.image
predicted = cv2.cvtColor(predicted,cv2.COLOR_GRAY2RGB)
for n in range(len(result_df)):
    predicted = draw_bboxes(predicted,
                            tl=(int(result_df.loc[n,"xmin"]),int(result_df.loc[n,"ymin"])),
                            br=(int(result_df.loc[n,"xmax"]),int(result_df.loc[n,"ymax"])),
                            rgb=palette[int(result_df.loc[n,"label"])],
                            score=float(result_df.loc[n,"score"]),
                            label=str(result_df.loc[n,"label"]),
                            label_location="tl",
                            opacity=0.1,
                            line_thickness=0,
                            font_scale=0.2,
                            font_thickness=1)
    predicted

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3000867239.py in <cell line: 0>()
      2 palette = "icefire"
      3 palette = [tuple([int(x) for x in np.array(c) * (255, 255, 255)]) for c in sns.color_palette(palette, 15)]
----> 4 predicted = xray.image
      5 predicted = cv2.cvtColor(predicted,cv2.COLOR_GRAY2RGB)
      6 for n in range(len(result_df)):

NameError: name 'xray' is not defined

## === cell 5
show_xray(xray.image,predicted,size=(30,30))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/403883697.py in <cell line: 0>()
----> 1 show_xray(xray.image,predicted,size=(30,30))

NameError: name 'xray' is not defined

## === cell 6
TH = 0.2
from tqdm import tqdm

for i in tqdm(range(len(df2))):
    name=df2.loc[i,"image_id"]
    pred = df2.loc[i,"PredictionString"]
    splited = pred.split(" ")
    result = {"label" : [],"score" : [],'xmin' : [],"ymin" : [],"xmax" : [], "ymax" : []}
    for n in range(len(splited)//6):
        result["label"].append(splited[n*6])
        result["score"].append(splited[n*6+1])
        result["xmin"].append(splited[n*6+2])
        result["ymin"].append(splited[n*6+3])
        result["xmax"].append(splited[n*6+4])
        result["ymax"].append(splited[n*6+5])
        
    result_df = pd.DataFrame(result)
    resultSTR = ""
    for k in range(len(result_df)):
        if float(result_df.loc[k,"score"]) > TH:
            resultSTR = resultSTR + ' ' + result_df.loc[k,"label"] + ' ' + result_df.loc[k,"score"] + ' ' + result_df.loc[k,"xmin"] + ' ' + result_df.loc[k,"ymin"] + ' ' + result_df.loc[k,"xmax"] + ' ' + result_df.loc[k,"ymax"]
    resultSTR = resultSTR.strip()
    df2.loc[i,"PredictionString"] = resultSTR

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3217426204.py in <cell line: 0>()
      2 from tqdm import tqdm
      3 
----> 4 for i in tqdm(range(len(df2))):
      5     name=df2.loc[i,"image_id"]
      6     pred = df2.loc[i,"PredictionString"]

NameError: name 'df2' is not defined

## === cell 7
df2 

df2.to_csv("./submission.csv",index=False)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4282296006.py in <cell line: 0>()
----> 1 df2
      2 
      3 df2.to_csv("./submission.csv",index=False)

NameError: name 'df2' is not defined
