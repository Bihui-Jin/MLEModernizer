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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9241562849155004

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

from fastai import *
from fastai.vision import *
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import scipy
import skimage
import skimage.io


## === cell 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3906662039.py in <cell line: 0>()
----> 1 device = 'cuda' if torch.cuda.is_available() else 'cpu'

NameError: name 'torch' is not defined

## === cell 2
!ls /kaggle/input/plant-pathology-2020-different-size-images/images

## === cell 3
path = Path('/kaggle/input/plant-pathology-2020-fgvc7/')

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1291748042.py in <cell line: 0>()
----> 1 path = Path('/kaggle/input/plant-pathology-2020-fgvc7/')

NameError: name 'Path' is not defined

## === cell 4
train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')
sample_df = pd.read_csv(path/'sample_submission.csv')

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/900277673.py in <cell line: 0>()
----> 1 train_df = pd.read_csv(path/'train.csv')
      2 test_df = pd.read_csv(path/'test.csv')
      3 sample_df = pd.read_csv(path/'sample_submission.csv')

NameError: name 'path' is not defined

## === cell 5
test_df['image_id'] = test_df['image_id'] + '.jpg'
train_df['image_id'] = train_df['image_id'] + '.jpg'

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3120717176.py in <cell line: 0>()
----> 1 test_df['image_id'] = test_df['image_id'] + '.jpg'
      2 train_df['image_id'] = train_df['image_id'] + '.jpg'

NameError: name 'test_df' is not defined

## === cell 6
train_df.head()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2804120662.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 7
def get_label(row):
    if row.healthy == 1:
        return 'healthy'
    elif row.rust == 1:
        return 'rust'
    elif row.scab == 1:
        return 'scab'
    else:
        return 'multiple_diseases'

## === cell 8
train_df['label'] = train_df.apply(get_label, axis=1)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/466093042.py in <cell line: 0>()
----> 1 train_df['label'] = train_df.apply(get_label, axis=1)

NameError: name 'train_df' is not defined

## === cell 9
train_df = train_df[['image_id', 'label']]

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/236978214.py in <cell line: 0>()
----> 1 train_df = train_df[['image_id', 'label']]

NameError: name 'train_df' is not defined

## === cell 10
train_df.head()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2804120662.py in <cell line: 0>()
----> 1 train_df.head()

NameError: name 'train_df' is not defined

## === cell 11
c = Counter(train_df.label), len(train_df)
c

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3453663945.py in <cell line: 0>()
----> 1 c = Counter(train_df.label), len(train_df)
      2 c

NameError: name 'Counter' is not defined

## === cell 12
id_label = list(enumerate(train_df.label.tolist()))
random.seed(100)
train_sample_per_class = {'scab': 532, 'multiple_diseases': 71, 'healthy': 476, 'rust': 542}
val_sample_per_class = {'scab': 60, 'multiple_diseases': 20, 'healthy': 40, 'rust': 80}
chose = lambda k: list(map(lambda x: x[0], random.sample(list(filter(lambda x: x[1] == k, id_label)), train_sample_per_class[k] + val_sample_per_class[k]))) 
scab_chosen = chose('scab')
multiple_diseases_chosen = chose('multiple_diseases')
healthy_chosen = chose('healthy')
rust_chosen = chose('rust')

train_idx = scab_chosen[-train_sample_per_class['scab']:] + multiple_diseases_chosen[-train_sample_per_class['multiple_diseases']:] + healthy_chosen[-train_sample_per_class['healthy']:] + rust_chosen[-train_sample_per_class['rust']:]
val_idx = scab_chosen[:val_sample_per_class['scab']] + multiple_diseases_chosen[:val_sample_per_class['multiple_diseases']] + healthy_chosen[:val_sample_per_class['healthy']] + rust_chosen[:val_sample_per_class['rust']]
random.shuffle(train_idx)
random.shuffle(val_idx)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2929855291.py in <cell line: 0>()
----> 1 id_label = list(enumerate(train_df.label.tolist()))
      2 random.seed(100)
      3 train_sample_per_class = {'scab': 532, 'multiple_diseases': 71, 'healthy': 476, 'rust': 542}
      4 val_sample_per_class = {'scab': 60, 'multiple_diseases': 20, 'healthy': 40, 'rust': 80}
      5 chose = lambda k: list(map(lambda x: x[0], random.sample(list(filter(lambda x: x[1] == k, id_label)), train_sample_per_class[k] + val_sample_per_class[k])))

NameError: name 'train_df' is not defined

## === cell 13
def plot_image(image_id, img_size, axis):
    images_path = '/kaggle/input/plant-pathology-2020-different-size-images-crop/images/images-' + str(img_size)
    img = skimage.io.imread(images_path + '/' + image_id)
    axis.imshow(img)


## === cell 14
def get_img_id_from_idx(idx):
    return train_df.iloc[idx].image_id

## === cell 15
def plot_ten_images(images, title, img_size):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(40, 20))
    fig.suptitle(title, fontsize=48)
    for i in range(10):
        plot_image(get_img_id_from_idx(images[i]), img_size, axes[i // 5][i % 5])
    fig.tight_layout()

## === cell 21
def create_databunch_from_img_size(img_size, bs):
    tfms = get_transforms(flip_vert=True,
                      max_rotate=None,
                      max_lighting=None,
                      max_zoom=0.0,
                      max_warp=None)
    p = 0.25
    tfms = ([dihedral(),
             brightness(change=(0.25, 0.75), p=p),
             contrast(scale=(0.80, 1.25), p=p),
             rotate(degrees=(-45.0, 45.0), p=p),
             skew(direction=(0, 7), magnitude=2.0, p=p),
             symmetric_warp(magnitude=(-0.3, 0.3), p=p),
             squish(scale=(0.75, 2.0), p=p),
             zoom(scale=(0.90, 1.10), p=p),
            ], [])
    
    images_path = '/kaggle/input/plant-pathology-2020-different-size-images-crop/images/images-' + str(img_size)

    test_data = ImageList.from_df(test_df, images_path)

    src = (ImageList.from_df(train_df, images_path)
           .split_by_idxs(train_idx, val_idx)
           .label_from_df()
           .add_test(test_data))

    train_data = (src
                  .transform(tfms, padding_mode='zeros')
                  .databunch(bs=bs, num_workers=4)
                  .normalize(imagenet_stats))
    return train_data

## === cell 22
img_size = 64
train_data = create_databunch_from_img_size(img_size, 32)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3876938640.py in <cell line: 0>()
      1 img_size = 64
----> 2 train_data = create_databunch_from_img_size(img_size, 32)

/tmp/ipykernel_11/206775422.py in create_databunch_from_img_size(img_size, bs)
      1 def create_databunch_from_img_size(img_size, bs):
----> 2     tfms = get_transforms(flip_vert=True,
      3                       max_rotate=None,
      4                       max_lighting=None,
      5                       max_zoom=0.0,

NameError: name 'get_transforms' is not defined

## === cell 23
train_data.show_batch()

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/603274633.py in <cell line: 0>()
----> 1 train_data.show_batch()

NameError: name 'train_data' is not defined

## === cell 24
train_data.show_batch(ds_type=DatasetType.Valid)

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3773045653.py in <cell line: 0>()
----> 1 train_data.show_batch(ds_type=DatasetType.Valid)

NameError: name 'train_data' is not defined

## === cell 25
class ROCAUCScore(Callback):
    
    def on_epoch_begin(self, **kwargs):
        self.targets, self.preds = [], []
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        self.targets.extend(last_target.tolist())
        self.preds.extend(list(map(scipy.special.softmax, last_output.tolist())))
    
    def on_epoch_end(self, last_metrics, **kwargs):
        sc = roc_auc_score(self.targets, self.preds, multi_class='ovr')
        return add_metrics(last_metrics, sc)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/94711744.py in <cell line: 0>()
----> 1 class ROCAUCScore(Callback):
      2 
      3     def on_epoch_begin(self, **kwargs):
      4         self.targets, self.preds = [], []
      5 

NameError: name 'Callback' is not defined

## === cell 26
class StopTrainingAtEpoch(Callback):
    
    def __init__(self, learn, n):
        self.learn = learn
        self.n = n
    
    def on_epoch_end(self, epoch, **kwargs):
        if epoch == self.n:
            self.learn.save('model-' + str(self.n))
            return {'stop_training': True}

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2544285340.py in <cell line: 0>()
----> 1 class StopTrainingAtEpoch(Callback):
      2 
      3     def __init__(self, learn, n):
      4         self.learn = learn
      5         self.n = n

NameError: name 'Callback' is not defined

## === cell 27
ce_weight = torch.tensor([1.0, 4.0, 1.0, 1.0]).to(device)
print(ce_weight)
ce_weight=None
learn = cnn_learner(train_data, models.resnet18, metrics=[ROCAUCScore(), accuracy, error_rate, Precision(average='macro'), Recall(average='macro'), FBeta(beta=1.0, average='macro')], 
                    loss_func=CrossEntropyFlat(reduction='mean', weight=ce_weight), ps=[0.5, 0.5, 0.5], wd=0.01,
                    path='/kaggle/working', callback_fns=[ShowGraph])#, partial(AccumulateScheduler, n_step=128)])


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1531470450.py in <cell line: 0>()
      1 # ce_weight = torch.tensor([1.75, 4.0, 2.0, 1.75]).to(device)
      2 # ce_weight = torch.tensor([3, 20, 3.5, 3]).to(device)
----> 3 ce_weight = torch.tensor([1.0, 4.0, 1.0, 1.0]).to(device)
      4 # ce_weight /= ce_weight.sum()
      5 print(ce_weight)

NameError: name 'torch' is not defined

## === cell 28
learn.load('/kaggle/input/plant-pathology-fgvc7-fastai/models/stage-02');


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3884541135.py in <cell line: 0>()
----> 1 learn.load('/kaggle/input/plant-pathology-fgvc7-fastai/models/stage-02');
      2 # learn.unfreeze()

NameError: name 'learn' is not defined

## === cell 29
len(learn.data.train_dl), len(train_data.train_dl)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1078873439.py in <cell line: 0>()
----> 1 len(learn.data.train_dl), len(train_data.train_dl)

NameError: name 'learn' is not defined

## === cell 30
learn.lr_find(start_lr=1e-7, end_lr=1e2, num_it=len(learn.data.train_dl)+1, stop_div=False)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4215449507.py in <cell line: 0>()
----> 1 learn.lr_find(start_lr=1e-7, end_lr=1e2, num_it=len(learn.data.train_dl)+1, stop_div=False)

NameError: name 'learn' is not defined

## === cell 31
learn.recorder.plot(suggestion=True, skip_end=15, skip_start=10)

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3150251252.py in <cell line: 0>()
----> 1 learn.recorder.plot(suggestion=True, skip_end=15, skip_start=10)

NameError: name 'learn' is not defined

## === cell 32
img_size = 128
train_data = create_databunch_from_img_size(img_size, 32)
learn.data = train_data

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2156022497.py in <cell line: 0>()
      1 img_size = 128
----> 2 train_data = create_databunch_from_img_size(img_size, 32)
      3 learn.data = train_data

/tmp/ipykernel_11/206775422.py in create_databunch_from_img_size(img_size, bs)
      1 def create_databunch_from_img_size(img_size, bs):
----> 2     tfms = get_transforms(flip_vert=True,
      3                       max_rotate=None,
      4                       max_lighting=None,
      5                       max_zoom=0.0,

NameError: name 'get_transforms' is not defined

## === cell 35
learn.validate(learn.data.train_dl)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1618581637.py in <cell line: 0>()
----> 1 learn.validate(learn.data.train_dl)

NameError: name 'learn' is not defined

## === cell 36
learn.validate(learn.data.valid_dl)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1303941282.py in <cell line: 0>()
----> 1 learn.validate(learn.data.valid_dl)

NameError: name 'learn' is not defined

## === cell 37
img_size = 256
train_data = create_databunch_from_img_size(img_size, 32)
learn.data = train_data

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4007225288.py in <cell line: 0>()
      1 img_size = 256
----> 2 train_data = create_databunch_from_img_size(img_size, 32)
      3 learn.data = train_data

/tmp/ipykernel_11/206775422.py in create_databunch_from_img_size(img_size, bs)
      1 def create_databunch_from_img_size(img_size, bs):
----> 2     tfms = get_transforms(flip_vert=True,
      3                       max_rotate=None,
      4                       max_lighting=None,
      5                       max_zoom=0.0,

NameError: name 'get_transforms' is not defined

## === cell 39
learn.save('stage-02')

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/695719991.py in <cell line: 0>()
----> 1 learn.save('stage-02')

NameError: name 'learn' is not defined

## === cell 40
learn.validate(learn.data.train_dl)

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1618581637.py in <cell line: 0>()
----> 1 learn.validate(learn.data.train_dl)

NameError: name 'learn' is not defined

## === cell 41
learn.validate(learn.data.valid_dl)

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1303941282.py in <cell line: 0>()
----> 1 learn.validate(learn.data.valid_dl)

NameError: name 'learn' is not defined

## === cell 47
learn.validate(learn.data.train_dl)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1618581637.py in <cell line: 0>()
----> 1 learn.validate(learn.data.train_dl)

NameError: name 'learn' is not defined

## === cell 48
learn.validate(learn.data.valid_dl)

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1303941282.py in <cell line: 0>()
----> 1 learn.validate(learn.data.valid_dl)

NameError: name 'learn' is not defined

## === cell 50
learn.recorder.plot_lr()

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2832237492.py in <cell line: 0>()
----> 1 learn.recorder.plot_lr()

NameError: name 'learn' is not defined

## === cell 51
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3614389671.py in <cell line: 0>()
----> 1 interp = ClassificationInterpretation.from_learner(learn)
      2 losses,idxs = interp.top_losses()

NameError: name 'ClassificationInterpretation' is not defined

## === cell 52
interp.plot_top_losses(16, figsize=(25,25), heatmap=False)

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2670020190.py in <cell line: 0>()
----> 1 interp.plot_top_losses(16, figsize=(25,25), heatmap=False)

NameError: name 'interp' is not defined

## === cell 53
interp.plot_top_losses(16, figsize=(25,25), heatmap=True)

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1253714957.py in <cell line: 0>()
----> 1 interp.plot_top_losses(16, figsize=(25,25), heatmap=True)

NameError: name 'interp' is not defined

## === cell 54
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)

## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3605952444.py in <cell line: 0>()
----> 1 interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)

NameError: name 'interp' is not defined

## === cell 55
interp.most_confused()

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3610672563.py in <cell line: 0>()
----> 1 interp.most_confused()

NameError: name 'interp' is not defined

## === cell 56
valid_tta_preds, y, losses = learn.TTA(scale=1.10, with_loss=True)

## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1993836494.py in <cell line: 0>()
----> 1 valid_tta_preds, y, losses = learn.TTA(scale=1.10, with_loss=True)

NameError: name 'learn' is not defined

## === cell 57
accuracy(valid_tta_preds, y)

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3884712873.py in <cell line: 0>()
----> 1 accuracy(valid_tta_preds, y)

NameError: name 'accuracy' is not defined

## === cell 58
confusion_matrix(y.cpu().numpy(), valid_tta_preds.argmax(1).cpu().numpy())

## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1465874589.py in <cell line: 0>()
----> 1 confusion_matrix(y.cpu().numpy(), valid_tta_preds.argmax(1).cpu().numpy())

NameError: name 'y' is not defined

## === cell 63
learn.export('model.pkl')

## --- ERROR in cell 63, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3860534495.py in <cell line: 0>()
----> 1 learn.export('model.pkl')

NameError: name 'learn' is not defined

## === cell 64
preds, y = learn.get_preds(DatasetType.Test) 

## --- ERROR in cell 64, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/462199569.py in <cell line: 0>()
----> 1 preds, y = learn.get_preds(DatasetType.Test)

NameError: name 'learn' is not defined

## === cell 65
sample_df.iloc[:,1:] = preds.numpy()
sample_df.to_csv('submission.csv', index=False)

## --- ERROR in cell 65, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1638919863.py in <cell line: 0>()
----> 1 sample_df.iloc[:,1:] = preds.numpy()
      2 sample_df.to_csv('submission.csv', index=False)

NameError: name 'preds' is not defined

## === cell 66
sample_df.head()

## --- ERROR in cell 66, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4042194153.py in <cell line: 0>()
----> 1 sample_df.head()

NameError: name 'sample_df' is not defined
