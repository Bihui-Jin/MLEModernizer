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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.9398686424300212

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from fastai.vision import *
import fastai
from fastai.metrics import *
from fastai import *
from os import *
import seaborn as sns
from sklearn.metrics import auc,roc_curve,accuracy_score, roc_auc_score
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
np.random.seed(42)
from glob import glob 
%matplotlib inline

## === cell 3
model_path='.'
path='/kaggle/input/histopathologic-cancer-detection/'
train_folder=f'{path}train'
test_folder=f'{path}test'
train_lbl=f'{path}train_labels.csv'

bs=64
num_workers=None 
sz=96

## === cell 5
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2223522271.py in <cell line: 0>()
      1 # Programming framework behind the scenes of NVIDIA GPU is CUDA
----> 2 print(torch.cuda.is_available())
      3 # Check if gpu is enabled
      4 print(torch.backends.cudnn.enabled)

NameError: name 'torch' is not defined

## === cell 7
df_train = pd.read_csv(train_lbl)
print(f'Number of labels {len(df_train)}')

## === cell 8
df_train['label'].value_counts(normalize=True)

## === cell 9
sns.countplot(x='label',data=df_train)

## === cell 11
cancer_cell = df_train[df_train['label']==1].head()
cancer_cell

## === cell 12
non_cancer_cell = df_train[df_train['label']==0].head()
non_cancer_cell

## === cell 13
plt.subplot(1 , 2 , 1)
img = np.asarray(plt.imread(train_folder+'/'+cancer_cell.iloc[1][0]+'.tif'))
plt.title('METASTATIC CELL TISSUE')
plt.imshow(img)

plt.subplot(1 , 2 , 2)
img = np.asarray(plt.imread(train_folder+'/'+ non_cancer_cell.iloc[1][0]+'.tif'))
plt.title('NON-METASTATIC CELL TISSUE')
plt.imshow(img)

plt.show()

## === cell 14
list = os.listdir(test_folder) # dir is your directory path
len(list)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4022005819.py in <cell line: 0>()
----> 1 list = os.listdir(test_folder) # dir is your directory path
      2 len(list)

NameError: name 'os' is not defined

## === cell 15
list = os.listdir(train_folder) # dir is your directory path
len(list)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2704481166.py in <cell line: 0>()
----> 1 list = os.listdir(train_folder) # dir is your directory path
      2 len(list)

NameError: name 'os' is not defined

## === cell 18
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=1.1,max_lighting=0.05, max_warp=0.)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/38585877.py in <cell line: 0>()
----> 1 tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=1.1,max_lighting=0.05, max_warp=0.)

NameError: name 'get_transforms' is not defined

## === cell 20
data = ImageDataBunch.from_csv(path,folder='train',valid_pct=0.3,csv_labels=train_lbl,ds_tfms=tfms, size=90, suffix='.tif',test=test_folder,bs=64)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4257084355.py in <cell line: 0>()
----> 1 data = ImageDataBunch.from_csv(path,folder='train',valid_pct=0.3,csv_labels=train_lbl,ds_tfms=tfms, size=90, suffix='.tif',test=test_folder,bs=64)

NameError: name 'ImageDataBunch' is not defined

## === cell 21
data.classes

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2117326187.py in <cell line: 0>()
----> 1 data.classes

AttributeError: module 'fastai.data' has no attribute 'classes'

## === cell 22
print(data.c, len(data.train_ds), len(data.valid_ds))

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3238321978.py in <cell line: 0>()
----> 1 print(data.c, len(data.train_ds), len(data.valid_ds))

AttributeError: module 'fastai.data' has no attribute 'c'

## === cell 24
stats=data.batch_stats()        
data.normalize(stats)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1753456537.py in <cell line: 0>()
----> 1 stats=data.batch_stats()
      2 data.normalize(stats)
      3 #data.normalize(imagenet_stats)

AttributeError: module 'fastai.data' has no attribute 'batch_stats'

## === cell 25
data.show_batch(rows=3, figsize=(8,5))

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2814136427.py in <cell line: 0>()
      1 #See the classes and labels
----> 2 data.show_batch(rows=3, figsize=(8,5))

AttributeError: module 'fastai.data' has no attribute 'show_batch'

## === cell 27
model_dir = "/kaggle/working/tmp/models/"
os.makedirs('/kaggle/working/tmp/models/')

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2677592014.py in <cell line: 0>()
      1 model_dir = "/kaggle/working/tmp/models/"
----> 2 os.makedirs('/kaggle/working/tmp/models/')

NameError: name 'os' is not defined

## === cell 29
dir(fastai.vision.models)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3352199310.py in <cell line: 0>()
      1 #fastai comes with various models
----> 2 dir(fastai.vision.models)

AttributeError: module 'fastai.vision' has no attribute 'models'

## === cell 30
learner_resnet50 = cnn_learner(data=data, base_arch=models.resnet50,model_dir=model_dir, metrics=[accuracy,error_rate], ps=0.5) #densenet201

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1609691730.py in <cell line: 0>()
      1 #create learner object by passing data bunch, specifying model architecture and metrics to use to evaluate training stats
----> 2 learner_resnet50 = cnn_learner(data=data, base_arch=models.resnet50,model_dir=model_dir, metrics=[accuracy,error_rate], ps=0.5) #densenet201

NameError: name 'cnn_learner' is not defined

## === cell 32
lr_find(learner_resnet50)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/871666596.py in <cell line: 0>()
----> 1 lr_find(learner_resnet50)

NameError: name 'lr_find' is not defined

## === cell 33
learner_resnet50.recorder.plot()

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/119903984.py in <cell line: 0>()
----> 1 learner_resnet50.recorder.plot()

NameError: name 'learner_resnet50' is not defined

## === cell 34
defaults.device = torch.device('cuda') # makes sure the gpu is used

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1569387812.py in <cell line: 0>()
----> 1 defaults.device = torch.device('cuda') # makes sure the gpu is used

NameError: name 'torch' is not defined

## === cell 36
learner_resnet50.fit_one_cycle(1, 1e-02)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3697210238.py in <cell line: 0>()
----> 1 learner_resnet50.fit_one_cycle(1, 1e-02)

NameError: name 'learner_resnet50' is not defined

## === cell 37
learner_resnet50.recorder.plot(return_fig=True)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/977101678.py in <cell line: 0>()
----> 1 learner_resnet50.recorder.plot(return_fig=True)

NameError: name 'learner_resnet50' is not defined

## === cell 38
learner_resnet50.recorder.plot_lr(show_moms=True)

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2043766682.py in <cell line: 0>()
      1 #See how the learning rate and momentum varies with the training and losses
----> 2 learner_resnet50.recorder.plot_lr(show_moms=True)

NameError: name 'learner_resnet50' is not defined

## === cell 39
learner_resnet50.recorder.plot_losses(show_grid=True)

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/944960664.py in <cell line: 0>()
----> 1 learner_resnet50.recorder.plot_losses(show_grid=True)

NameError: name 'learner_resnet50' is not defined

## === cell 40
learner_resnet50.show_results(alpha=1)

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3497689374.py in <cell line: 0>()
----> 1 learner_resnet50.show_results(alpha=1)

NameError: name 'learner_resnet50' is not defined

## === cell 41
learner_resnet50.save('stage-1',return_path=True)

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4176772983.py in <cell line: 0>()
      1 #save weights in a file
----> 2 learner_resnet50.save('stage-1',return_path=True)

NameError: name 'learner_resnet50' is not defined

## === cell 43
learner_resnet50.unfreeze()

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4231454009.py in <cell line: 0>()
      1 #Unfreeze the encoder resnet
----> 2 learner_resnet50.unfreeze()

NameError: name 'learner_resnet50' is not defined

## === cell 44
lr_find(learner_resnet50)
learner_resnet50.recorder.plot()

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1938071356.py in <cell line: 0>()
----> 1 lr_find(learner_resnet50)
      2 learner_resnet50.recorder.plot()

NameError: name 'lr_find' is not defined

## === cell 45
learner_resnet50.fit_one_cycle(1,slice(1e-06,1e-05),pct_start=0.8)

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1915486142.py in <cell line: 0>()
      1 #slice suggests is, train the initial layers at start value specified and last layer at the end value specified and interpolate for the rest of the layers
----> 2 learner_resnet50.fit_one_cycle(1,slice(1e-06,1e-05),pct_start=0.8)

NameError: name 'learner_resnet50' is not defined

## === cell 46
learner_resnet50.recorder.plot_losses()

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3758662832.py in <cell line: 0>()
----> 1 learner_resnet50.recorder.plot_losses()

NameError: name 'learner_resnet50' is not defined

## === cell 47
learner_resnet50.save('stage-2',return_path=True)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3535854174.py in <cell line: 0>()
----> 1 learner_resnet50.save('stage-2',return_path=True)

NameError: name 'learner_resnet50' is not defined

## === cell 49
interp = ClassificationInterpretation.from_learner(learner_resnet50)

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2526210968.py in <cell line: 0>()
      1 #create interpreter object
----> 2 interp = ClassificationInterpretation.from_learner(learner_resnet50)

NameError: name 'ClassificationInterpretation' is not defined

## === cell 50
interp.plot_top_losses(9,figsize=(12,12),heatmap=False)

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2724128035.py in <cell line: 0>()
      1 #Plot the biggest losses of the model
----> 2 interp.plot_top_losses(9,figsize=(12,12),heatmap=False)

NameError: name 'interp' is not defined

## === cell 51
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2743478180.py in <cell line: 0>()
----> 1 losses,idxs = interp.top_losses()
      2 len(data.valid_ds)==len(losses)==len(idxs)

NameError: name 'interp' is not defined

## === cell 52
interp.plot_confusion_matrix(figsize=(15,5))

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1863627501.py in <cell line: 0>()
----> 1 interp.plot_confusion_matrix(figsize=(15,5))

NameError: name 'interp' is not defined

## === cell 53
interp.most_confused(min_val=2)

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/54309329.py in <cell line: 0>()
      1 #To view the list of classes most misclassified as a list
      2 #Sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences
----> 3 interp.most_confused(min_val=2)

NameError: name 'interp' is not defined

## === cell 55
pred_val ,y_val = learner_resnet50.get_preds()

def auc_score(y_pred,y_true,tens=True):
    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    if tens:
        score=tensor(score)
    else:
        score=score
    return score

pred_score=auc_score(pred_val ,y_val)
pred_score

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4202023336.py in <cell line: 0>()
----> 1 pred_val ,y_val = learner_resnet50.get_preds()
      2 
      3 def auc_score(y_pred,y_true,tens=True):
      4     score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
      5     if tens:

NameError: name 'learner_resnet50' is not defined

## === cell 57
pred_score_acc=accuracy(pred_val ,y_val)
pred_score_acc

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/733306076.py in <cell line: 0>()
----> 1 pred_score_acc=accuracy(pred_val ,y_val)
      2 pred_score_acc

NameError: name 'pred_val' is not defined

## === cell 59
fpr, tpr, thresholds = roc_curve(y_val.numpy(), pred_val.numpy()[:,1], pos_label=1)
pred_score_auc = auc(fpr, tpr)
print(f'ROC area: {pred_score_auc}')

## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4012490240.py in <cell line: 0>()
----> 1 fpr, tpr, thresholds = roc_curve(y_val.numpy(), pred_val.numpy()[:,1], pos_label=1)
      2 pred_score_auc = auc(fpr, tpr)
      3 print(f'ROC area: {pred_score_auc}')

NameError: name 'y_val' is not defined

## === cell 60
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % pred_score_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2858179812.py in <cell line: 0>()
      1 plt.figure()
----> 2 plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % pred_score_auc)
      3 plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
      4 plt.xlim([-0.01, 1.0])
      5 plt.ylim([0.0, 1.01])

NameError: name 'fpr' is not defined

## === cell 62
learner_resnet50.export('/kaggle/working/tmp/models/export.pkl')

## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2540792934.py in <cell line: 0>()
----> 1 learner_resnet50.export('/kaggle/working/tmp/models/export.pkl')

NameError: name 'learner_resnet50' is not defined

## === cell 64
loaded_learner = load_learner(Path(model_dir))
loaded_learner.data.classes

## --- ERROR in cell 64, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3510026563.py in <cell line: 0>()
----> 1 loaded_learner = load_learner(Path(model_dir))
      2 loaded_learner.data.classes

NameError: name 'load_learner' is not defined

## === cell 66
img, cat = data.train_ds[0]
img.show()
print(cat)

## --- ERROR in cell 66, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/382184237.py in <cell line: 0>()
----> 1 img, cat = data.train_ds[0]
      2 img.show()
      3 print(cat)

AttributeError: module 'fastai.data' has no attribute 'train_ds'

## === cell 67
pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
print(pred_class, pred_idx,pred_probs)

## --- ERROR in cell 67, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/376113115.py in <cell line: 0>()
----> 1 pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
      2 print(pred_class, pred_idx,pred_probs)

NameError: name 'loaded_learner' is not defined

## === cell 69
img, cat = data.valid_ds[1]
img.show()
print(cat)

## --- ERROR in cell 69, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/728760486.py in <cell line: 0>()
----> 1 img, cat = data.valid_ds[1]
      2 img.show()
      3 print(cat)

AttributeError: module 'fastai.data' has no attribute 'valid_ds'

## === cell 70
pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
print(pred_class, pred_idx,pred_probs)

## --- ERROR in cell 70, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/376113115.py in <cell line: 0>()
----> 1 pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
      2 print(pred_class, pred_idx,pred_probs)

NameError: name 'loaded_learner' is not defined

## === cell 72
img = open_image(Path('../input/test-image/test_img.tif'))
pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
img.show()
targets = ['Non-Cancerous','Cancerous'] #since sequence of classes in data is as 0,1
print("Tissue cell is identified as" , targets[pred_idx] , "with probability of", float(pred_probs[pred_idx]*100))

## --- ERROR in cell 72, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3763778290.py in <cell line: 0>()
----> 1 img = open_image(Path('../input/test-image/test_img.tif'))
      2 pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
      3 img.show()
      4 targets = ['Non-Cancerous','Cancerous'] #since sequence of classes in data is as 0,1
      5 print("Tissue cell is identified as" , targets[pred_idx] , "with probability of", float(pred_probs[pred_idx]*100))

NameError: name 'open_image' is not defined

## === cell 74
loaded_learner_val = load_learner(Path(model_dir),test=ImageList.from_folder(Path(test_folder)))

## --- ERROR in cell 74, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3770620891.py in <cell line: 0>()
----> 1 loaded_learner_val = load_learner(Path(model_dir),test=ImageList.from_folder(Path(test_folder)))

NameError: name 'load_learner' is not defined

## === cell 75
pred_test ,y_test = loaded_learner_val.get_preds(ds_type=DatasetType.Test)

## --- ERROR in cell 75, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3465528997.py in <cell line: 0>()
----> 1 pred_test ,y_test = loaded_learner_val.get_preds(ds_type=DatasetType.Test)

NameError: name 'loaded_learner_val' is not defined

## === cell 77
sub=pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv').set_index('id')

## === cell 78
clean_names = np.vectorize(lambda imgname: str(imgname).split('/')[-1][:-4])
cleaned_names = clean_names(data.test_ds.items).astype(str)

## --- ERROR in cell 78, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3046999008.py in <cell line: 0>()
      1 clean_names = np.vectorize(lambda imgname: str(imgname).split('/')[-1][:-4])
----> 2 cleaned_names = clean_names(data.test_ds.items).astype(str)

AttributeError: module 'fastai.data' has no attribute 'test_ds'

## === cell 79
sub.loc[cleaned_names,'label']=pred_test.numpy()[:,1]
sub.to_csv(f'/kaggle/working/submission_{int(pred_score_auc*100)}auc.csv')

## --- ERROR in cell 79, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1644668142.py in <cell line: 0>()
----> 1 sub.loc[cleaned_names,'label']=pred_test.numpy()[:,1]
      2 sub.to_csv(f'/kaggle/working/submission_{int(pred_score_auc*100)}auc.csv')

NameError: name 'pred_test' is not defined

## === cell 80
predicted_prob_test = pd.read_csv('./submission_98auc.csv')
predicted_prob_test.head(10)

## --- ERROR in cell 80, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/957893787.py in <cell line: 0>()
----> 1 predicted_prob_test = pd.read_csv('./submission_98auc.csv')
      2 predicted_prob_test.head(10)

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

FileNotFoundError: [Errno 2] No such file or directory: './submission_98auc.csv'
