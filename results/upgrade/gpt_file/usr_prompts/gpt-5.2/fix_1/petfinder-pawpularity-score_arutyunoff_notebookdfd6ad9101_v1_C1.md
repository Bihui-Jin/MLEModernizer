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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.14

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
numpy==1.26.4
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

17.9695593080615

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from tqdm import tqdm

from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score
)


## === cell 2
train_path = "/kaggle/input/petfinder-pawpularity-score/train.csv"
test_path = "/kaggle/input/petfinder-pawpularity-score/test.csv"

IMAGE_ID_COL = "Id"     # поменяй, если у тебя другой столбец

TARGET_COL = "Pawpularity"         # сюда поставь имя своей целевой переменной

TASK_TYPE = "regression"      # если классификация, поставь "classification"

train_images_dir = "/kaggle/input/petfinder-pawpularity-score/train"
test_images_dir = "/kaggle/input/petfinder-pawpularity-score/test"

IMAGE_EXT = ".jpg"

submission_path = "submission.csv"


## === cell 4
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())


## === cell 6
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)


## === cell 9
import torchvision.models as models
import torch
import torch.nn as nn

WEIGHTS_PATH = "/kaggle/input/resnet50/resnet50_imagenet.pth"

resnet = models.resnet50(weights=None)  # для старой версии: pretrained=False

state_dict = torch.load(WEIGHTS_PATH, map_location=device)

resnet.load_state_dict(state_dict)


resnet.fc = nn.Identity()

resnet.to(device)

resnet.eval()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/252723391.py in <cell line: 0>()
     10 
     11 # 2. Загружаем state_dict из файла
---> 12 state_dict = torch.load(WEIGHTS_PATH, map_location=device)
     13 
     14 # 3. Кладём веса в модель

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/resnet50/resnet50_imagenet.pth'

## === cell 11
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


## === cell 13
def get_normalized_embedding(image_path):
    """
    1. Загружает картинку по пути image_path.
    2. Применяет стандартные трансформации под ResNet.
    3. Прогоняет через ResNet на GPU/CPU.
    4. Возвращает L2-нормализованный вектор (numpy, shape = (2048,)).
    """
    image = Image.open(image_path).convert("RGB")
    
    image = image_transform(image)
    
    image = image.unsqueeze(0)
    
    image = image.to(device)
    
    with torch.no_grad():
        emb_tensor = resnet(image)   # shape: [1, 2048]
    
    emb = emb_tensor.cpu().numpy().reshape(-1)  # shape: (2048,)
    
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    
    return emb


## === cell 15
train_embeddings = []

for image_id in tqdm(train[IMAGE_ID_COL], desc="Train embeddings"):
    img_name = str(image_id) + IMAGE_EXT
    img_path = os.path.join(train_images_dir, img_name)
    
    emb = get_normalized_embedding(img_path)
    train_embeddings.append(emb)

train_embeddings = np.array(train_embeddings)

print("Train embeddings shape:", train_embeddings.shape)


## === cell 17
test_embeddings = []

for image_id in tqdm(test[IMAGE_ID_COL], desc="Test embeddings"):
    img_name = str(image_id) + IMAGE_EXT
    img_path = os.path.join(test_images_dir, img_name)
    
    emb = get_normalized_embedding(img_path)
    test_embeddings.append(emb)

test_embeddings = np.array(test_embeddings)

print("Test embeddings shape:", test_embeddings.shape)


## === cell 19
emb_dim = train_embeddings.shape[1]
emb_cols = [f"f_{i}" for i in range(emb_dim)]

train_emb_df = pd.DataFrame(train_embeddings, columns=emb_cols)
test_emb_df = pd.DataFrame(test_embeddings, columns=emb_cols)

train_full = pd.concat([train.reset_index(drop=True), train_emb_df], axis=1)
test_full = pd.concat([test.reset_index(drop=True), test_emb_df], axis=1)

print("Train full shape:", train_full.shape)
print("Test full shape:", test_full.shape)


## === cell 21
mean_emb = train_embeddings.mean(axis=0)

mean_norm = np.linalg.norm(mean_emb)
if mean_norm > 0:
    mean_emb = mean_emb / mean_norm

def cosine_similarity(a, b):
    """
    Косинусная похожесть между двумя векторами a и b.
    a и b могут быть уже нормализованы, но мы перестрахуемся.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    value = np.dot(a, b) / denom
    return value

train_cos_sim = []
for emb in train_embeddings:
    sim = cosine_similarity(emb, mean_emb)
    train_cos_sim.append(sim)

train_full["cos_sim_to_mean"] = train_cos_sim

test_cos_sim = []
for emb in test_embeddings:
    sim = cosine_similarity(emb, mean_emb)
    test_cos_sim.append(sim)

test_full["cos_sim_to_mean"] = test_cos_sim

print("Added feature cos_sim_to_mean")


## === cell 23
embedding_features = []
for col in train_full.columns:
    if col.startswith("f_"):
        embedding_features.append(col)

numeric_features = [
]

binary_features = [
    'Subject Focus',
    'Eyes',
    'Face',
    'Near',
    'Action',
    'Accessory',
    'Group',
    'Collage',
    'Human',
    'Occlusion',
    'Info',
    'Blur'
]

distance_features = [
    "cos_sim_to_mean"  # убери, если не использовал блок с косинусом
]

categorical_features = [
]

feature_cols = []
feature_cols.extend(embedding_features)
feature_cols.extend(numeric_features)
feature_cols.extend(binary_features)
feature_cols.extend(distance_features)
feature_cols.extend(categorical_features)

print("Всего фич:", len(feature_cols))
print("Первые 10 фич:", feature_cols[:10])


## === cell 25
cat_feature_indices = []

for cat_col in categorical_features:
    if cat_col in feature_cols:
        idx = feature_cols.index(cat_col)
        cat_feature_indices.append(idx)

print("Категориальные столбцы:", categorical_features)
print("Индексы категориальных столбцов:", cat_feature_indices)


## === cell 27
X = train_full[feature_cols]
y = train_full[TARGET_COL]

X_test = test_full[feature_cols]

if TASK_TYPE == "classification":
    stratify_param = y
else:
    stratify_param = None

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=stratify_param
)

print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)


## === cell 29
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_feature_indices
)

valid_pool = Pool(
    data=X_valid,
    label=y_valid,
    cat_features=cat_feature_indices
)

test_pool = Pool(
    data=X_test,
    cat_features=cat_feature_indices
)


## === cell 32
if TASK_TYPE == "regression":
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="GPU",   # использование GPU
        devices="0",
        verbose=100
    )
    
    model.fit(train_pool, eval_set=valid_pool)
    
    y_pred_valid = model.predict(valid_pool)
    
    rmse = mean_squared_error(y_valid, y_pred_valid, squared=False)
    mae = mean_absolute_error(y_valid, y_pred_valid)
    
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE:  {mae:.4f}")


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/2387302493.py in <cell line: 0>()
     11     )
     12 
---> 13     model.fit(train_pool, eval_set=valid_pool)
     14 
     15     # Оценка

/usr/local/lib/python3.11/dist-packages/catboost/core.py in fit(self, X, y, cat_features, text_features, embedding_features, graph, sample_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   5871         if 'loss_function' in params:
   5872             CatBoostRegressor._check_is_compatible_loss(params['loss_function'])
-> 5873         return self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline,
   5874                          use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description,
   5875                          verbose_eval, metric_period, silent, early_stopping_rounds,

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _fit(self, X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   2408 
   2409             with plot_wrapper(plot, plot_file, 'Training plots', [_get_train_dir(self.get_params())]):
-> 2410                 self._train(
   2411                     train_pool,
   2412                     train_params["eval_sets"],

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _train(self, train_pool, test_pool, params, allow_clear_pool, init_model)
   1788 
   1789     def _train(self, train_pool, test_pool, params, allow_clear_pool, init_model):
-> 1790         self._object._train(train_pool, test_pool, params, allow_clear_pool, init_model._object if init_model else None)
   1791         self._set_trained_model_attributes()
   1792 

_catboost.pyx in _catboost._CatBoost._train()

_catboost.pyx in _catboost._CatBoost._train()

CatBoostError: catboost/cuda/cuda_lib/cuda_base.h:281: CUDA error 35: CUDA driver version is insufficient for CUDA runtime version

## === cell 36
if TASK_TYPE == "regression":
    test_pred = model.predict(test_pool)

elif TASK_TYPE == "classification":
    test_pred = model.predict_proba(test_pool)[:, 1]

print("Test predictions shape:", test_pred.shape)


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/3778040216.py in <cell line: 0>()
      1 if TASK_TYPE == "regression":
----> 2     test_pred = model.predict(test_pool)
      3 
      4 elif TASK_TYPE == "classification":
      5     # Для сабмита обычно полезны либо вероятности, либо классы.

/usr/local/lib/python3.11/dist-packages/catboost/core.py in predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type)
   5922         if prediction_type is None:
   5923             prediction_type = self._get_default_prediction_type()
-> 5924         return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict', task_type)
   5925 
   5926     def staged_predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, parent_method_name, task_type)
   2618         if verbose is None:
   2619             verbose = False
-> 2620         data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
   2621         self._validate_prediction_type(prediction_type)
   2622 

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _process_predict_input_data(self, data, parent_method_name, thread_count, label)
   2594     def _process_predict_input_data(self, data, parent_method_name, thread_count, label=None):
   2595         if not self.is_fitted() or self.tree_count_ is None:
-> 2596             raise CatBoostError(("There is no trained model to use {}(). "
   2597                                  "Use fit() to train model. Then use this method.").format(parent_method_name))
   2598         is_single_object = _is_data_single_object(data)

CatBoostError: There is no trained model to use predict(). Use fit() to train model. Then use this method.

## === cell 38
submission = pd.DataFrame()
submission["Id"] = test[IMAGE_ID_COL]      # колонка id
submission["Pawpularity"] = test_pred       # колонка с предсказаниями

print(submission.head())

submission.to_csv(submission_path, index=False)
print("Saved submission to:", submission_path)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/533371474.py in <cell line: 0>()
      1 submission = pd.DataFrame()
      2 submission["Id"] = test[IMAGE_ID_COL]      # колонка id
----> 3 submission["Pawpularity"] = test_pred       # колонка с предсказаниями
      4 
      5 print(submission.head())

NameError: name 'test_pred' is not defined
