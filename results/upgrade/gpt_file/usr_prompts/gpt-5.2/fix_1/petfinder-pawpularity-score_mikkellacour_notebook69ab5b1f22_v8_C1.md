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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

18.70930197642729

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
image_folder_test = r'/kaggle/input/petfinder-pawpularity-score/test'
csv_file_test = r'/kaggle/input/petfinder-pawpularity-score/test.csv'
model_path = r'/kaggle/input/resnet101-1/tensorflow2/true/1/resnet101_model.h5'

df_test = pd.read_csv(csv_file_test)


pretrained_model = load_model(model_path)


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size) # Use Keras preprocessing
    image = img_to_array(image)
    image = preprocess_input(image)  # Preprocess using ResNet101
    return image


image_features_test = []
features_test = []

for _, row in df_test.iterrows():
    image_name_test = row['Id']  # Coloumn with names
    image_path_test = os.path.join(image_folder_test, f'{image_name_test}.jpg') # Save path to img
    
    if os.path.exists(image_path_test): # check if image exists
        image_test = load_and_preprocess_image(image_path_test) # Use function
        image_test = np.expand_dims(image_test, axis=0)  # Add batch dimension
        
        extracted_features_test = pretrained_model.predict(image_test)
        image_features_test.append(extracted_features_test.flatten()) 
        
        feature_test = row[1:13].values.astype(np.float32)
        
        features_test.append(feature_test)
image_features_test = np.array(image_features_test)
features_test = np.array(features_test)


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1344878314.py in <cell line: 0>()
      9 
     10 
---> 11 pretrained_model = load_model(model_path)
     12 
     13 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/input/resnet101-1/tensorflow2/true/1/resnet101_model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 2
image_features = np.load('/kaggle/input/preprocesseddata/image_features101.npy')
features = np.load('/kaggle/input/preprocesseddata/features101.npy')
responses = np.load('/kaggle/input/preprocesseddata/responses101.npy')
print("Data has been loaded.")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3317263719.py in <cell line: 0>()
      1 # Loading data, as saved, only done if not first run
----> 2 image_features = np.load('/kaggle/input/preprocesseddata/image_features101.npy')
      3 features = np.load('/kaggle/input/preprocesseddata/features101.npy')
      4 responses = np.load('/kaggle/input/preprocesseddata/responses101.npy')
      5 print("Data has been loaded.")

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/preprocesseddata/image_features101.npy'

## === cell 3

X_images_train, X_images_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
    image_features, features, responses, test_size=0.2, random_state=42
)

image_input = Input(shape=(image_features.shape[1],))
x = Dense(256, activation='relu')(image_input)  

feature_input = Input(shape=(12,))
y = Dense(64, activation='relu')(feature_input)

combined = Concatenate()([x, y])
z = Dense(256, activation='relu')(combined)
z = Dropout(0.2)(z)
z = Dense(512, activation='relu')(z)
z = Dropout(0.2)(z)
z = Dense(128, activation='relu')(z)
z = Dropout(0.2)(z)
z = Dense(48, activation='relu')(z)
z = Dropout(0.2)(z)
z = Dense(10, activation='relu')(z)
z = Dropout(0.3)(z)
output = Dense(1, activation='sigmoid')(z) 

batch_size = 70
epochs = 1000
early_stop = EarlyStopping(
    monitor="val_loss", # Important to monitor val_loss
    min_delta=0.0000001, # Any reduction is good
    patience= 20, # In this case 20 patience is plenty
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,  # Secures that the model restores best weights
    start_from_epoch=3, # Start from epoch > 1 to remove possible errors
)

y_train_100 = y_train/100
y_test_100 = y_test/100
model = Model(inputs=[image_input, feature_input], outputs=output)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(
    [X_images_train, X_features_train], y_train_100, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_split=0.2, # Validation to use previous mentioned stuff
    shuffle=True, # Shuffle data each epoch
    callbacks=[early_stop] # Use defined callback
)

loss, mae = model.evaluate([X_images_test, X_features_test], y_test_100)
print(f'Test Loss: {loss}, Test MAE: {mae}')


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2737817713.py in <cell line: 0>()
      1 # Split the data into training and testing sets for features and images respectively
      2 X_images_train, X_images_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
----> 3     image_features, features, responses, test_size=0.2, random_state=42
      4 )
      5 

NameError: name 'image_features' is not defined

## === cell 4

predictions = model.predict([X_images_test, X_features_test])

plt.figure(figsize=(8, 8))
plt.scatter(y_test, predictions*100, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=2, color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.grid(True)
plt.show()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1482253877.py in <cell line: 0>()
      1 # Generate predictions
----> 2 predictions = model.predict([X_images_test, X_features_test])
      3 
      4 # Plot actual values vs. predictions of test data
      5 plt.figure(figsize=(8, 8))

NameError: name 'model' is not defined

## === cell 5
print("RMSE:", (np.sqrt(loss)*100))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1865486461.py in <cell line: 0>()
----> 1 print("RMSE:", (np.sqrt(loss)*100))

NameError: name 'loss' is not defined

## === cell 6
prediction = model.predict([image_features_test, features_test])
prediction *= 100
results_df = pd.DataFrame({
    'Id': df_test['Id'],  # Names
    'Pawpularity': prediction.flatten() 
})
results_df.to_csv(('submission.csv'), index=False)
print("Results saved to 'submission.csv'")


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1883005444.py in <cell line: 0>()
      1 # Model prediction on actual test data
----> 2 prediction = model.predict([image_features_test, features_test])
      3 prediction *= 100
      4 # Save the results to a pd.DataFrame
      5 results_df = pd.DataFrame({

NameError: name 'model' is not defined
