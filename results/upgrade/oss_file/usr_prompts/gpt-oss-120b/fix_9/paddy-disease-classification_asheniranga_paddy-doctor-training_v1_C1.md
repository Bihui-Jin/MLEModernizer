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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.87096

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.02998) has done: 'The fixes address the protobuf import issue, correct iterator usage, match loss weights to multiple outputs, use a proper filename for saving weights, and properly load and predict on test images while skipping sub‑directories. These changes unblock the pipeline, produce a valid CSV submission, and keep the model architecture untouched.'
- What this solution (achieved 0.0) has done: 'I fixed the mismatch between the model’s three‑output architecture and the single‑output labels supplied by the ImageDataGenerator. The model now returns only the main Inception output, compiled with a single loss/metric, so training and evaluation work correctly. I also updated the history‑plotting and test‑prediction code to use the new single‑output naming, ensuring a proper CSV submission is written.'
- What this solution (achieved 0.03267) has done: 'I fixed the file paths so the script can actually read the CSV and image folders, let `flow_from_directory` infer class names automatically, and added a robust recursive scan for test images. I also created a simple inverse‑mapping dictionary to translate predicted indices back to the original label names. These changes unblock the whole pipeline and ensure a correctly‑sized `baseline_submission.csv` is written without altering the model architecture or training logic.'
- What this solution (achieved 0.13951) has done: 'The main slowdown is the per‑epoch data loading with `ImageDataGenerator` and a small batch size, as well as the single‑image inference loop on the test set.  
- Increase the batch size to 64 to halve the number of steps per epoch.  
- Replace the slower `ImageDataGenerator` pipelines with the more efficient `image_dataset_from_directory`, scaling the images in a `map` operation.  
- Enable multiprocessing data loading in `model.fit`.  
- Batch all test images into one NumPy array and predict once, eliminating the costly per‑image model call.  
These changes keep the exact model architecture, loss, optimizer, early stopping and label handling intact while significantly reducing runtime.'

# 9. Code solution

## === cell 0
train_meta_data = "../train.csv"
train_data_dir = "../input/paddy-disease-classification/train_images"
epochs = 200  # increased epochs to give the model more training time
lr = 1e-3
valid_split = 0.2
input_size = 224
batch_size = 64  # increased batch size to speed up epoch time
classes = 10
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.CategoricalCrossentropy()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/837198087.py in <cell line: 0>()
      7 batch_size = 64  # increased batch size to speed up epoch time
      8 classes = 10
----> 9 initializer = tf.keras.initializers.HeUniform()
     10 optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
     11 loss = tf.keras.losses.CategoricalCrossentropy()

NameError: name 'tf' is not defined

## === cell 1
train_meta_path = os.path.join("input", "paddy-disease-classification", "train.csv")
train_images_dir = os.path.join("input", "paddy-disease-classification", "train_images")
if not os.path.exists(train_meta_path):
    train_meta_path = "../input/paddy-disease-classification/train.csv"
if not os.path.exists(train_images_dir):
    train_images_dir = "../input/paddy-disease-classification/train_images"

train_df = pd.read_csv(train_meta_path)
class_names = sorted(train_df["label"].unique())  # ensures exactly 10 classes

train_data = tf.keras.utils.image_dataset_from_directory(
    directory=train_images_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    batch_size=batch_size,
    image_size=(input_size, input_size),
    shuffle=True,
    seed=42,
    validation_split=valid_split,
    subset="training",
)

valid_data = tf.keras.utils.image_dataset_from_directory(
    directory=train_images_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    batch_size=batch_size,
    image_size=(input_size, input_size),
    shuffle=False,
    seed=42,
    validation_split=valid_split,
    subset="validation",
)


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label


def _scale_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


train_data = (
    train_data.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    .map(_scale_img, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

valid_data = valid_data.map(_scale_img, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/838223616.py in <cell line: 0>()
----> 1 train_meta_path = os.path.join("input", "paddy-disease-classification", "train.csv")
      2 train_images_dir = os.path.join("input", "paddy-disease-classification", "train_images")
      3 if not os.path.exists(train_meta_path):
      4     train_meta_path = "../input/paddy-disease-classification/train.csv"
      5 if not os.path.exists(train_images_dir):

NameError: name 'os' is not defined

## === cell 2
train_batch = next(iter(train_data))
valid_batch = next(iter(valid_data))
len(train_batch[0]), len(valid_batch[0])




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2668079402.py in <cell line: 0>()
----> 1 train_batch = next(iter(train_data))
      2 valid_batch = next(iter(valid_data))
      3 len(train_batch[0]), len(valid_batch[0])
      4 
      5 

NameError: name 'train_data' is not defined

## === cell 3
model = model_builder(shape=(input_size, input_size, 3), classes=classes)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4093687855.py in <cell line: 0>()
----> 1 model = model_builder(shape=(input_size, input_size, 3), classes=classes)
      2 
      3 

NameError: name 'model_builder' is not defined

## === cell 4
model.summary()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/775241066.py in <cell line: 0>()
----> 1 model.summary()
      2 
      3 

NameError: name 'model' is not defined

## === cell 5
tf.keras.utils.plot_model(model, "baseline_inception.png")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2758658000.py in <cell line: 0>()
----> 1 tf.keras.utils.plot_model(model, "baseline_inception.png")
      2 
      3 

NameError: name 'tf' is not defined

## === cell 6
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    callbacks=[early_stop],
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/847396831.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_data,
      3     validation_data=valid_data,
      4     epochs=epochs,
      5     callbacks=[early_stop],

NameError: name 'model' is not defined

## === cell 7
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["accuracy"])),
    y=history.history["accuracy"],
    label="train",
)
sns.lineplot(
    x=range(len(history.history["val_accuracy"])),
    y=history.history["val_accuracy"],
    label="validation",
)
plt.show()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/456603417.py in <cell line: 0>()
----> 1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
      3     x=range(len(history.history["accuracy"])),
      4     y=history.history["accuracy"],
      5     label="train",

NameError: name 'plt' is not defined

## === cell 8
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["loss"])),
    y=history.history["loss"],
    label="train",
)
sns.lineplot(
    x=range(len(history.history["val_loss"])),
    y=history.history["val_loss"],
    label="validation",
)
plt.show()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/152515602.py in <cell line: 0>()
----> 1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
      3     x=range(len(history.history["loss"])),
      4     y=history.history["loss"],
      5     label="train",

NameError: name 'plt' is not defined

## === cell 9
print(
    f"train score : {model.evaluate(train_data)} -- validation : {model.evaluate(valid_data)}"
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2111908320.py in <cell line: 0>()
      1 print(
----> 2     f"train score : {model.evaluate(train_data)} -- validation : {model.evaluate(valid_data)}"
      3 )
      4 
      5 

NameError: name 'model' is not defined

## === cell 10
pd.DataFrame(history.history).to_csv("history.csv", index=False)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1445596701.py in <cell line: 0>()
----> 1 pd.DataFrame(history.history).to_csv("history.csv", index=False)
      2 
      3 

NameError: name 'pd' is not defined

## === cell 11
model.save("baseline.hdf5")




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2081490051.py in <cell line: 0>()
----> 1 model.save("baseline.hdf5")
      2 
      3 

NameError: name 'model' is not defined

## === cell 12
model.save_weights("baseline_inception_weights.weights.h5")




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1303483513.py in <cell line: 0>()
----> 1 model.save_weights("baseline_inception_weights.weights.h5")
      2 
      3 

NameError: name 'model' is not defined

## === cell 13
class_indices = {name: idx for idx, name in enumerate(class_names)}
idx_to_label = {v: k for k, v in class_indices.items()}




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3131747457.py in <cell line: 0>()
----> 1 class_indices = {name: idx for idx, name in enumerate(class_names)}
      2 idx_to_label = {v: k for k, v in class_indices.items()}
      3 
      4 

NameError: name 'class_names' is not defined

## === cell 14
test_root_dir = os.path.join("input", "paddy-disease-classification", "test_images")
if not os.path.isdir(test_root_dir):
    test_root_dir = "../input/paddy-disease-classification/test_images"

test_files = [
    os.path.join(root, f)
    for root, _, files in os.walk(test_root_dir)
    for f in files
    if f.lower().endswith(".jpg")
]

test_images = []
filenames = []

for path in sorted(test_files):
    img = load_img(path, target_size=(input_size, input_size))
    arr = img_to_array(img) / 255.0
    test_images.append(arr)
    filenames.append(os.path.basename(path))

test_images_np = np.stack(test_images, axis=0)  # shape (N, H, W, 3)

preds = model.predict(test_images_np, verbose=0)
class_idxs = np.argmax(preds, axis=1)
test_preds = [[fn, idx_to_label[int(idx)]] for fn, idx in zip(filenames, class_idxs)]




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/245976618.py in <cell line: 0>()
----> 1 test_root_dir = os.path.join("input", "paddy-disease-classification", "test_images")
      2 if not os.path.isdir(test_root_dir):
      3     test_root_dir = "../input/paddy-disease-classification/test_images"
      4 
      5 test_files = [

NameError: name 'os' is not defined

## === cell 15
submission = pd.DataFrame(test_preds, columns=["image_id", "label"])
submission.to_csv("baseline_submission.csv", index=False)
submission

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/37438247.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(test_preds, columns=["image_id", "label"])
      2 submission.to_csv("baseline_submission.csv", index=False)
      3 submission

NameError: name 'pd' is not defined
