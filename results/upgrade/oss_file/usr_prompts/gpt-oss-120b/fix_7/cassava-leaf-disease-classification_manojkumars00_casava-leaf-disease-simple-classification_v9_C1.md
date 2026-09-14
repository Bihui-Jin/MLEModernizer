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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8637050468419462

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.54335) has done: 'The changes speed up data loading and model training without altering the model architecture or training logic:  
* **Cell 13** – adds multiprocessing workers to `model.fit` so image batches are prepared in parallel, reducing epoch time.  
* **Cell 15** – replaces the per‑image prediction loop with batched loading and a single `model.predict` call (processed in reasonable batch chunks), eliminating the huge overhead of 2 700 separate forward passes while keeping the exact same predictions.'

# 9. Code solution

## === cell 0
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"



## === cell 1
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/408714542.py in <cell line: 0>()
----> 1 train_csv = pd.read_csv(train_csv_path)
      2 train_csv["label"] = train_csv["label"].astype("string")
      3 
      4 label_class = pd.read_json(label_json_path, orient="index")
      5 label_class = label_class.values.flatten().tolist()

NameError: name 'pd' is not defined

## === cell 2
train_data_label_3 = train_csv[train_csv["label"] == "3"]
train_data_label_3 = shuffle(train_data_label_3, random_state=42)
train_data_label_3 = train_data_label_3[:3000]

train_data_label_not_3 = train_csv[train_csv["label"] != "3"]

train_csv = pd.concat([train_data_label_3, train_data_label_not_3], ignore_index=True)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4242518803.py in <cell line: 0>()
----> 1 train_data_label_3 = train_csv[train_csv["label"] == "3"]
      2 train_data_label_3 = shuffle(train_data_label_3, random_state=42)
      3 train_data_label_3 = train_data_label_3[:3000]
      4 
      5 train_data_label_not_3 = train_csv[train_csv["label"] != "3"]

NameError: name 'train_csv' is not defined

## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2872813560.py in <cell line: 0>()
      1 print("Label names :")
----> 2 for i, label in enumerate(label_class):
      3     print(f" {i}. {label}")
      4 

NameError: name 'label_class' is not defined

## === cell 4
train_csv.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2594079454.py in <cell line: 0>()
----> 1 train_csv.head()
      2 

NameError: name 'train_csv' is not defined

## === cell 5
BATCH_SIZE = 64
IMG_SIZE = 320



## === cell 6
train_gen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.1, 0.9],
    shear_range=25,
    zoom_range=0.3,
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255,
    validation_split=0.15,
)

valid_gen = ImageDataGenerator(rescale=1 / 255, validation_split=0.15)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1487264624.py in <cell line: 0>()
----> 1 train_gen = ImageDataGenerator(
      2     rotation_range=360,
      3     width_shift_range=0.1,
      4     height_shift_range=0.1,
      5     brightness_range=[0.1, 0.9],

NameError: name 'ImageDataGenerator' is not defined

## === cell 7
train_generator = train_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset="training",
    seed=42,
    workers=8,
    use_multiprocessing=True,
)

valid_generator = valid_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    subset="validation",
    seed=42,
    workers=8,
    use_multiprocessing=True,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/643563707.py in <cell line: 0>()
----> 1 train_generator = train_gen.flow_from_dataframe(
      2     dataframe=train_csv,
      3     directory=images_dir_path,
      4     x_col="image_id",
      5     y_col="label",

NameError: name 'train_gen' is not defined

## === cell 8
batch = next(train_generator)
images = batch[0]
labels = batch[1]

plt.figure(figsize=(12, 9))
for i, (img, label) in enumerate(zip(images, labels)):
    plt.subplot(2, 3, i % 6 + 1)
    plt.axis("off")
    plt.imshow(img)
    plt.title(label_class[np.argmax(label)])
    if i == 15:
        break



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1362102377.py in <cell line: 0>()
----> 1 batch = next(train_generator)
      2 images = batch[0]
      3 labels = batch[1]
      4 
      5 plt.figure(figsize=(12, 9))

NameError: name 'train_generator' is not defined

## === cell 9
base = applications.InceptionResNetV2(
    include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/410515271.py in <cell line: 0>()
----> 1 base = applications.InceptionResNetV2(
      2     include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
      3 )
      4 

NameError: name 'applications' is not defined

## === cell 10
model = tf.keras.Sequential()
model.add(base)
model.add(BatchNormalization(axis=-1))
model.add(GlobalAveragePooling2D())
model.add(Dense(5, activation="softmax"))

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
    metrics=["acc"],
)
model.summary()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2290115966.py in <cell line: 0>()
----> 1 model = tf.keras.Sequential()
      2 model.add(base)
      3 model.add(BatchNormalization(axis=-1))
      4 model.add(GlobalAveragePooling2D())
      5 model.add(Dense(5, activation="softmax"))

NameError: name 'tf' is not defined

## === cell 11
def scheduler(epoch, lr):
    if epoch > 6 and epoch % 2 == 0:
        return lr / 1.5
    return lr


callback0 = tf.keras.callbacks.ModelCheckpoint(
    "./CasavaLeafDiseaseModel.h5", monitor="val_loss", save_best_only=True
)

callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1944164109.py in <cell line: 0>()
      5 
      6 
----> 7 callback0 = tf.keras.callbacks.ModelCheckpoint(
      8     "./CasavaLeafDiseaseModel.h5", monitor="val_loss", save_best_only=True
      9 )

NameError: name 'tf' is not defined

## === cell 12
pretrained_paths = [
    "../input/casavaleafdiseasemodel-tf/CasavaLeafDiseaseModel_epoch_12_acc_85.h5",
    "../input/casavaleafdiseasemodel-tf/CasavaLeafDiseaseModel.h5",
    "./CasavaLeafDiseaseModel.h5",
]
model_loaded = False
for p in pretrained_paths:
    if os.path.exists(p):
        try:
            model = tf.keras.models.load_model(p)
            print(f"Loaded pretrained model from {p}")
            model_loaded = True
            break
        except Exception as e:
            print(f"Failed to load model from {p}: {e}")

if not model_loaded:
    print("No pretrained model found; training from scratch.")
    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=15,
        callbacks=[callback0, callback1],
        verbose=2,
    )



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3169429961.py in <cell line: 0>()
      7 model_loaded = False
      8 for p in pretrained_paths:
----> 9     if os.path.exists(p):
     10         try:
     11             model = tf.keras.models.load_model(p)

NameError: name 'os' is not defined

## === cell 13
test_img_path = (
    "../input/cassava-leaf-disease-classification/test_images/2216849948.jpg"
)

img = cv2.imread(test_img_path)
if img is None:
    print("Test image not found; skipping visualization.")
else:
    resized_img = (
        cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255
    )
    plt.figure(figsize=(8, 4))
    plt.title("TEST IMAGE")
    plt.imshow(resized_img[0])



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/436708239.py in <cell line: 0>()
      3 )
      4 
----> 5 img = cv2.imread(test_img_path)
      6 if img is None:
      7     print("Test image not found; skipping visualization.")

NameError: name 'cv2' is not defined

## === cell 14
preds = []
ss = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")

test_paths = [
    os.path.join("../input/cassava-leaf-disease-classification/test_images", img_name)
    for img_name in ss.image_id
]

batch_size_pred = 256  # larger batch improves throughput without affecting results
for start in range(0, len(test_paths), batch_size_pred):
    batch_paths = test_paths[start : start + batch_size_pred]
    batch_imgs = np.empty((len(batch_paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for i, p in enumerate(batch_paths):
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Test image not found: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        batch_imgs[i] = img / 255.0
    batch_pred = model.predict(batch_imgs, verbose=0)
    preds.extend(np.argmax(batch_pred, axis=1))

my_submission = pd.DataFrame({"image_id": ss.image_id, "label": preds})
my_submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4043304440.py in <cell line: 0>()
      1 preds = []
----> 2 ss = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
      3 
      4 test_paths = [
      5     os.path.join("../input/cassava-leaf-disease-classification/test_images", img_name)

NameError: name 'pd' is not defined

## === cell 15
print("Submission File: \n---------------\n")
print(my_submission.head())

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3014216753.py in <cell line: 0>()
      1 print("Submission File: \n---------------\n")
----> 2 print(my_submission.head())

NameError: name 'my_submission' is not defined
