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

0.5945

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
class get_dataset(Dataset):
    def __init__(self, data_path, csv_label_path, train, train_size, transforms=None):
        self.train = train
        self.data_path = data_path
        self.transforms = transforms

        self.label_csv = pd.read_csv(csv_label_path)
        self.label_dict = self.label_csv.set_index("image_id")["label"].to_dict()

        images_name_list = os.listdir(data_path)
        train_image, test_image = train_test_split(
            images_name_list, train_size=train_size, random_state=0
        )
        self.image_list = train_image if self.train else test_image

    def __getitem__(self, index):
        image_name = self.image_list[index]
        label = self.label_dict[image_name]
        image = Image.open(os.path.join(self.data_path, image_name))

        if self.transforms:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.image_list)


class get_test_dataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        self.image_list = os.listdir(data_path)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(os.path.join(self.data_path, image_name))
        if self.transforms:
            image = self.transforms(image)
        return image, image_name

    def __len__(self):
        return len(self.image_list)


mytransforms = transforms.Compose(
    [
        transforms.Resize(img_resize),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(),
    ]
)


train_dataset = get_dataset(train_data_path, train_csv_path, True, 0.9, mytransforms)
validation_dataset = get_dataset(
    train_data_path, train_csv_path, False, 0.9, mytransforms
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

test_dataset = get_test_dataset(test_data_path, mytransforms)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/64942315.py in <cell line: 0>()
----> 1 class get_dataset(Dataset):
      2     def __init__(self, data_path, csv_label_path, train, train_size, transforms=None):
      3         self.train = train
      4         self.data_path = data_path
      5         self.transforms = transforms

NameError: name 'Dataset' is not defined

## === cell 1
model = models.vgg16_bn(pretrained=model_pretrained)
sequential = list(model.classifier[:3])
sequential.append(nn.Linear(4096, 5))
model.classifier = nn.Sequential(*sequential)
model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5
)
criterion = nn.CrossEntropyLoss()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/14882059.py in <cell line: 0>()
----> 1 model = models.vgg16_bn(pretrained=model_pretrained)
      2 sequential = list(model.classifier[:3])
      3 sequential.append(nn.Linear(4096, 5))
      4 model.classifier = nn.Sequential(*sequential)
      5 model.to(device)

NameError: name 'models' is not defined

## === cell 2
def train(cur_epoch, dataloader, compute_grid=True):
    tq_description = "epoch %d" % cur_epoch
    tqbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))

    total_loss = 0.0
    preds_list = []
    labels_list = []

    for i, item in tqbar:
        tqbar.set_description(tq_description)
        images, labels = item
        images = images.to(device)
        labels = labels.to(device)

        model_out = model(images)
        loss = criterion(model_out, labels)
        _, preds = torch.max(model_out, 1)

        if compute_grid:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds_list += preds.tolist()
        labels_list += labels.tolist()

    return preds_list, labels_list, total_loss


def generate_submission_csv():
    tq_description = "generate csv"
    tqbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    model.load_state_dict(torch.load("model.pkl"))

    names_list = []
    preds_list = []
    for i, item in tqbar:
        tqbar.set_description(tq_description)
        images, names = item

        images = images.to(device)
        model_out = model(images)
        _, preds = torch.max(model_out, 1)

        names_list += list(names)
        preds_list += preds.tolist()

    submission = pd.DataFrame({"image_id": names_list, "label": preds_list})
    submission.to_csv("submission.csv", index=False)




## === cell 3
def compute_recall(preds_list, labels_list, class_num):
    preds_arr = np.array(preds_list)
    labels_arr = np.array(labels_list)
    recall_arr = np.zeros(class_num)

    for i in range(class_num):
        i_labels_mask = labels_arr == i
        i_preds_mask = preds_arr == i
        total_i_class_num = np.sum(i_labels_mask)
        preds_i_class_num = np.sum(i_preds_mask & i_labels_mask)
        recall_arr[i] = (
            preds_i_class_num / total_i_class_num if total_i_class_num != 0 else 0
        )
    return recall_arr


def compute_accuracy(preds_list, labels_list):
    preds_arr = np.array(preds_list)
    labels_arr = np.array(labels_list)
    return np.sum(preds_arr == labels_arr) / len(labels_arr)


def do_train(epoch):
    train_loss_list = []
    train_accuracy_list = []
    train_recall_list = []

    val_loss_list = []
    val_accuracy_list = []
    val_recall_list = []

    best_accuracy = [-1, -1]  # (epoch, value)
    train_image_num = len(train_dataset)
    val_image_num = len(validation_dataset)

    print("info:")
    print("train image number: ", train_image_num)
    print("validation image number:", val_image_num)
    print("train on: %s" % device)
    print("train epoch: %d" % epoch)

    for i in range(epoch):
        preds_list, labels_list, total_loss = train(i, train_dataloader, True)
        accuracy = compute_accuracy(preds_list, labels_list)
        recall = compute_recall(preds_list, labels_list, 5)
        train_loss_list.append(total_loss)
        train_accuracy_list.append(accuracy)
        train_recall_list.append(recall)
        print("train loss: %f" % total_loss)
        print("train accuracy: %f" % accuracy)
        print("train recall:", recall)

        preds_list, labels_list, total_loss = train(i, validation_dataloader, False)
        accuracy = compute_accuracy(preds_list, labels_list)
        recall = compute_recall(preds_list, labels_list, 5)
        val_loss_list.append(total_loss)
        val_accuracy_list.append(accuracy)
        val_recall_list.append(recall)
        print("test loss: %f" % total_loss)
        print("test accuracy: %f" % accuracy)
        print("test recall:", recall)

        if best_accuracy[1] < accuracy:
            best_accuracy[0] = i
            best_accuracy[1] = accuracy
            torch.save(model.state_dict(), "model.pkl")

    plt.figure()
    plt.plot(train_loss_list, label="train")
    plt.plot(val_loss_list, label="validation")
    plt.title("loss")
    plt.legend()

    plt.figure()
    plt.plot(train_accuracy_list, label="train")
    plt.plot(val_accuracy_list, label="validation")
    plt.title("accuracy")
    plt.legend()

    plt.figure()
    plt.plot(train_recall_list, label="train")
    plt.plot(val_recall_list, label="validation")
    plt.title("recall")
    plt.legend()




## === cell 4
do_train(20)
generate_submission_csv()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3988294012.py in <cell line: 0>()
----> 1 do_train(20)
      2 generate_submission_csv()

/tmp/ipykernel_55/3091204086.py in do_train(epoch)
     31 
     32     best_accuracy = [-1, -1]  # (epoch, value)
---> 33     train_image_num = len(train_dataset)
     34     val_image_num = len(validation_dataset)
     35 

NameError: name 'train_dataset' is not defined
