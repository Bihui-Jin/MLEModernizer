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

3.8

# 3. Installed packages

numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        input/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
            test_images/
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
            train_images/
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> input/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> working/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8747400261442159

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -0.241) has done: 'I added a missing dataset class (`eye_dataset_orl`) that loads the test images, applies the preprocessing defined earlier, and returns each image together with its ID code. I also renumbered the cells to start at 1 and placed the new class definition after the image‑processing utilities, so the main script can instantiate it without errors. This fix enables the script to run end‑to‑end and generate a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
name_file = "../input/aptos2019-blindness-detection/test.csv"
csv_file = csv.reader(open(name_file, "r"))
content = []
for line in csv_file:
    content.append(line[0] + ".png")
content = content[1:]




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/181248057.py in <cell line: 0>()
      1 name_file = "../input/aptos2019-blindness-detection/test.csv"
----> 2 csv_file = csv.reader(open(name_file, "r"))
      3 content = []
      4 for line in csv_file:
      5     content.append(line[0] + ".png")

NameError: name 'csv' is not defined

## === cell 1
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type
        resnet50 = torchvision.models.densenet201(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-1])
        self.feature_dim = 1920
        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.classifiers = nn.Linear(
                in_features=self.feature_dim, out_features=num_classes
            )
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            self.cal_score = nn.Linear(in_features=num_classes, out_features=1)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x1):
        x = self.base(x1)
        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
        else:
            ys = x
        return ys




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/32514156.py in <cell line: 0>()
----> 1 class Baseline_single(nn.Module):
      2     def __init__(self, num_classes, loss_type="single BCE", **kwargs):
      3         super(Baseline_single, self).__init__()
      4         self.loss_type = loss_type
      5         resnet50 = torchvision.models.densenet201(pretrained=True)

NameError: name 'nn' is not defined

## === cell 2
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def change_size(image):
    b = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    print(binary_image.shape)  # 改为单通道

    x = binary_image.shape[0]
    print("高度x=", x)
    y = binary_image.shape[1]
    print("宽度y=", y)
    edges_x = []
    edges_y = []

    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)  # 左边界
    right = max(edges_x)  # 右边界
    width = right - left  # 宽度

    bottom = min(edges_y)  # 底部
    top = max(edges_y)  # 顶部
    height = top - bottom  # 高度

    pre1_picture = image[left : left + width, bottom : bottom + height]  # 图片截取

    return pre1_picture  # 返回图片数据


def crop_image1(img, tol=7):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (492, 492))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def findCircle(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_img = image[:, :, 0]
    s_img = image[:, :, 1]
    v_img = image[:, :, 2]
    height, width = v_img.shape
    mask_v_a = cv2.adaptiveThreshold(
        v_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        int(max(height, width) / 16) * 2 + 1,
        1,
    )

    ratio = 128 / min(height, width)
    msk = cv2.resize(
        mask_v_a,
        (int(width * ratio), int(height * ratio)),
        interpolation=cv2.INTER_CUBIC,
    )
    h, w = msk.shape
    msk_expand = np.zeros((3 * h, 3 * w), np.uint8)
    msk_expand[h : 2 * h, w : 2 * w] = msk
    long_edge = max(h, w)
    r0 = round(0.3 * long_edge)
    r1 = round(0.7 * long_edge)
    circles = cv2.HoughCircles(
        msk_expand,
        cv2.HOUGH_GRADIENT,
        1,
        90,
        param1=50,
        param2=5,
        minRadius=r0,
        maxRadius=r1,
    )

    if circles is None:
        c_x = width / 2
        c_y = height / 2
        radius = 0.55 * max(height, width)
    else:
        circles = np.uint16(np.around(circles))
        c_x = (circles[0, 0, 0] - w) / ratio
        c_y = (circles[0, 0, 1] - h) / ratio
        radius = circles[0, 0, 2] / ratio
    return c_x, c_y, radius


def circleCrop(c_x, c_y, radius, height, width):
    if math.floor(radius + c_y) > height:
        y0 = max(math.ceil(c_y - radius), 0)
        y1 = height
        if math.floor(radius + c_x) > width:
            x1 = width
        else:
            x1 = math.floor(radius + c_x)
        if math.floor(c_x - radius < 0):
            x0 = 0
        else:
            x0 = math.floor(c_x - radius)
    elif math.ceil(c_y - radius) < 0:
        y0 = 0
        y1 = min(math.floor(c_y + radius), height)
        if math.floor(radius + c_x) > width:
            x1 = width
        else:
            x1 = math.floor(radius + c_x)
        if math.floor(c_x - radius < 0):
            x0 = 0
        else:
            x0 = math.floor(c_x - radius)
    else:
        y0 = math.ceil(c_y - radius)
        y1 = math.floor(c_y + radius)
        x0 = math.ceil(c_x - radius)
        x1 = math.floor(c_x + radius)
    return x0, x1, y0, y1


def trimFundus(image):
    c_x, c_y, radius = findCircle(image)
    height, width = image.shape[0], image.shape[1]
    x0, x1, y0, y1 = circleCrop(c_x, c_y, radius, height, width)
    trimmed = image[y0:y1, x0:x1, :]
    return trimmed


def load_ben_yuan(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    return image


PARAM = 92


def Radius_Reduction(img, PARAM):
    h, w, c = img.shape
    Frame = np.zeros((h, w, c), dtype=np.uint8)
    cv2.circle(
        Frame,
        (int(math.floor(w / 2)), int(math.floor(h / 2))),
        int(math.floor((h * PARAM) / float(2 * 100))),
        (255, 255, 255),
        -1,
    )
    Frame1 = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    img1 = cv2.bitwise_and(img, img, mask=Frame1)
    return img1


def info_image(im):
    cy = im.shape[0] // 2
    midline = im[cy, :]
    midline = np.where(midline > midline.mean() / 3)[0]
    if len(midline) > im.shape[1] // 2:
        x_start, x_end = np.min(midline), np.max(midline)
    else:
        x_start, x_end = im.shape[1] // 10, 9 * im.shape[1] // 10
    cx = (x_start + x_end) / 2
    r = (x_end - x_start) / 2
    return cx, cy, r


def resize_image(im, img_size, augmentation=False):
    cx, cy, r = info_image(im)
    scaling = img_size / (2 * r)
    rotation = 0
    if augmentation:
        scaling *= 1 + 0.3 * (np.random.rand() - 0.5)
        rotation = 360 * np.random.rand()
    M = cv2.getRotationMatrix2D((cx, cy), rotation, scaling)
    M[0, 2] -= cx - img_size / 2
    M[1, 2] -= cy - img_size / 2
    return cv2.warpAffine(im, M, (img_size, img_size))


def subtract_median_bg_image(im):
    k = np.max(im.shape) // 20 * 2 + 1
    bg = cv2.medianBlur(im, k)
    return cv2.addWeighted(im, 4, bg, -4, 128)


def subtract_gaussian_bg_image(im):
    bg = cv2.GaussianBlur(im, (0, 0), 10)
    return cv2.addWeighted(im, 4, bg, -4, 128)


def open_img(fn, size):
    image = cv2.imread(fn)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image, size)
    image = subtract_gaussian_bg_image(image)
    image = Radius_Reduction(image, PARAM)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    return image




## === cell 3
class eye_dataset_orl(Dataset):
    """
    Simple dataset for loading test images.
    Returns a pre‑processed tensor and the image id (without the .png suffix).
    """

    def __init__(
        self,
        file_names,
        transform=None,
        root_dir="../input/aptos2019-blindness-detection/test_images",
    ):
        self.file_names = file_names
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        full_path = os.path.join(self.root_dir, img_name)
        img = open_img(full_path, 512)  # returns a NumPy array (H,W,C)
        if self.transform:
            img = self.transform(img)
        img_id = img_name.replace(".png", "")
        return img, img_id




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1614678548.py in <cell line: 0>()
----> 1 class eye_dataset_orl(Dataset):
      2     """
      3     Simple dataset for loading test images.
      4     Returns a pre‑processed tensor and the image id (without the .png suffix).
      5     """

NameError: name 'Dataset' is not defined

## === cell 4
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(0)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    name_file = "../input/aptos2019-blindness-detection/test.csv"
    csv_file = csv.reader(open(name_file, "r"))
    content = []
    for line in csv_file:
        content.append(line[0] + ".png")
    content = content[1:]

    test_data = eye_dataset_orl(content, transform2)

    net = Baseline_single(num_classes=5)
    if use_gpu:
        net = net.cuda()

    ckpt_path = "/kaggle/input/temp-file/model_yuan512_dense201_00001_adam_combine_orl_bce_maxest.pkl"
    if os.path.exists(ckpt_path):
        net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        fallback = False
    else:
        print(
            f"Warning: checkpoint not found at {ckpt_path}. Using a constant‑prediction fallback."
        )
        fallback = True

        train_csv_path = "../input/aptos2019-blindness-detection/train.csv"
        train_df = pd.read_csv(train_csv_path)
        most_common = int(train_df["diagnosis"].value_counts().idxmax())

        class ConstantPredictor(nn.Module):
            def __init__(self, const_class, num_classes=5):
                super().__init__()
                self.const_class = const_class
                logits = torch.full((1, num_classes), -1e9)
                logits[0, const_class] = 1e9
                self.register_buffer("logits", logits)

            def forward(self, x):
                batch_size = x.size(0)
                return self.logits.expand(batch_size, -1)

        net = ConstantPredictor(const_class=most_common, num_classes=5)
        if use_gpu:
            net = net.cuda()

    net.eval()
    dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    submission_path = "/kaggle/working/submission.csv"
    with open(submission_path, "w", newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["id_code", "diagnosis"])

    with torch.no_grad():
        for idx, (data, name) in enumerate(tqdm(dataloader_test)):
            if use_gpu:
                data = data.cuda()
            out = net(data)  # logits (or constant logits)
            pred = torch.argmax(out, dim=1).cpu().numpy()  # class indices 0‑4
            row = [str(name[0]), str(int(pred[0]))]
            with open(submission_path, "a", newline="") as f:
                f_csv = csv.writer(f)
                f_csv.writerow(row)

    print(pd.read_csv(submission_path).diagnosis.value_counts())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/654067197.py in <cell line: 0>()
      1 if __name__ == "__main__":
----> 2     use_gpu = torch.cuda.is_available()
      3     if use_gpu:
      4         cudnn.benchmark = True
      5         torch.cuda.manual_seed_all(0)

NameError: name 'torch' is not defined
