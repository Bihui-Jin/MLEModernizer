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

3.10

# 3. Installed packages

albumentations==2.0.8
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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

17.90779

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))




## === cell 1
base_dir = "/kaggle/input"
model_weights = os.path.join(base_dir, "saved-weights", "swin_large_fastai(final).pth")
test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3731336891.py in <cell line: 0>()
      1 base_dir = "/kaggle/input"
----> 2 model_weights = os.path.join(base_dir, "saved-weights", "swin_large_fastai(final).pth")
      3 test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
      4 test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")
      5 

NameError: name 'os' is not defined

## === cell 2
input_shape = (224, 224)
mean, std_dev = [0.5023, 0.4615, 0.4226], [0.2640, 0.2593, 0.2575]
model_name = "swin_large_patch4_window7_224_in22k"
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
output_categories = 1  # single regression output
n_epochs = 15
num_of_hidden = 2
hidden_dimension = [256, 64]
save_name = "/kaggle/working/swin_fastai(final).pth"



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/307193526.py in <cell line: 0>()
      3 model_name = "swin_large_patch4_window7_224_in22k"
      4 batch_size = 64
----> 5 device = "cuda" if torch.cuda.is_available() else "cpu"
      6 output_categories = 1  # single regression output
      7 n_epochs = 15

NameError: name 'torch' is not defined

## === cell 3
test_csv = pd.read_csv(test_file)
test_csv["path_img"] = test_csv["Id"].apply(
    lambda x: os.path.join(test_folder, f"{x}.jpg")
)
test_csv["Pawpularity"] = 1  # placeholder




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1133431365.py in <cell line: 0>()
----> 1 test_csv = pd.read_csv(test_file)
      2 test_csv["path_img"] = test_csv["Id"].apply(
      3     lambda x: os.path.join(test_folder, f"{x}.jpg")
      4 )
      5 test_csv["Pawpularity"] = 1  # placeholder

NameError: name 'pd' is not defined

## === cell 4
class PetsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.cat = [
            "Subject Focus",
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["path_img"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tab = torch.tensor(self.df.iloc[idx][self.cat].values.astype(np.float32))
        label = torch.tensor(self.df.iloc[idx]["Pawpularity"], dtype=torch.float32)
        return img, tab, label




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/658158067.py in <cell line: 0>()
----> 1 class PetsDataset(Dataset):
      2     def __init__(self, df, transform=None):
      3         self.df = df
      4         self.transform = transform
      5         self.cat = [

NameError: name 'Dataset' is not defined

## === cell 5
test_transform = transforms.Compose(
    [
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(input_shape),
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)
                )
            ],
            p=0.5,
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev),
    ]
)
test_dataset = PetsDataset(test_csv, transform=test_transform)
testloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=1, shuffle=False, pin_memory=True
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/187124450.py in <cell line: 0>()
----> 1 test_transform = transforms.Compose(
      2     [
      3         transforms.Resize(448, interpolation=transforms.InterpolationMode.BILINEAR),
      4         transforms.CenterCrop(input_shape),
      5         transforms.RandomApply(

NameError: name 'transforms' is not defined

## === cell 6
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Network(nn.Module):
    def __init__(
        self,
        base,
        number_of_hidden,
        hidden,
        regression_out,
        output_categories,
        freeze_layer,
        input_shape,
        device,
    ):
        super().__init__()
        if number_of_hidden != len(hidden):
            raise ValueError(
                "Number of Hidden layer and length of hidden dim must be same"
            )
        base.head = Identity()  # remove original classifier
        dummy = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
        with torch.no_grad():
            dummy_feat = base(dummy)
            if dummy_feat.dim() > 2:
                dummy_feat = torch.flatten(dummy_feat, 1)
        visual_feat_dim = dummy_feat.shape[1]  # e.g. 75276
        target_vis_dim = (
            base.head.in_features if hasattr(base.head, "in_features") else hidden[0]
        )

        self.proj = nn.Linear(visual_feat_dim, target_vis_dim)

        hidden_dim = hidden[:]
        hidden_dim.insert(
            0, target_vis_dim + 12
        )  # after concatenating 12 tabular features
        self.p = 0.5
        self.regression = self._fully_connected(
            number_of_hidden, hidden_dim, output_categories
        )
        self.network = self._freeze_layer(base, freeze_layer)
        self._initialise_weights()

    def _initialise_weights(self):
        for m in self.regression:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_layer(self, base, freeze_layer):
        for i, child in enumerate(base.children()):
            if i < freeze_layer:
                for param in child.parameters():
                    param.requires_grad = False
        return base

    def _fully_connected(self, number_of_hidden, hidden_dim, output_categories):
        layers = []
        for i in range(number_of_hidden):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(hidden_dim[i + 1]))
            if i != number_of_hidden - 1:
                layers.append(nn.Dropout(self.p))
        layers.append(nn.Linear(hidden_dim[-1], output_categories))
        return nn.Sequential(*layers)

    def forward(self, x, tab):
        x_feat = self.network(x)
        if x_feat.dim() > 2:
            x_feat = torch.flatten(x_feat, 1)
        x_feat = self.proj(x_feat)
        x_comb = torch.cat([x_feat, tab], dim=1)
        return self.regression(x_comb)


network = timm.create_model(model_name, pretrained=True)
model = Network(
    network,
    num_of_hidden,
    hidden_dimension,
    regression_out=1,
    output_categories=output_categories,
    freeze_layer=0,
    input_shape=input_shape,
    device=device,
).to(device)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/15081565.py in <cell line: 0>()
----> 1 class Identity(nn.Module):
      2     def __init__(self):
      3         super().__init__()
      4 
      5     def forward(self, x):

NameError: name 'nn' is not defined

## === cell 7
try:
    state_dict = torch.load(model_weights, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # allow missing fastai keys
except Exception as e:
    print(f"Weight file not loaded ({e}); proceeding with pretrained model.")
model.eval()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/609733586.py in <cell line: 0>()
      4 except Exception as e:
      5     print(f"Weight file not loaded ({e}); proceeding with pretrained model.")
----> 6 model.eval()
      7 

NameError: name 'model' is not defined

## === cell 8
tta_steps = 4
all_outputs = []
for _ in range(tta_steps):
    step_outputs = []
    with torch.no_grad():
        for imgs, tabs, _ in testloader:
            imgs = imgs.to(device, non_blocking=True)
            tabs = tabs.to(device, non_blocking=True)
            preds = model(imgs, tabs)
            preds = 100 * torch.sigmoid(preds)
            preds = torch.clamp(preds, 0.0, 100.0)  # ensure valid range
            step_outputs.extend(preds.squeeze().cpu().numpy().tolist())
    all_outputs.append(step_outputs)

tta_outputs_arr = np.mean(np.array(all_outputs), axis=0)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/311779574.py in <cell line: 0>()
      3 for _ in range(tta_steps):
      4     step_outputs = []
----> 5     with torch.no_grad():
      6         for imgs, tabs, _ in testloader:
      7             imgs = imgs.to(device, non_blocking=True)

NameError: name 'torch' is not defined

## === cell 9
submission = pd.DataFrame({"Id": test_csv["Id"], "Pawpularity": tta_outputs_arr})
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2996826766.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"Id": test_csv["Id"], "Pawpularity": tta_outputs_arr})
      2 submission.to_csv("submission.csv", index=False)

NameError: name 'pd' is not defined
