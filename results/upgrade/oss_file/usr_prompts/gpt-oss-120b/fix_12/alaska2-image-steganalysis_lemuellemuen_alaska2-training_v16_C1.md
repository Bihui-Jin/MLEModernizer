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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.13

# 3. Installed packages

albumentations==2.0.8
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.8876821205467409

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59505) has done: 'I remove the unavailable `torchsampler` import and replace its usage with a `WeightedRandomSampler`, which resolves the import error and the subsequent NameErrors. This also restores the data loaders so training can run, after which inference and the ensemble creation work and produce a valid `submission.csv`. The core model architecture and training logic remain unchanged.'
- What this solution (achieved 0.59417) has done: 'I add all required imports and definitions at the start, fixing the NameError issues for torch, random, os, numpy, pandas, glob, cv2, albumentations, torch modules, and sklearn utilities. This makes the pipeline runnable end‑to‑end and ensures a proper `submission.csv` is created.'

# 9. Code solution

## === cell 0
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1208484179.py in <cell line: 0>()
     10 
     11 
---> 12 seed_everything(SEED)
     13 
     14 

/tmp/ipykernel_55/1208484179.py in seed_everything(seed)
      3 
      4 def seed_everything(seed):
----> 5     random.seed(seed)
      6     os.environ["PYTHONHASHSEED"] = str(seed)
      7     np.random.seed(seed)

NameError: name 'random' is not defined

## === cell 1
class Config:
    batch_size = 64
    n_epochs = 2  # keep training but enough to finish under the limit
    num_workers = 8
    lr = 0.001
    verbose = True
    verbose_step = 1
    step_scheduler = False
    validation_scheduler = True
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode="min",
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=1e-4,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-8,
        eps=1e-8,
    )




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/572918323.py in <cell line: 0>()
----> 1 class Config:
      2     # larger batch reduces number of optimizer steps per epoch
      3     batch_size = 64
      4     n_epochs = 2  # keep training but enough to finish under the limit
      5     num_workers = 8

/tmp/ipykernel_55/572918323.py in Config()
      9     step_scheduler = False
     10     validation_scheduler = True
---> 11     SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
     12     scheduler_params = dict(
     13         mode="min",

NameError: name 'torch' is not defined

## === cell 2
class Fitter:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.epoch = 0
        self.best_summary_loss = float("inf")
        self.log_path = "./log.txt"
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(
            self.optimizer, **config.scheduler_params
        )
        self.criterion = LabelSmoothing().to(self.device)

        self.scaler = (
            torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
        )

    def log(self, msg):
        if self.config.verbose:
            print(msg)
        with open(self.log_path, "a+") as f:
            f.write(msg + "\n")

    def fit(self, train_loader, val_loader):
        for e in range(self.epoch, self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                self.log(f"\nEpoch {e} - LR {lr}")
            t0 = time.time()
            train_loss, train_score = self._run_epoch(train_loader, train=True)
            self.log(
                f"[RESULT]: Train. Epoch: {e}, loss: {train_loss.avg:.5f}, "
                f"score: {train_score.avg:.5f}, time: {time.time()-t0:.2f}"
            )
            self.save("last-checkpoint.bin")
            t0 = time.time()
            val_loss, val_score = self._run_epoch(val_loader, train=False)
            self.log(
                f"[RESULT]: Val. Epoch: {e}, loss: {val_loss.avg:.5f}, "
                f"score: {val_score.avg:.5f}, time: {time.time()-t0:.2f}"
            )
            if val_loss.avg < self.best_summary_loss:
                self.best_summary_loss = val_loss.avg
                self.save(f"best-checkpoint-{e:03d}epoch.bin")
            if self.config.validation_scheduler:
                self.scheduler.step(metrics=val_loss.avg)
            self.epoch += 1

    def _run_epoch(self, loader, train):
        self.model.train() if train else self.model.eval()
        loss_meter = AverageMeter()
        score_meter = RocAucMeter() if not train else None
        for step, (imgs, targets) in enumerate(loader):
            if train:
                self.optimizer.zero_grad()
            imgs = imgs.to(self.device, non_blocking=True).float()
            targets = targets.to(self.device, non_blocking=True).float()

            if train and self.scaler is not None:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with (
                    torch.autocast(device_type=self.device.type, dtype=torch.float16)
                    if self.device.type == "cuda"
                    else torch.no_grad()
                ):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, targets)

                if train:
                    loss.backward()
                    self.optimizer.step()

            loss_meter.update(loss.item(), imgs.size(0))
            if not train:
                score_meter.update(targets, outputs)

        if train:
            dummy_score = RocAucMeter()
            dummy_score.score = 0.0
            return loss_meter, dummy_score
        return loss_meter, score_meter

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_summary_loss": self.best_summary_loss,
                "epoch": self.epoch,
            },
            path,
        )

    def load(self, path):
        chk = torch.load(path, map_location=self.device)
        self.model.load_state_dict(chk["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(chk["optimizer_state_dict"])
        self.scheduler.load_state_dict(chk["scheduler_state_dict"])
        self.best_summary_loss = chk["best_summary_loss"]
        self.epoch = chk["epoch"]
        self.log(f"Loaded checkpoint {path}, resume epoch {self.epoch}")
