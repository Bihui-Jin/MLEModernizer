# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (321 lines)
            metadata.zip (1.2 kB)
            sample_submission.csv (37088 lines)
            sample_submission.csv.zip (108.0 kB)
            test.zip (557.7 MB)
            train.zip (3.7 GB)
            metadata/
                accumulated_delta_range_state_bit_map.json (1 lines)
                constellation_type_mapping.csv (9 lines)
                ... and 1 other files
            smartphone-decimeter-2022/
                description.md (321 lines)
                metadata.zip (1.2 kB)
                ... and 4 other files
                metadata/
                    accumulated_delta_range_state_bit_map.json (1 lines)
                    constellation_type_mapping.csv (9 lines)
                    ... and 1 other files
                smartphone-decimeter-2022/
                test/
                    2020-06-04-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-06-04-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2021-04-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                    2021-04-29-US-MTV-1/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-04-29-US-MTV-2/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-08-24-US-SVL-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    test/
                train/
                    2020-05-15-US-MTV-1/
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-05-21-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    ... and 53 other folders
            test/
                2020-06-04-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (56087 lines)
                        device_imu.csv (340189 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (58761 lines)
                        device_imu.csv (342285 lines)
                        supplemental/
                            ... (max depth reached)
                2020-06-04-US-MTV-2/
                    GooglePixel4/
                        device_gnss.csv (68061 lines)
                        device_imu.csv (338641 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (68855 lines)
                        device_imu.csv (339610 lines)
                        supplemental/
                            ... (max depth reached)
                2020-07-08-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (73508 lines)
                        device_imu.csv (456999 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (77061 lines)
                        device_imu.csv (454150 lines)
                        supplemental/
                            ... (max depth reached)
                2020-07-08-US-MTV-2/
                    GooglePixel4/
                        device_gnss.csv (64478 lines)
                        device_imu.csv (456044 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (68307 lines)
                        device_imu.csv (449696 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-08-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (19537 lines)
                        device_imu.csv (221095 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel5/
                        device_gnss.csv (34594 lines)
                        device_imu.csv (222954 lines)
                        supplemental/
                            ... (max depth reached)
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (40323 lines)
                        device_imu.csv (216914 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-29-US-MTV-1/
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (60277 lines)
                        device_imu.csv (344013 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (61077 lines)
                        device_imu.csv (235288 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-29-US-MTV-2/
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (66015 lines)
                        device_imu.csv (371204 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (65501 lines)
                        device_imu.csv (257874 lines)
                        supplemental/
                            ... (max depth reached)
                2021-08-24-US-SVL-1/
                    GooglePixel4/
                        device_gnss.csv (101566 lines)
                        device_imu.csv (711980 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel5/
                        device_gnss.csv (112728 lines)
                        device_imu.csv (721330 lines)
                        supplemental/
                            ... (max depth reached)
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (122140 lines)
                        device_imu.csv (700392 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (133142 lines)
                        device_imu.csv (478300 lines)
                        supplemental/
                            ... (max depth reached)
                test/
            train/
                2020-05-15-US-MTV-1/
                    GooglePixel4XL/
                        device_gnss.csv (90154 lines)
                        device_imu.csv (734857 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                2020-05-21-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (61368 lines)
                        device_imu.csv (415251 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (64498 lines)
                        device_imu.csv (415486 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                ... and 53 other folders
        input/
            description.md (321 lines)
            metadata.zip (1.2 kB)
            sample_submission.csv (37088 lines)
            sample_submission.csv.zip (108.0 kB)
            test.zip (557.7 MB)
            train.zip (3.7 GB)
            metadata/
                accumulated_delta_range_state_bit_map.json (1 lines)
                constellation_type_mapping.csv (9 lines)
                ... and 1 other files
            smartphone-decimeter-2022/
                description.md (321 lines)
                metadata.zip (1.2 kB)
                ... and 4 other files
                metadata/
                    accumulated_delta_range_state_bit_map.json (1 lines)
                    constellation_type_mapping.csv (9 lines)
                    ... and 1 other files
                smartphone-decimeter-2022/
                test/
                    2020-06-04-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-06-04-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2021-04-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                    2021-04-29-US-MTV-1/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-04-29-US-MTV-2/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-08-24-US-SVL-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    test/
                train/
                    2020-05-15-US-MTV-1/
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-05-21-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    ... and 53 other folders
            test/
                2020-06-04-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (56087 lines)
                        device_imu.csv (340189 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (58761 lines)
                        device_imu.csv (342285 lines)
                        supplemental/
                            ... (max depth reached)
                2020-06-04-US-MTV-2/
                    GooglePixel4/
                        device_gnss.csv (68061 lines)
                        device_imu.csv (338641 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (68855 lines)
                        device_imu.csv (339610 lines)
                        supplemental/
                            ... (max depth reached)
                2020-07-08-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (73508 lines)
                        device_imu.csv (456999 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (77061 lines)
                        device_imu.csv (454150 lines)
                        supplemental/
                            ... (max depth reached)
                2020-07-08-US-MTV-2/
                    GooglePixel4/
                        device_gnss.csv (64478 lines)
                        device_imu.csv (456044 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (68307 lines)
                        device_imu.csv (449696 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-08-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (19537 lines)
                        device_imu.csv (221095 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel5/
                        device_gnss.csv (34594 lines)
                        device_imu.csv (222954 lines)
                        supplemental/
                            ... (max depth reached)
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (40323 lines)
                        device_imu.csv (216914 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-29-US-MTV-1/
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (60277 lines)
                        device_imu.csv (344013 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (61077 lines)
                        device_imu.csv (235288 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-29-US-MTV-2/
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (66015 lines)
                        device_imu.csv (371204 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (65501 lines)
                        device_imu.csv (257874 lines)
                        supplemental/
                            ... (max depth reached)
                2021-08-24-US-SVL-1/
                    GooglePixel4/
                        device_gnss.csv (101566 lines)
                        device_imu.csv (711980 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel5/
                        device_gnss.csv (112728 lines)
                        device_imu.csv (721330 lines)
                        supplemental/
                            ... (max depth reached)
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (122140 lines)
                        device_imu.csv (700392 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (133142 lines)
                        device_imu.csv (478300 lines)
                        supplemental/
                            ... (max depth reached)
                test/
                    2020-06-04-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-06-04-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2021-04-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                    2021-04-29-US-MTV-1/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-04-29-US-MTV-2/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-08-24-US-SVL-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    test/
            train/
                2020-05-15-US-MTV-1/
                    GooglePixel4XL/
                        device_gnss.csv (90154 lines)
                        device_imu.csv (734857 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                2020-05-21-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (61368 lines)
                        device_imu.csv (415251 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (64498 lines)
                        device_imu.csv (415486 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                ... and 53 other folders
        working/
            smartphone-decimeter-2022/
                description.md (321 lines)
                metadata.zip (1.2 kB)
                ... and 4 other files
                metadata/
                    accumulated_delta_range_state_bit_map.json (1 lines)
                    constellation_type_mapping.csv (9 lines)
                    ... and 1 other files
                smartphone-decimeter-2022/
                test/
                    2020-06-04-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-06-04-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2021-04-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                    2021-04-29-US-MTV-1/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-04-29-US-MTV-2/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-08-24-US-SVL-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    test/
                train/
                    2020-05-15-US-MTV-1/
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-05-21-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    ... and 53 other folders
```

-> data/metadata/accumulated_delta_range_state_bit_map.json has auto-generated json schema:
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

-> data/metadata/constellation_type_mapping.csv has 8 rows and 2 columns.
The columns are: constellationType, constellationName

-> data/metadata/raw_state_bit_map.json has auto-generated json schema:
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
    },
    "5": {
      "type": "string"
    },
    "6": {
      "type": "string"
    },
    "7": {
      "type": "string"
    },
    "8": {
      "type": "string"
    },
    "9": {
      "type": "string"
    },
    "10": {
      "type": "string"
    },
    "11": {
      "type": "string"
    },
    "12": {
      "type": "string"
    },
    "13": {
      "type": "string"
    },
    "14": {
      "type": "string"
    },
    "15": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
  ]
}

-> data/sample_submission.csv has 37087 rows and 4 columns.
The columns are: tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees

-> data/smartphone-decimeter-2022/metadata/accumulated_delta_range_state_bit_map.json has auto-generated json schema:
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

-> data/smartphone-decimeter-2022/metadata/constellation_type_mapping.csv has 8 rows and 2 columns.
The columns are: constellationType, constellationName

-> data/smartphone-decimeter-2022/metadata/raw_state_bit_map.json has auto-generated json schema:
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
    },
    "5": {
      "type": "string"
    },
    "6": {
      "type": "string"
    },
    "7": {
      "type": "string"
    },
    "8": {
      "type": "string"
    },
    "9": {
      "type": "string"
    },
    "10": {
      "type": "string"
    },
    "11": {
      "type": "string"
    },
    "12": {
      "type": "string"
    },
    "13": {
      "type": "string"
    },
    "14": {
      "type": "string"
    },
    "15": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
  ]
}

-> data/smartphone-decimeter-2022/sample_submission.csv has 37087 rows and 4 columns.
The columns are: tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
from glob import glob
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold


## === cell 1
from contextlib import contextmanager
from time import time

class Timer:
    def __init__(self, logger=None, format_str="{:.3f}[s]", prefix=None, suffix=None, sep=" "):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


## === cell 2
BASE_DIR = Path('../input/smartphone-decimeter-2022')
df_sample_submission = pd.read_csv(BASE_DIR / "sample_submission.csv")


## === cell 3
train_folders = glob(str(BASE_DIR / "train/*/*"))

X = []
y_LatitudeDegrees = []
y_LongitudeDegrees = []

for train_folder in tqdm(train_folders):
    df_device_gnss = pd.read_csv(f"{train_folder}/device_gnss.csv").rename(
        columns={"utcTimeMillis": "UnixTimeMillis"}
    )
    df_device_gnss = (
        df_device_gnss.groupby("UnixTimeMillis").mean(numeric_only=True).reset_index()
    )

    df_ground_truth = pd.read_csv(
        f"{train_folder}/ground_truth.csv",
        usecols=["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"],
    )

    df_merged = pd.merge(
        df_device_gnss, df_ground_truth, on="UnixTimeMillis", how="left"
    )

    X.append(df_merged.drop(columns=["LatitudeDegrees", "LongitudeDegrees"]))
    y_LatitudeDegrees.append(df_merged["LatitudeDegrees"])
    y_LongitudeDegrees.append(df_merged["LongitudeDegrees"])

X = pd.concat(X).values
y_LatitudeDegrees = pd.concat(y_LatitudeDegrees).values
y_LongitudeDegrees = pd.concat(y_LongitudeDegrees).values


## === cell 4
test_folders = glob(str(BASE_DIR / "test/*/*"))

test_index = []
test_X = []
test_y_LatitudeDegrees = []
test_y_LongitudeDegrees = []

for test_folder in tqdm(test_folders):
    df_device_gnss = pd.read_csv(f"{test_folder}/device_gnss.csv").rename(
        columns={"utcTimeMillis": "UnixTimeMillis"}
    )
    df_device_gnss = (
        df_device_gnss.groupby("UnixTimeMillis").mean(numeric_only=True).reset_index()
    )

    dir_name, device_name = os.path.split(test_folder)
    _, id = os.path.split(dir_name)

    df_sample_by_tripId = df_sample_submission[
        df_sample_submission["tripId"] == f"{id}/{device_name}"
    ]
    df_merged = pd.merge(
        df_device_gnss,
        df_sample_by_tripId.drop(columns=["tripId"]),
        on="UnixTimeMillis",
        how="left",
    )

    df_index = df_merged[["UnixTimeMillis"]]
    df_index["tripId"] = f"{id}/{device_name}"

    test_index.append(df_index)
    test_X.append(df_merged.drop(columns=["LatitudeDegrees", "LongitudeDegrees"]))
    test_y_LatitudeDegrees.append(df_merged["LatitudeDegrees"])
    test_y_LongitudeDegrees.append(df_merged["LongitudeDegrees"])

test_index = pd.concat(test_index).reset_index(drop=True)
test_X = pd.concat(test_X).values
test_y_LatitudeDegrees = pd.concat(test_y_LatitudeDegrees).values
test_y_LongitudeDegrees = pd.concat(test_y_LongitudeDegrees).values


## === cell 5
fold = KFold(n_splits=5, shuffle=True, random_state=510)
cv = fold.split(X, y_LatitudeDegrees)
cv = list(cv)


## === cell 6
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm
import numpy as np

def fit_lgbm(X, 
             y, 
             cv, 
             params: dict=None, 
             verbose: int=50):

    if params is None:
        params = {}

    models = []
    n_records = len(X)
    oof_pred = np.zeros((n_records,), dtype=np.float32)

    for i, (idx_train, idx_valid) in enumerate(cv): 
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMRegressor(**params)

        with Timer(prefix="fit fold={} ".format(i)):
            clf.fit(x_train, y_train, 
                    eval_set=[(x_valid, y_valid)],  
                    early_stopping_rounds=100,
                    verbose=verbose)
            
        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)
        score = mean_squared_error(y_valid, pred_i)
        print(f" - fold{i + 1} - {score:.10f}")

    score = mean_squared_error(y, oof_pred)
    print(f"{score:.10f}")
    return oof_pred, models


## === cell 7
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm
import numpy as np


def fit_lgbm(X, y, cv, params: dict = None, verbose: int = 50):

    if params is None:
        params = {}

    models = []
    n_records = len(X)
    oof_pred = np.zeros((n_records,), dtype=np.float32)

    for i, (idx_train, idx_valid) in enumerate(cv):
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMRegressor(**params)

        callbacks = [
            lgbm.early_stopping(stopping_rounds=100),
            lgbm.log_evaluation(period=verbose),
        ]

        with Timer(prefix="fit fold={} ".format(i)):
            clf.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                callbacks=callbacks,
            )

        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)
        score = mean_squared_error(y_valid, pred_i)
        print(f" - fold{i + 1} - {score:.10f}")

    score = mean_squared_error(y, oof_pred)
    print(f"{score:.10f}")
    return oof_pred, models


## === cell 8
pred_LatitudeDegrees = np.mean(np.array([model.predict(test_X) for model in models_LatitudeDegrees]), axis=0)
pred_LongitudeDegrees = np.mean(np.array([model.predict(test_X) for model in models_LongitudeDegrees]), axis=0)
df_res = pd.DataFrame({
    "LatitudeDegrees": pred_LatitudeDegrees,
    "LongitudeDegrees": pred_LongitudeDegrees
})
df_res = pd.concat([test_index, df_res], axis=1)


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2831227464.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mpred_LatitudeDegrees[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0;34m[[0m[0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mtest_X[0m[0;34m)[0m [0;32mfor[0m [0mmodel[0m [0;32min[0m [0mmodels_LatitudeDegrees[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mpred_LongitudeDegrees[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0;34m[[0m[0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mtest_X[0m[0;34m)[0m [0;32mfor[0m [0mmodel[0m [0;32min[0m [0mmodels_LongitudeDegrees[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m df_res = pd.DataFrame({
[1;32m      4[0m     [0;34m"LatitudeDegrees"[0m[0;34m:[0m [0mpred_LatitudeDegrees[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0;34m"LongitudeDegrees"[0m[0;34m:[0m [0mpred_LongitudeDegrees[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: name 'models_LatitudeDegrees' is not defined

## === cell 9
df_res = df_res.reindex(columns=['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']).sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True).to_csv("submission.csv", index=False)
