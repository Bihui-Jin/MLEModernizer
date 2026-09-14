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
plotly==5.24.1
plotly-express==0.4.1
scipy==1.15.3
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
import os, sys, warnings

warnings.filterwarnings("ignore")



## === cell 1
try:
    ip = get_ipython()
    if ip is not None:
        try:
            ip.run_line_magic("load_ext", "lab_black")
        except ModuleNotFoundError:
            try:
                ip.run_line_magic("load_ext", "nb_black")
            except ModuleNotFoundError:
                pass
except NameError:
    pass



## === cell 2
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import plotly.express as px

pd.set_option("display.max_columns", 500)



## === cell 3
trip_id = "2020-05-15-US-MTV-1/GooglePixel4XL"



## === cell 4
gt = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/ground_truth.csv"
)
gnss = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/device_gnss.csv"
)
imu = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/device_imu.csv"
)



## === cell 5
import glob
from dataclasses import dataclass
from tqdm.notebook import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

INPUT_PATH = "../input/smartphone-decimeter-2022"

WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_SQUARED_FIRST_ECCENTRICITY = 6.69437999013e-3
WGS84_SQUARED_SECOND_ECCENTRICITY = 6.73949674226e-3

HAVERSINE_RADIUS = 6_371_000


@dataclass
class ECEF:
    x: np.array
    y: np.array
    z: np.array

    def to_numpy(self):
        return np.stack([self.x, self.y, self.z], axis=0)

    @staticmethod
    def from_numpy(pos):
        x, y, z = [np.squeeze(w) for w in np.split(pos, 3, axis=-1)]
        return ECEF(x=x, y=y, z=z)


@dataclass
class BLH:
    lat: np.array
    lng: np.array
    hgt: np.array


def ECEF_to_BLH(ecef):
    a = WGS84_SEMI_MAJOR_AXIS
    b = WGS84_SEMI_MINOR_AXIS
    e2 = WGS84_SQUARED_FIRST_ECCENTRICITY
    e2_ = WGS84_SQUARED_SECOND_ECCENTRICITY
    x = ecef.x
    y = ecef.y
    z = ecef.z
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(z * (a / b), r)
    B = np.arctan2(z + (e2_ * b) * np.sin(t) ** 3, r - (e2 * a) * np.cos(t) ** 3)
    L = np.arctan2(y, x)
    n = a / np.sqrt(1 - e2 * np.sin(B) ** 2)
    H = (r / np.cos(B)) - n
    return BLH(lat=B, lng=L, hgt=H)


def haversine_distance(blh_1, blh_2):
    dlat = blh_2.lat - blh_1.lat
    dlng = blh_2.lng - blh_1.lng
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(blh_1.lat) * np.cos(blh_2.lat) * np.sin(dlng / 2) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    dist = 2 * HAVERSINE_RADIUS * np.arcsin(np.sqrt(a))
    return dist


def pandas_haversine_distance(df1, df2):
    blh1 = BLH(
        lat=np.deg2rad(df1["LatitudeDegrees"].to_numpy()),
        lng=np.deg2rad(df1["LongitudeDegrees"].to_numpy()),
        hgt=0,
    )
    blh2 = BLH(
        lat=np.deg2rad(df2["LatitudeDegrees"].to_numpy()),
        lng=np.deg2rad(df2["LongitudeDegrees"].to_numpy()),
        hgt=0,
    )
    return haversine_distance(blh1, blh2)


def _wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis):
    """
    WLS ECEF->BLH + spline interpolation.
    Small robustness guards only (avoid spline NaNs / wrap jumps), preserving core logic.
    """
    ecef_columns = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    columns = ["utcTimeMillis"] + ecef_columns

    ecef_df = gnss_df[columns].copy()
    ecef_df = ecef_df.dropna().sort_values("utcTimeMillis").reset_index(drop=True)

    if len(ecef_df) > 0:
        t = ecef_df["utcTimeMillis"].to_numpy()
        _, unique_idx = np.unique(t, return_index=True)
        ecef_df = ecef_df.iloc[np.sort(unique_idx)].reset_index(drop=True)

    if len(ecef_df) == 0:
        return pd.DataFrame(
            {
                "tripId": tripID,
                "UnixTimeMillis": np.asarray(UnixTimeMillis, dtype=np.int64),
                "LatitudeDegrees": np.nan,
                "LongitudeDegrees": np.nan,
            }
        )

    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
    blh = ECEF_to_BLH(ecef)

    TIME = ecef_df["utcTimeMillis"].to_numpy(dtype=np.float64)
    m = len(TIME)

    if m < 2:
        lat = np.clip(blh.lat.astype(np.float64)[-1], -np.pi / 2, np.pi / 2)
        lng = _wrap_to_pi(blh.lng.astype(np.float64)[-1])
        UnixTimeMillis = np.asarray(UnixTimeMillis, dtype=np.int64)
        return pd.DataFrame(
            {
                "tripId": tripID,
                "UnixTimeMillis": UnixTimeMillis,
                "LatitudeDegrees": np.degrees(
                    np.full_like(UnixTimeMillis, lat, dtype=float)
                ),
                "LongitudeDegrees": np.degrees(
                    np.full_like(UnixTimeMillis, lng, dtype=float)
                ),
            }
        )

    k = int(min(3, max(1, m - 1)))

    lat_u = np.unwrap(blh.lat.astype(np.float64))
    lng_u = np.unwrap(blh.lng.astype(np.float64))

    lat_spl = InterpolatedUnivariateSpline(TIME, lat_u, k=k, ext=3)
    lng_spl = InterpolatedUnivariateSpline(TIME, lng_u, k=k, ext=3)

    UnixTimeMillis = np.asarray(UnixTimeMillis, dtype=np.float64)
    lat = lat_spl(UnixTimeMillis)
    lng = lng_spl(UnixTimeMillis)

    lat = np.clip(lat, -np.pi / 2, np.pi / 2)
    lng = _wrap_to_pi(lng)

    return pd.DataFrame(
        {
            "tripId": tripID,
            "UnixTimeMillis": UnixTimeMillis.astype(np.int64),
            "LatitudeDegrees": np.degrees(lat),
            "LongitudeDegrees": np.degrees(lng),
        }
    )


def calc_score(tripID, pred_df, gt_df):
    d = pandas_haversine_distance(pred_df, gt_df)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])
    return score




## === cell 6
ss = pd.read_csv(f"{INPUT_PATH}/sample_submission.csv")



## === cell 7
trip_id = "2020-05-15-US-MTV-1/GooglePixel4XL"
baseline = ecef_to_lat_lng(trip_id, gnss, gt["UnixTimeMillis"].values)




## === cell 8
def visualize_traffic(
    df,
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    center=None,
    color_col="tripId",
    label_col="tripId",
    zoom=9,
    opacity=1,
):
    if center is None:
        center = {
            "lat": df[lat_col].mean(),
            "lon": df[lon_col].mean(),
        }
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        labels=label_col,
        zoom=zoom,
        center=center,
        height=600,
        width=800,
        opacity=0.5,
    )
    fig.update_layout(mapbox_style="stamen-terrain")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()


def plot_gt_vs_baseline(tripId):
    gt = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/ground_truth.csv")
    gnss = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/device_gnss.csv")
    _ = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/device_imu.csv")

    baseline = ecef_to_lat_lng(tripId, gnss, gt["UnixTimeMillis"].values)
    baseline["isGT"] = False
    gt["isGT"] = True
    gt["tripId"] = tripId

    combined = (
        pd.concat([baseline, gt[baseline.columns]], axis=0)
        .reset_index(drop=True)
        .copy()
    )

    visualize_traffic(
        combined,
        lat_col="LatitudeDegrees",
        lon_col="LongitudeDegrees",
        color_col="isGT",
        zoom=10,
    )




## === cell 9
plot_gt_vs_baseline(trip_id)



## === cell 10
from glob import glob

train_gts = glob(f"{INPUT_PATH}/train/*/*/ground_truth.csv")
trip_ids = ["/".join(p.split("/")[-3:-1]) for p in train_gts]



## === cell 11
tripId = trip_ids[10]
gt = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/ground_truth.csv")
gnss = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/device_gnss.csv")
imu = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/device_imu.csv")

baseline = ecef_to_lat_lng(tripId, gnss, gt["UnixTimeMillis"].values)
baseline["isGT"] = False
gt["isGT"] = True
gt["tripId"] = tripId

combined = (
    pd.concat([baseline, gt[baseline.columns]], axis=0).reset_index(drop=True).copy()
)



## === cell 12
offset = 150_000
start = 1607640760432 + offset
combined.query("UnixTimeMillis < @start")
visualize_traffic(combined.query("UnixTimeMillis < @start"), color_col="isGT", zoom=16)



## === cell 13
import glob

sample_df = ss.copy()

pred_dfs = []
for dirname in tqdm(sorted(glob.glob(f"{INPUT_PATH}/test/*/*"))):
    drive, phone = dirname.split("/")[-2:]
    tripID = f"{drive}/{phone}"
    gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv", low_memory=False)

    UnixTimeMillis = sample_df.loc[
        sample_df["tripId"] == tripID, "UnixTimeMillis"
    ].to_numpy()
    pred_dfs.append(ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis))

sub_df_raw = pd.concat(pred_dfs, ignore_index=True)

sub_df_raw = (
    sub_df_raw.sort_values(["tripId", "UnixTimeMillis"])
    .groupby(["tripId", "UnixTimeMillis"], as_index=False)
    .agg({"LatitudeDegrees": "median", "LongitudeDegrees": "median"})
)

sub_df = ss[["tripId", "UnixTimeMillis"]].merge(
    sub_df_raw, on=["tripId", "UnixTimeMillis"], how="left", validate="one_to_one"
)

baselines = []
gts = []
for dirname in tqdm(sorted(glob.glob(f"{INPUT_PATH}/train/*/*"))):
    drive, phone = dirname.split("/")[-2:]
    tripID = f"{drive}/{phone}"
    gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv", low_memory=False)
    gt_df = pd.read_csv(f"{dirname}/ground_truth.csv", low_memory=False)
    baseline_df = ecef_to_lat_lng(tripID, gnss_df, gt_df["UnixTimeMillis"].to_numpy())
    baselines.append(baseline_df)
    gts.append(gt_df)
baselines = pd.concat(baselines, ignore_index=True)
gts = pd.concat(gts, ignore_index=True)



## === cell 14
baselines["group"] = "train_baseline"
sub_df["group"] = "submission_baseline"
gts["group"] = "train_ground_truth"
combined = pd.concat([baselines, sub_df, gts]).reset_index(drop=True).copy()



## === cell 15
sf_paths = combined.query("LatitudeDegrees > 36").copy()
la_paths = combined.query("LatitudeDegrees < 36").copy()



## === cell 16
visualize_traffic(
    sf_paths.sample(frac=0.2, random_state=0),
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    color_col="group",
    zoom=9,
)



## === cell 17
visualize_traffic(
    la_paths.sample(frac=0.2, random_state=0),
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    color_col="group",
    zoom=9,
)




## === cell 18
def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points on the earth."""
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    dist = 2 * RADIUS * np.arcsin(np.sqrt(a))
    return dist


def add_prev_post_shift(
    df,
    lat_col="LatitudeDegrees",
    lng_col="LongitudeDegrees",
    dist_suffix="",
    sortby=["tripId", "UnixTimeMillis"],
):
    """
    Shifts within tripId to avoid boundary leakage that can cause false stop detection.
    """
    df = df.sort_values(sortby).reset_index(drop=True)
    df[f"{lat_col}_shift1"] = df.groupby(["tripId"])[lat_col].shift(1)
    df[f"{lng_col}_shift1"] = df.groupby(["tripId"])[lng_col].shift(1)
    df[f"{lat_col}_shift-1"] = df.groupby(["tripId"])[lat_col].shift(-1)
    df[f"{lng_col}_shift-1"] = df.groupby(["tripId"])[lng_col].shift(-1)

    df["UnixTimeMillis_shift1"] = df.groupby(["tripId"])["UnixTimeMillis"].shift(1)
    df["UnixTimeMillis_shift-1"] = df.groupby(["tripId"])["UnixTimeMillis"].shift(-1)

    df[f"dist_prev{dist_suffix}"] = calc_haversine(
        df[lat_col], df[lng_col], df[f"{lat_col}_shift1"], df[f"{lng_col}_shift1"]
    )
    df[f"dist_post{dist_suffix}"] = calc_haversine(
        df[lat_col], df[lng_col], df[f"{lat_col}_shift-1"], df[f"{lng_col}_shift-1"]
    )

    df.loc[
        df[f"{lat_col}_shift1"].isna() | df[f"{lng_col}_shift1"].isna(),
        f"dist_prev{dist_suffix}",
    ] = np.nan
    df.loc[
        df[f"{lat_col}_shift-1"].isna() | df[f"{lng_col}_shift-1"].isna(),
        f"dist_post{dist_suffix}",
    ] = np.nan

    return df




## === cell 19
baselines2 = add_prev_post_shift(baselines)
baselines2["UnixTimeMillis_prev_diff"] = (
    baselines2["UnixTimeMillis"] - baselines2["UnixTimeMillis_shift1"]
)
baselines2["speed_calc"] = baselines2["dist_prev"] / (
    baselines2["UnixTimeMillis_prev_diff"] / 1000.0
)

sub_df = add_prev_post_shift(sub_df)
sub_df["UnixTimeMillis_prev_diff"] = (
    sub_df["UnixTimeMillis"] - sub_df["UnixTimeMillis_shift1"]
)
sub_df["speed_calc"] = sub_df["dist_prev"] / (
    sub_df["UnixTimeMillis_prev_diff"] / 1000.0
)



## === cell 20
baselines2.query("dist_prev < 5")["dist_prev"].plot(
    kind="hist", bins=50, title="Distribution of Speeds < 5"
)
plt.show()



## === cell 21
sub = sub_df.copy()
sub = sub.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)

pred_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
tmp = sub[pred_cols].copy()
for c in ["LatitudeDegrees", "LongitudeDegrees"]:
    tmp.loc[~np.isfinite(tmp[c].to_numpy()), c] = np.nan
global_lat_med = np.nanmedian(tmp["LatitudeDegrees"].to_numpy())
global_lng_med = np.nanmedian(tmp["LongitudeDegrees"].to_numpy())


def _fill_trip_nearest(trip_df):
    trip_df = trip_df.sort_values("UnixTimeMillis").copy()
    for col, global_med in [
        ("LatitudeDegrees", global_lat_med),
        ("LongitudeDegrees", global_lng_med),
    ]:
        s = trip_df[col]
        n_valid = int(s.notna().sum())
        if n_valid >= 2:
            x = trip_df["UnixTimeMillis"].to_numpy(dtype=np.float64)
            y = s.to_numpy(dtype=np.float64)
            mask = np.isfinite(y)
            trip_df[col] = np.interp(x, x[mask], y[mask])
        elif n_valid == 1:
            trip_df[col] = s.ffill().bfill()
        else:
            trip_df[col] = global_med
    return trip_df


tmp = tmp.groupby("tripId", group_keys=False).apply(_fill_trip_nearest)
sub = sub.drop(columns=["LatitudeDegrees", "LongitudeDegrees"]).merge(
    tmp, on=["tripId", "UnixTimeMillis"], how="left", validate="one_to_one"
)

sub["stopped"] = False
sub = add_prev_post_shift(sub)

for trip, sub_trip in sub.groupby("tripId", sort=False):
    sub_trip = sub_trip.sort_values("UnixTimeMillis").copy()

    sub_stopped = sub_trip.loc[
        sub_trip["dist_prev"].notna() & (sub_trip["dist_prev"] < 2)
    ].copy()
    sub_stopped = sub_stopped.sort_values("UnixTimeMillis").reset_index(drop=True)

    if len(sub_stopped) == 0:
        continue

    full_dt = sub_trip["UnixTimeMillis"].diff().to_numpy()
    full_dt_by_time = pd.Series(full_dt, index=sub_trip["UnixTimeMillis"].to_numpy())
    sub_stopped["full_dt"] = sub_stopped["UnixTimeMillis"].map(full_dt_by_time)

    sub_stopped["UnixTimeMillis_diff"] = sub_stopped["UnixTimeMillis"].diff()
    sub_stopped["big_timeshift"] = (sub_stopped["UnixTimeMillis_diff"] > 2_000) | (
        sub_stopped["full_dt"].fillna(0) > 2_000
    )
    sub_stopped["time_group"] = sub_stopped["big_timeshift"].astype("int").cumsum()

    for stop_group, d in sub_stopped.groupby("time_group"):
        tstart, tstop = d["UnixTimeMillis"].min(), d["UnixTimeMillis"].max()

        mask_trip_window = (
            (sub["tripId"] == trip)
            & (sub["UnixTimeMillis"] >= tstart)
            & (sub["UnixTimeMillis"] <= tstop)
        )
        stopped_len = int(mask_trip_window.sum())

        if stopped_len >= 20:
            buffer = 800
            mask_trip_buffer = (
                (sub["tripId"] == trip)
                & (sub["UnixTimeMillis"] >= (tstart - buffer))
                & (sub["UnixTimeMillis"] <= (tstop + buffer))
            )

            latDegmean = np.nanmean(
                sub.loc[mask_trip_buffer, "LatitudeDegrees"].to_numpy()
            )
            lngDegmean = np.nanmean(
                sub.loc[mask_trip_buffer, "LongitudeDegrees"].to_numpy()
            )

            if np.isfinite(latDegmean) and np.isfinite(lngDegmean):
                sub.loc[mask_trip_window, "LatitudeDegrees"] = latDegmean
                sub.loc[mask_trip_window, "LongitudeDegrees"] = lngDegmean
                sub.loc[mask_trip_window, "stopped"] = True

sub["stopped"] = sub["stopped"].fillna(False)

sub = add_prev_post_shift(sub)



## === cell 22
pred_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]

sub_pred = sub[pred_cols].copy()
sub_pred = sub_pred.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)

for c in ["LatitudeDegrees", "LongitudeDegrees"]:
    sub_pred.loc[~np.isfinite(sub_pred[c].to_numpy()), c] = np.nan

global_lat_med = np.nanmedian(sub_pred["LatitudeDegrees"].to_numpy())
global_lng_med = np.nanmedian(sub_pred["LongitudeDegrees"].to_numpy())


def _fill_trip_nearest(trip_df):
    trip_df = trip_df.sort_values("UnixTimeMillis").copy()
    for col, global_med in [
        ("LatitudeDegrees", global_lat_med),
        ("LongitudeDegrees", global_lng_med),
    ]:
        s = trip_df[col]
        n_valid = int(s.notna().sum())
        if n_valid >= 2:
            x = trip_df["UnixTimeMillis"].to_numpy(dtype=np.float64)
            y = s.to_numpy(dtype=np.float64)
            mask = np.isfinite(y)
            trip_df[col] = np.interp(x, x[mask], y[mask])
        elif n_valid == 1:
            trip_df[col] = s.ffill().bfill()
        else:
            trip_df[col] = global_med
    return trip_df


sub_pred = sub_pred.groupby("tripId", group_keys=False).apply(_fill_trip_nearest)

sub_pred["LatitudeDegrees"] = sub_pred.groupby("tripId")["LatitudeDegrees"].transform(
    lambda s: s.ffill().bfill()
)
sub_pred["LongitudeDegrees"] = sub_pred.groupby("tripId")["LongitudeDegrees"].transform(
    lambda s: s.ffill().bfill()
)

sub_pred["LatitudeDegrees"] = sub_pred.groupby("tripId")["LatitudeDegrees"].transform(
    lambda s: s.fillna(s.median())
)
sub_pred["LongitudeDegrees"] = sub_pred.groupby("tripId")["LongitudeDegrees"].transform(
    lambda s: s.fillna(s.median())
)

sub_pred = (
    sub_pred.sort_values(["tripId", "UnixTimeMillis"])
    .groupby(["tripId", "UnixTimeMillis"], as_index=False)
    .agg({"LatitudeDegrees": "median", "LongitudeDegrees": "median"})
)

submission = ss[["tripId", "UnixTimeMillis"]].merge(
    sub_pred, on=["tripId", "UnixTimeMillis"], how="left", validate="one_to_one"
)

for c in ["LatitudeDegrees", "LongitudeDegrees"]:
    submission.loc[~np.isfinite(submission[c].to_numpy()), c] = np.nan

submission = submission.groupby("tripId", group_keys=False).apply(_fill_trip_nearest)

submission["LatitudeDegrees"] = submission.groupby("tripId")[
    "LatitudeDegrees"
].transform(lambda s: s.ffill().bfill())
submission["LongitudeDegrees"] = submission.groupby("tripId")[
    "LongitudeDegrees"
].transform(lambda s: s.ffill().bfill())

submission["LatitudeDegrees"] = submission.groupby("tripId")[
    "LatitudeDegrees"
].transform(lambda s: s.fillna(s.median()))
submission["LongitudeDegrees"] = submission.groupby("tripId")[
    "LongitudeDegrees"
].transform(lambda s: s.fillna(s.median()))

submission["LatitudeDegrees"] = submission["LatitudeDegrees"].fillna(
    submission["LatitudeDegrees"].median()
)
submission["LongitudeDegrees"] = submission["LongitudeDegrees"].fillna(
    submission["LongitudeDegrees"].median()
)

sample_index = ss.set_index(["tripId", "UnixTimeMillis"]).index
submission = (
    submission.set_index(["tripId", "UnixTimeMillis"]).loc[sample_index].reset_index()
)

submission["LatitudeDegrees"] = submission["LatitudeDegrees"].clip(-90.0, 90.0)
submission["LongitudeDegrees"] = (
    (submission["LongitudeDegrees"] + 180.0) % 360.0
) - 180.0

assert list(submission.columns) == [
    "tripId",
    "UnixTimeMillis",
    "LatitudeDegrees",
    "LongitudeDegrees",
], "Submission columns mismatch."
assert len(submission) == len(ss), "Row count mismatch vs sample_submission."
assert (
    submission[["tripId", "UnixTimeMillis"]].duplicated().sum() == 0
), "Duplicate keys."
assert (
    not submission[["LatitudeDegrees", "LongitudeDegrees"]].isna().any().any()
), "NaNs in coords."
assert np.isfinite(
    submission[["LatitudeDegrees", "LongitudeDegrees"]].to_numpy()
).all(), "Non-finite coords."

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", list(submission.columns))
print(
    "Any NaNs:",
    submission[["LatitudeDegrees", "LongitudeDegrees"]].isna().any().to_dict(),
)
print(
    "Any non-finite:",
    (
        ~np.isfinite(submission[["LatitudeDegrees", "LongitudeDegrees"]].to_numpy())
    ).any(),
)

## --- ERROR in cell 22, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAssertionError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1518370025.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    106[0m ), "Duplicate keys."
[1;32m    107[0m assert (
[0;32m--> 108[0;31m     [0;32mnot[0m [0msubmission[0m[0;34m[[0m[0;34m[[0m[0;34m"LatitudeDegrees"[0m[0;34m,[0m [0;34m"LongitudeDegrees"[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0misna[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0many[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0many[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    109[0m ), "NaNs in coords."
[1;32m    110[0m assert np.isfinite(

[0;31mAssertionError[0m: NaNs in coords.
