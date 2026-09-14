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
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import plotly.express as px

pd.set_option("display.max_columns", 500)



## === cell 1
trip_id = "2020-05-15-US-MTV-1/GooglePixel4XL"



## === cell 2
gt = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/ground_truth.csv"
)
gnss = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/device_gnss.csv"
)
imu = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/device_imu.csv"
)



## === cell 3
import glob
from dataclasses import dataclass

from tqdm import tqdm
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


def ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis):
    """
    NOTE (score-relevant fix): device_gnss.csv uses 'UnixTimeMillis' (not 'utcTimeMillis') to align with
    ground_truth/sample_submission. Interpolating on utcTimeMillis and querying with UnixTimeMillis
    can severely misalign timestamps and worsen the distance error metric.

    NOTE (robustness fix): Spline interpolation requires strictly increasing X. We sort and deduplicate
    times, and fall back to linear interpolation if the spline fails. This preserves the same overall
    interpolation-based approach while preventing NaNs that would yield invalid submissions.
    """
    ecef_columns = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    if "UnixTimeMillis" in gnss_df.columns:
        time_col = "UnixTimeMillis"
    else:
        time_col = "utcTimeMillis"

    columns = [time_col] + ecef_columns
    ecef_df = gnss_df[columns].dropna().copy()
    ecef_df = ecef_df.drop_duplicates(subset=time_col, keep="first").sort_values(
        time_col
    )
    ecef_df = ecef_df.reset_index(drop=True)

    if len(ecef_df) < 2:
        return pd.DataFrame(
            {
                "tripId": tripID,
                "UnixTimeMillis": UnixTimeMillis,
                "LatitudeDegrees": np.nan,
                "LongitudeDegrees": np.nan,
            }
        )

    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
    blh = ECEF_to_BLH(ecef)

    TIME = ecef_df[time_col].to_numpy(dtype=np.float64)
    query_t = np.asarray(UnixTimeMillis, dtype=np.float64)

    try:
        lat = InterpolatedUnivariateSpline(TIME, blh.lat.astype(np.float64), ext=3)(
            query_t
        )
        lng = InterpolatedUnivariateSpline(TIME, blh.lng.astype(np.float64), ext=3)(
            query_t
        )
    except Exception:
        lat = np.interp(
            query_t,
            TIME,
            blh.lat.astype(np.float64),
            left=blh.lat[0],
            right=blh.lat[-1],
        )
        lng = np.interp(
            query_t,
            TIME,
            blh.lng.astype(np.float64),
            left=blh.lng[0],
            right=blh.lng[-1],
        )

    return pd.DataFrame(
        {
            "tripId": tripID,
            "UnixTimeMillis": UnixTimeMillis,
            "LatitudeDegrees": np.degrees(lat),
            "LongitudeDegrees": np.degrees(lng),
        }
    )


def calc_score(tripID, pred_df, gt_df):
    d = pandas_haversine_distance(pred_df, gt_df)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])
    return score




## === cell 4
ss = pd.read_csv("../input/smartphone-decimeter-2022/sample_submission.csv")
if "tripId" not in ss.columns and "phone" in ss.columns:
    ss = ss.rename(columns={"phone": "tripId"}).copy()

required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
missing_cols = [c for c in required_cols if c not in ss.columns]
if missing_cols:
    raise ValueError(f"sample_submission missing required columns: {missing_cols}")



## === cell 5
trip_id = "2020-05-15-US-MTV-1/GooglePixel4XL"
baseline = ecef_to_lat_lng(trip_id, gnss, gt["UnixTimeMillis"].values)




## === cell 6
def visualize_traffic(
    df,
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    center=None,
    color_col="phone",
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
    """
    Create a plot of the baseline predictions vs. the ground truth
    for a given tripId
    """
    gt_ = pd.read_csv(
        f"../input/smartphone-decimeter-2022/train/{tripId}/ground_truth.csv"
    )
    gnss_ = pd.read_csv(
        f"../input/smartphone-decimeter-2022/train/{tripId}/device_gnss.csv"
    )
    imu_ = pd.read_csv(
        f"../input/smartphone-decimeter-2022/train/{tripId}/device_imu.csv"
    )

    baseline_ = ecef_to_lat_lng(tripId, gnss_, gt_["UnixTimeMillis"].values)
    baseline_["isGT"] = False
    gt_["isGT"] = True
    gt_["tripId"] = tripId

    combined_ = (
        pd.concat([baseline_, gt_[baseline_.columns]], axis=0)
        .reset_index(drop=True)
        .copy()
    )

    visualize_traffic(
        combined_,
        lat_col="LatitudeDegrees",
        lon_col="LongitudeDegrees",
        color_col="isGT",
        zoom=10,
    )




## === cell 7
from glob import glob

train_gts = glob("../input/smartphone-decimeter-2022/train/*/*/ground_truth.csv")
trip_ids = ["/".join(p.split("/")[-3:-1]) for p in train_gts]



## === cell 8
tripId = trip_ids[10]
gt = pd.read_csv(f"../input/smartphone-decimeter-2022/train/{tripId}/ground_truth.csv")
gnss = pd.read_csv(f"../input/smartphone-decimeter-2022/train/{tripId}/device_gnss.csv")
imu = pd.read_csv(f"../input/smartphone-decimeter-2022/train/{tripId}/device_imu.csv")
baseline = ecef_to_lat_lng(tripId, gnss, gt["UnixTimeMillis"].values)
baseline["isGT"] = False
gt["isGT"] = True
gt["tripId"] = tripId

combined = (
    pd.concat([baseline, gt[baseline.columns]], axis=0).reset_index(drop=True).copy()
)



## === cell 9
import glob

INPUT_PATH = "../input/smartphone-decimeter-2022"

sample_df = pd.read_csv(f"{INPUT_PATH}/sample_submission.csv")
if "tripId" not in sample_df.columns and "phone" in sample_df.columns:
    sample_df = sample_df.rename(columns={"phone": "tripId"}).copy()

pred_dfs = []
for dirname in tqdm(sorted(glob.glob(f"{INPUT_PATH}/test/*/*"))):
    drive, phone = dirname.split("/")[-2:]
    tripID = f"{drive}/{phone}"
    gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv")
    UnixTimeMillis = sample_df[sample_df["tripId"] == tripID][
        "UnixTimeMillis"
    ].to_numpy()
    pred_dfs.append(ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis))
sub_df = pd.concat(pred_dfs, ignore_index=True)

baselines = []
gts = []
for dirname in tqdm(sorted(glob.glob(f"{INPUT_PATH}/train/*/*"))):
    drive, phone = dirname.split("/")[-2:]
    tripID = f"{drive}/{phone}"
    gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv", low_memory=False)
    gt_df = pd.read_csv(f"{dirname}/ground_truth.csv", low_memory=False)
    gt_df["tripId"] = tripID
    baseline_df = ecef_to_lat_lng(tripID, gnss_df, gt_df["UnixTimeMillis"].to_numpy())
    baselines.append(baseline_df)
    gts.append(gt_df)
baselines = pd.concat(baselines, ignore_index=True)
gts = pd.concat(gts, ignore_index=True)



## === cell 10
baselines["group"] = "train_baseline"
sub_df["group"] = "submission_baseline"
gts["group"] = "train_ground_truth"
combined = pd.concat([baselines, sub_df, gts], ignore_index=True).copy()



## === cell 11
sf_paths = combined.query("LatitudeDegrees > 36").copy()
la_paths = combined.query("LatitudeDegrees < 36").copy()




## === cell 12
def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points on the earth."""
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist


def add_prev_post_shift(
    df,
    lat_col="LatitudeDegrees",
    lng_col="LongitudeDegrees",
    dist_suffix="",
    sortby=("tripId", "UnixTimeMillis"),
):
    df = df.sort_values(list(sortby)).reset_index(drop=True)
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
    return df




## === cell 13
baselines2 = add_prev_post_shift(baselines)
baselines2["UnixTimeMillis_prev_diff"] = (
    baselines2["UnixTimeMillis"] - baselines2["UnixTimeMillis_shift1"]
)
baselines2["speed_calc"] = (
    baselines2["dist_prev"] / baselines2["UnixTimeMillis_prev_diff"]
)

sub_df2 = add_prev_post_shift(sub_df)
sub_df2["UnixTimeMillis_prev_diff"] = (
    sub_df2["UnixTimeMillis"] - sub_df2["UnixTimeMillis_shift1"]
)
sub_df2["speed_calc"] = sub_df2["dist_prev"] / sub_df2["UnixTimeMillis_prev_diff"]




## === cell 14
def do_postprocess(sub_df, thres=1):
    sub = sub_df.copy()
    if "stopped" not in sub.columns:
        sub["stopped"] = False

    for c, sub_stopped in sub.groupby("tripId"):
        sub_stopped = sub_stopped.loc[sub_stopped["dist_prev"] < thres].copy()
        sub_stopped["UnixTimeMillis_diff"] = sub_stopped["UnixTimeMillis"].diff()
        sub_stopped["big_timeshift"] = sub_stopped["UnixTimeMillis_diff"] > 2_000
        sub_stopped["time_group"] = sub_stopped["big_timeshift"].astype("int").cumsum()

        for stop_group, d in sub_stopped.groupby("time_group"):
            tstart, tstop = d["UnixTimeMillis"].min(), d["UnixTimeMillis"].max()
            stopped_len = len(
                sub.loc[
                    (sub["UnixTimeMillis"] >= tstart) & (sub["UnixTimeMillis"] <= tstop)
                ]
            )
            if stopped_len >= 20:
                buffer = 800
                latDegmean = sub.loc[
                    (sub["UnixTimeMillis"] >= (tstart - buffer))
                    & (sub["UnixTimeMillis"] <= (tstop + buffer))
                ]["LatitudeDegrees"].mean()
                lngDegmean = sub.loc[
                    (sub["UnixTimeMillis"] >= (tstart - buffer))
                    & (sub["UnixTimeMillis"] <= (tstop + buffer))
                ]["LongitudeDegrees"].mean()

                sub.loc[
                    (sub["UnixTimeMillis"] >= tstart)
                    & (sub["UnixTimeMillis"] <= tstop),
                    "LatitudeDegrees",
                ] = latDegmean
                sub.loc[
                    (sub["UnixTimeMillis"] >= tstart)
                    & (sub["UnixTimeMillis"] <= tstop),
                    "LongitudeDegrees",
                ] = lngDegmean
                sub.loc[
                    (sub["UnixTimeMillis"] >= tstart)
                    & (sub["UnixTimeMillis"] <= tstop),
                    "stopped",
                ] = True
    sub["stopped"] = sub["stopped"].fillna(False)
    return sub




## === cell 15
sub_pp = do_postprocess(sub_df2, thres=1)



## === cell 16
if "tripId" not in gts.columns:
    gt_paths = glob.glob(
        "../input/smartphone-decimeter-2022/train/*/*/ground_truth.csv"
    )
    gt_parts = []
    for p in gt_paths:
        tripID = "/".join(p.split("/")[-3:-1])
        gt_df = pd.read_csv(p, low_memory=False)
        gt_df["tripId"] = tripID
        gt_parts.append(gt_df)
    gts_with_tripid = pd.concat(gt_parts, ignore_index=True)
else:
    gts_with_tripid = gts

baselines2_pp = do_postprocess(baselines2, thres=1)
scores = []
for tripID in baselines2_pp["tripId"].unique():
    trip_pred = baselines2_pp.loc[
        baselines2_pp["tripId"] == tripID, ["LatitudeDegrees", "LongitudeDegrees"]
    ].reset_index(drop=True)
    trip_gt = gts_with_tripid.loc[
        gts_with_tripid["tripId"] == tripID, ["LatitudeDegrees", "LongitudeDegrees"]
    ].reset_index(drop=True)
    if len(trip_pred) == len(trip_gt) and len(trip_pred) > 0:
        d = pandas_haversine_distance(trip_pred, trip_gt)
        scores.append(np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)]))
mean_score = float(np.mean(scores)) if len(scores) else float("nan")
print(f"mean_score (train, baseline+pp) = {mean_score:.3f}")



## === cell 17
ss_keys = ss[["tripId", "UnixTimeMillis"]].copy()

sub_df_dedup = (
    sub_df[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]
    .sort_values(["tripId", "UnixTimeMillis"])
    .drop_duplicates(subset=["tripId", "UnixTimeMillis"], keep="first")
    .reset_index(drop=True)
)

sub_pp_dedup = (
    sub_pp[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]
    .sort_values(["tripId", "UnixTimeMillis"])
    .drop_duplicates(subset=["tripId", "UnixTimeMillis"], keep="first")
    .reset_index(drop=True)
)

base_pred = ss_keys.merge(
    sub_df_dedup,
    on=["tripId", "UnixTimeMillis"],
    how="left",
)

pp_pred = ss_keys.merge(
    sub_pp_dedup,
    on=["tripId", "UnixTimeMillis"],
    how="left",
    suffixes=("", "_pp"),
)

sub_out = base_pred.copy()
mask_pp = pp_pred["LatitudeDegrees"].notna() & pp_pred["LongitudeDegrees"].notna()
sub_out.loc[mask_pp, ["LatitudeDegrees", "LongitudeDegrees"]] = pp_pred.loc[
    mask_pp, ["LatitudeDegrees", "LongitudeDegrees"]
].to_numpy()

sub_out = sub_out.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)
sub_out[["LatitudeDegrees", "LongitudeDegrees"]] = (
    sub_out.groupby("tripId")[["LatitudeDegrees", "LongitudeDegrees"]].ffill().bfill()
)

sub_out = ss_keys.merge(sub_out, on=["tripId", "UnixTimeMillis"], how="left")

if sub_out[["LatitudeDegrees", "LongitudeDegrees"]].isna().any().any():
    nan_trip_mask = (
        sub_out["LatitudeDegrees"].isna() | sub_out["LongitudeDegrees"].isna()
    )
    nan_trips = sub_out.loc[nan_trip_mask, "tripId"].unique().tolist()

    first_base = (
        sub_df_dedup.sort_values(["tripId", "UnixTimeMillis"])
        .groupby("tripId")[["LatitudeDegrees", "LongitudeDegrees"]]
        .first()
    )

    first_pp = (
        sub_pp_dedup.sort_values(["tripId", "UnixTimeMillis"])
        .groupby("tripId")[["LatitudeDegrees", "LongitudeDegrees"]]
        .first()
    )

    for t in nan_trips:
        tmask = (sub_out["tripId"] == t) & (
            sub_out["LatitudeDegrees"].isna() | sub_out["LongitudeDegrees"].isna()
        )
        if t in first_base.index:
            lat0 = float(first_base.loc[t, "LatitudeDegrees"])
            lon0 = float(first_base.loc[t, "LongitudeDegrees"])
            sub_out.loc[tmask, "LatitudeDegrees"] = lat0
            sub_out.loc[tmask, "LongitudeDegrees"] = lon0
        elif t in first_pp.index:
            lat0 = float(first_pp.loc[t, "LatitudeDegrees"])
            lon0 = float(first_pp.loc[t, "LongitudeDegrees"])
            sub_out.loc[tmask, "LatitudeDegrees"] = lat0
            sub_out.loc[tmask, "LongitudeDegrees"] = lon0

if sub_out[["LatitudeDegrees", "LongitudeDegrees"]].isna().any().any():
    global_source = pd.concat(
        [
            sub_df_dedup[["LatitudeDegrees", "LongitudeDegrees"]],
            sub_pp_dedup[["LatitudeDegrees", "LongitudeDegrees"]],
        ],
        ignore_index=True,
    ).dropna()
    if len(global_source) == 0:
        raise ValueError(
            "No valid predicted coordinates available in sub_df/sub_pp to fill missing submission keys."
        )
    global_lat0 = float(global_source.iloc[0]["LatitudeDegrees"])
    global_lon0 = float(global_source.iloc[0]["LongitudeDegrees"])

    nan_mask = sub_out["LatitudeDegrees"].isna() | sub_out["LongitudeDegrees"].isna()
    sub_out.loc[nan_mask, "LatitudeDegrees"] = global_lat0
    sub_out.loc[nan_mask, "LongitudeDegrees"] = global_lon0

if sub_out[["LatitudeDegrees", "LongitudeDegrees"]].isna().any().any():
    raise ValueError(
        "Still have NaNs in submission lat/lon after base+overlay+ffill/bfill+fallback; cannot write valid submission."
    )

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print("Submission columns:", list(sub_out.columns))
print(
    "Any remaining NaNs:",
    sub_out[["LatitudeDegrees", "LongitudeDegrees"]].isna().any().to_dict(),
)


## --- ERROR in cell 17, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/88222019.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     89[0m     ).dropna()
[1;32m     90[0m     [0;32mif[0m [0mlen[0m[0;34m([0m[0mglobal_source[0m[0;34m)[0m [0;34m==[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 91[0;31m         raise ValueError(
[0m[1;32m     92[0m             [0;34m"No valid predicted coordinates available in sub_df/sub_pp to fill missing submission keys."[0m[0;34m[0m[0;34m[0m[0m
[1;32m     93[0m         )

[0;31mValueError[0m: No valid predicted coordinates available in sub_df/sub_pp to fill missing submission keys.
