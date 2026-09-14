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
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

bokeh==3.7.3
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
scipy==1.15.3
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
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Target score

0.95487

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd


from bokeh.io import output_notebook

from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper, BasicTicker, HoverTool
from bokeh.layouts import row, column, grid

from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Viridis256, Spectral7, Inferno
from bokeh.transform import factor_cmap

import os

import seaborn as sns
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras


import sklearn

from sklearn.model_selection import train_test_split


from scipy.stats import skew

output_notebook()


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
files = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
        
raw_data = pd.read_csv(files[1]).set_index('Id')

test_df = pd.read_csv(files[2]).set_index('Id')


## === cell 2
print(f'Number of samples in train.csv : {len(raw_data)}\n')

print(f'Number of rows in test.csv : {len(test_df)}\n')

print(f'Number of features for both training and testing : {len(raw_data.columns)}\n ')
raw_data.head(5)


## === cell 3
print(raw_data.isna().sum()[0:10])



## === cell 4
raw_data.describe().transpose()


## === cell 5
clean_df = raw_data.copy()


clean_df.drop(columns = ['Soil_Type7', 'Soil_Type15'], inplace = True)

test_df.drop(columns = ['Soil_Type7', 'Soil_Type15'], inplace = True)


train_target = clean_df.pop('Cover_Type')

clean_df['Hillshade_9am'] = np.clip(clean_df['Hillshade_9am'].values, 0,255)
test_df['Hillshade_9am'] = np.clip(test_df['Hillshade_9am'].values, 0,255)

clean_df['Hillshade_Noon'] = np.clip(clean_df['Hillshade_Noon'].values, 0,255)
test_df['Hillshade_Noon'] = np.clip(test_df['Hillshade_Noon'].values, 0,255)


clean_df['Hillshade_3pm'] = np.clip(clean_df['Hillshade_3pm'].values, 0,255)
test_df['Hillshade_3pm'] = np.clip(test_df['Hillshade_3pm'].values, 0,255)





clean_df['Aspect'] = np.mod(clean_df['Aspect'].values, 360)
test_df['Aspect'] = np.mod(test_df['Aspect'].values, 360)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/257438960.py in <cell line: 0>()
      4 clean_df.drop(columns = ['Soil_Type7', 'Soil_Type15'], inplace = True)
      5 
----> 6 test_df.drop(columns = ['Soil_Type7', 'Soil_Type15'], inplace = True)
      7 
      8 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['Soil_Type7', 'Soil_Type15'] not found in axis"

## === cell 6
clean_df['HighWater'] = (clean_df['Vertical_Distance_To_Hydrology']< 0).astype(np.int16)

test_df['HighWater'] = (test_df['Vertical_Distance_To_Hydrology']< 0).astype(np.int16)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'Vertical_Distance_To_Hydrology'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/877035646.py in <cell line: 0>()
      1 clean_df['HighWater'] = (clean_df['Vertical_Distance_To_Hydrology']< 0).astype(np.int16)
      2 
----> 3 test_df['HighWater'] = (test_df['Vertical_Distance_To_Hydrology']< 0).astype(np.int16)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'Vertical_Distance_To_Hydrology'

## === cell 7
forest_type = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'CottonWood/willow', 'Aspen', 'Douglas-fir', 'Krummholz']

class_ratio = train_target.value_counts().sort_index().to_list()



source = ColumnDataSource(data = dict(x_values = forest_type, y_values = class_ratio, color = Spectral7))

TOOLTIPS = [
    ("Type of forest", "@x_values"),
    ('value', "@y_values")
]



p = figure(x_range = forest_type,
           y_axis_type="log",
           y_range = [0.1,10**7],
           width = 800,
           height = 600,
           title = 'Target feature proportion by class', tooltips = TOOLTIPS)

p.vbar(x = 'x_values',
       top = 'y_values',
       source = source, 
       alpha = 0.8,
       line_color = 'white',
       color = 'color',
       bottom=0.1,
      width = 0.9)

p.xgrid.grid_line_color = None
p.background_fill_color = '#fafafa'

show(p)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/331502686.py in <cell line: 0>()
      1 forest_type = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'CottonWood/willow', 'Aspen', 'Douglas-fir', 'Krummholz']
      2 
----> 3 class_ratio = train_target.value_counts().sort_index().to_list()
      4 
      5 

NameError: name 'train_target' is not defined

## === cell 8
number_sample = len(train_target)

for idx,forest in enumerate(forest_type):
    
    print(f'Class {idx+1} corresponding to {forest} forest represents {class_ratio[idx]/number_sample:.5%} of the total dataset')


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1826287139.py in <cell line: 0>()
----> 1 number_sample = len(train_target)
      2 
      3 for idx,forest in enumerate(forest_type):
      4 
      5     print(f'Class {idx+1} corresponding to {forest} forest represents {class_ratio[idx]/number_sample:.5%} of the total dataset')

NameError: name 'train_target' is not defined

## === cell 9
cat_features = [ 'Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
       'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
       'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
       'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
       'Soil_Type39', 'Soil_Type40','HighWater']



number_cat_features = len(cat_features)

figures = []

palette = [Viridis256[63], Viridis256[191]]

for num, cat in enumerate(cat_features):
        
    cat_serie = clean_df[cat].value_counts()
    
    source = ColumnDataSource(data = 
                              dict(x_values = cat_serie.index.astype('str').to_list(),
                                   y_values = cat_serie.to_list()))
    
    p = figure(x_range = source.data['x_values'],
               width = 200, height = 200,
               title = 'Proportion of '+cat,
               toolbar_location=None,
               tools="")
    
    p.vbar(x = 'x_values',
           top ='y_values',
           source = source,
           line_color='white',
           fill_color=factor_cmap('x_values', palette=palette, factors = source.data['x_values']),
           width = 0.9)
    
    p.title.text_font_size = '8pt'
    p.xgrid.grid_line_color = None
    p.yaxis.visible = False
    
    figures.append(p)
    

show(grid(figures, ncols = 6))
    
    


## === cell 10
num_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points']


def plot_violin(dataframe, features, sample_size):
    
    distrib_df = dataframe[features].sample(sample_size).melt(var_name = 'column_name', value_name = 'values')
    
    
    plt.figure(figsize = (20,8))
    
    sns.violinplot(x = 'column_name', y = 'values', data = distrib_df)
    
    plt.title('Distribution of numerical features without normalisation')
    plt.xlabel(" ")
    plt.xticks(rotation=45)
    plt.show()
    
plot_violin(clean_df,num_features, 100000)
    
    


## === cell 11
def make_histogram(dataframe, features):
    
    figures = []
    
    for feature in features : 
    
        hist_val, edges_val = np.histogram(dataframe[feature], density=True, bins=50)

        p = figure(width = 500, height = 300, title = 'Histogram of raw '+ feature, tools='', background_fill_color="#fafafa")

        p.quad(top=hist_val, bottom=0, left=edges_val[:-1], right=edges_val[1:],
               fill_color="navy", line_color="white", alpha=0.5)
        p.y_range.start = 0
        p.xaxis.axis_label = 'x'
        p.yaxis.axis_label = 'Pr(x)'
        p.grid.grid_line_color="white"
        
        figures.append(p)
        
    return figures

histos = make_histogram(clean_df,num_features)

show(grid(histos, ncols = 3))
    
    


## === cell 12
print('Training data ----------- \n')

for feat in num_features : 
    print('Skewness of the ' +feat+ f' distribution is : {skew(clean_df[feat])}')
    

print('Testing data ----------- \n')

for feat in num_features : 
    print('Skewness of the ' +feat+ f' distribution is : {skew(test_df[feat])}')


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'Elevation'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3991538375.py in <cell line: 0>()
      8 
      9 for feat in num_features :
---> 10     print('Skewness of the ' +feat+ f' distribution is : {skew(test_df[feat])}')

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'Elevation'

## === cell 13
def handle_neg_skew(dataframe, feature):
    
    find_max = (dataframe[feature]+1).max()
        

    return  np.sqrt(find_max - dataframe.pop(feature))




clean_df['Sqrt_Hillshade_Noon'] = handle_neg_skew(clean_df,'Hillshade_Noon')
test_df['Sqrt_Hillshade_Noon'] = handle_neg_skew(test_df,'Hillshade_Noon')



clean_df['Sqrt_Hillshade_9am'] = handle_neg_skew(clean_df,'Hillshade_9am')
test_df['Sqrt_Hillshade_9am'] = handle_neg_skew(test_df,'Hillshade_9am')




print('Training data')
print('Skewness of Sqrt_Hillshade_Noon distribution is : ',skew(clean_df['Sqrt_Hillshade_Noon']))
print('Skewness of Sqrt_Hillshade_9am distribution is : ',skew(clean_df['Sqrt_Hillshade_9am']))

print('Testing data')
print('Skewness of Sqrt_Hillshade_Noon distribution is : ',skew(test_df['Sqrt_Hillshade_Noon']))
print('Skewness of Sqrt_Hillshade_9am distribution is : ',skew(test_df['Sqrt_Hillshade_9am']))


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'Hillshade_Noon'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/723660641.py in <cell line: 0>()
     10 
     11 clean_df['Sqrt_Hillshade_Noon'] = handle_neg_skew(clean_df,'Hillshade_Noon')
---> 12 test_df['Sqrt_Hillshade_Noon'] = handle_neg_skew(test_df,'Hillshade_Noon')
     13 
     14 

/tmp/ipykernel_11/723660641.py in handle_neg_skew(dataframe, feature)
      1 def handle_neg_skew(dataframe, feature):
      2 
----> 3     find_max = (dataframe[feature]+1).max()
      4 
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'Hillshade_Noon'

## === cell 14
histos = make_histogram(clean_df,['Sqrt_Hillshade_9am','Sqrt_Hillshade_Noon'])

show(grid(histos, ncols = 2))


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'Sqrt_Hillshade_9am'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2090033528.py in <cell line: 0>()
----> 1 histos = make_histogram(clean_df,['Sqrt_Hillshade_9am','Sqrt_Hillshade_Noon'])
      2 
      3 show(grid(histos, ncols = 2))

/tmp/ipykernel_11/3487183580.py in make_histogram(dataframe, features)
      5     for feature in features :
      6 
----> 7         hist_val, edges_val = np.histogram(dataframe[feature], density=True, bins=50)
      8 
      9         p = figure(width = 500, height = 300, title = 'Histogram of raw '+ feature, tools='', background_fill_color="#fafafa")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'Sqrt_Hillshade_9am'

## === cell 15
clean_df['target'] = train_target


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/798568860.py in <cell line: 0>()
----> 1 clean_df['target'] = train_target

NameError: name 'train_target' is not defined

## === cell 16
def plot_correlation_heatmap(dataframe_corr, width = 900, height = 800):
    
    dataframe_corr.index.name = 'Features1'
    dataframe_corr.columns.name = 'Features2'

    corr_matrix = pd.DataFrame(dataframe_corr.stack(), columns = ['correlation']).reset_index()



    mapper = LinearColorMapper(palette = Viridis256,
                              low = corr_matrix['correlation'].min(),
                              high = corr_matrix['correlation'].max())


    TOOLS = "hover,save,pan, box_zoom,reset"

    p = figure(title = 'Correlation Matrix',
              x_range = corr_matrix['Features1'].drop_duplicates().to_list(),
              y_range = corr_matrix['Features2'].drop_duplicates().to_list(),
              x_axis_location = 'below',
              width = width,
              height = height,
              tools = TOOLS,
              toolbar_location = 'left')


    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "7px"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect(x='Features1', y="Features2", width=1, height=1,
           source=corr_matrix,
           fill_color={'field': 'correlation', 'transform': mapper},
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper,
                         major_label_text_font_size="7px",
                         ticker=BasicTicker(desired_num_ticks=256),
                         border_line_color=None)
    p.add_layout(color_bar, 'right')

    return p



corr_df = clean_df.corr()

p = plot_correlation_heatmap(corr_df)

show(p)


## === cell 17
clean_df['EVDtH'] = clean_df['Elevation'] -clean_df['Vertical_Distance_To_Hydrology']
test_df['EVDtH'] = test_df['Elevation'] -test_df['Vertical_Distance_To_Hydrology']


clean_df['EHDtH'] = clean_df['Elevation'] -clean_df['Horizontal_Distance_To_Hydrology']*0.2
test_df['EHDtH'] = test_df['Elevation'] -test_df['Horizontal_Distance_To_Hydrology']*0.2


soil_features = [x for x in clean_df.columns if x.startswith('Soil_Type')]

clean_df['Soil_type_count'] = clean_df[soil_features].sum(axis = 1)
test_df['Soil_typ_count'] = test_df[soil_features].sum(axis = 1)


wilderness_features = [x for x in clean_df.columns if x.startswith('Wilderness_Area')]

clean_df['Wilderness_area_count'] = clean_df[wilderness_features].sum(axis=1)
test_df['Wilderness_area_count'] = test_df[wilderness_features].sum(axis=1)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'Elevation'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/98070319.py in <cell line: 0>()
      1 clean_df['EVDtH'] = clean_df['Elevation'] -clean_df['Vertical_Distance_To_Hydrology']
----> 2 test_df['EVDtH'] = test_df['Elevation'] -test_df['Vertical_Distance_To_Hydrology']
      3 
      4 
      5 clean_df['EHDtH'] = clean_df['Elevation'] -clean_df['Horizontal_Distance_To_Hydrology']*0.2

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'Elevation'

## === cell 18
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df





clean_df = reduce_mem_usage(clean_df)

test_df = reduce_mem_usage(test_df)


## === cell 19
new_num_features = ['Elevation', 'Aspect', 'Slope',
                    'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology',
                    'Horizontal_Distance_To_Roadways',
                    'Sqrt_Hillshade_9am', 'Sqrt_Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points', 'EVDtH','EHDtH']


mean_val = clean_df[new_num_features].mean()
standard_dev_val = clean_df[new_num_features].std()   


clean_df[new_num_features] = (clean_df[new_num_features] - mean_val )/ standard_dev_val



test_df[new_num_features] = (test_df[new_num_features] - mean_val )/ standard_dev_val  


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1879661557.py in <cell line: 0>()
      7 
      8 
----> 9 mean_val = clean_df[new_num_features].mean()
     10 standard_dev_val = clean_df[new_num_features].std()
     11 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['Sqrt_Hillshade_9am', 'EHDtH'] not in index"

## === cell 20
BATCH_SIZE = 2048

EPOCHS = 35

NUMBER_FEATURE = 57


## === cell 21
clean_df = clean_df.loc[clean_df['target'] !=5]

mapping = {1: 0,
           2: 1,
           3: 2,
           4: 3,
           6: 4,
           7: 5}


clean_df['target'].replace(mapping, inplace=True)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'target'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2336568226.py in <cell line: 0>()
----> 1 clean_df = clean_df.loc[clean_df['target'] !=5]
      2 
      3 mapping = {1: 0,
      4            2: 1,
      5            3: 2,

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'target'

## === cell 22
my_dict = clean_df['target'].value_counts().to_dict()

class_weight = {}

total = len(clean_df)

for key, value in my_dict.items():
    class_weight[key] = (1/value) * (total/len(my_dict))
    
print( class_weight)


train_target = np.array(clean_df.pop('target'))


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'target'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1769623597.py in <cell line: 0>()
----> 1 my_dict = clean_df['target'].value_counts().to_dict()
      2 
      3 class_weight = {}
      4 
      5 total = len(clean_df)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'target'

## === cell 23
x_train, x_val, y_train, y_val = train_test_split(np.asarray(clean_df),train_target , test_size = 0.15, stratify =train_target )




def build_dataset(features_matrix, target):
    
    dataset = tf.data.Dataset.from_tensor_slices((features_matrix,target))
    dataset = dataset.shuffle(buffer_size = 1000).batch(BATCH_SIZE, drop_remainder = True).prefetch(1)
    
    return dataset


train_dataset = build_dataset(x_train,y_train)

val_dataset = build_dataset(x_val,y_val)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1551791285.py in <cell line: 0>()
----> 1 x_train, x_val, y_train, y_val = train_test_split(np.asarray(clean_df),train_target , test_size = 0.15, stratify =train_target )
      2 
      3 
      4 #Stratify option allows to split the dataset in order to keep proportion regarding train_target,
      5 #So classes proportions will remain the same accross training and validation set

NameError: name 'train_target' is not defined

## === cell 24
def build_model(input_size):
    
    model = tf.keras.Sequential([
        keras.layers.InputLayer(input_shape = (input_size,)),
        keras.layers.Dense(units = 256, activation = 'relu'),
        
        keras.layers.BatchNormalization(),
        keras.layers.Dense(units = 128, activation = 'relu'),
        
        keras.layers.BatchNormalization(),
        keras.layers.Dense(units = 128, activation = 'relu'),
        
        keras.layers.BatchNormalization(),
        keras.layers.Dense(units = 64, activation = 'relu'),
        keras.layers.Dense(units = 6, activation = 'softmax'),
        
    ])
    
    return model

    


## === cell 25
mymodel = build_model(input_size = NUMBER_FEATURE)


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',
                                              patience = 2,
                                              factor = 0.5,
                                              verbose = 1)


early_stop = keras.callbacks.EarlyStopping(monitor ='val_accuracy',
                                           patience = 4,
                                           restore_best_weights = True, 
                                           verbose = 1)


mymodel.compile(optimizer = keras.optimizers.Adam(learning_rate = 5e-4),
                 loss =  tf.keras.losses.SparseCategoricalCrossentropy() ,
                 metrics = ['accuracy'])


history = mymodel.fit(train_dataset,
                   batch_size = BATCH_SIZE,
                   epochs = EPOCHS,
                   validation_data = val_dataset,
                   callbacks = [reduce_lr,early_stop])
    
    


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2195094317.py in <cell line: 0>()
     19 
     20 
---> 21 history = mymodel.fit(train_dataset,
     22                    batch_size = BATCH_SIZE,
     23                    epochs = EPOCHS,

NameError: name 'train_dataset' is not defined

## === cell 26
def plot_metrics(history,epochs):
    
    titles = ['Training loss', 'Validation loss', 'Training Accuracy', 'Validation Accuracy']
    metrics = ['loss', 'val_loss', 'accuracy','val_accuracy']
    
    palette = Inferno[4]
    figures = []
    
    
    for k in range(4):
        
        p = figure(width = 600, height = 400, title = titles[k])
        
        p.line(np.arange(epochs), history.history[metrics[k]], line_width = 4, color = palette[k%2+1])
        
        figures.append(p)
        
    show(grid([figures[:2], figures[2:]]))
    
plot_metrics(history,EPOCHS)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3033188420.py in <cell line: 0>()
     18     show(grid([figures[:2], figures[2:]]))
     19 
---> 20 plot_metrics(history,EPOCHS)

NameError: name 'history' is not defined

## === cell 27
ids = test_df.index


predictions = mymodel.predict(np.asarray(test_df), 
               batch_size = 128,
               verbose = 1)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/666531466.py in <cell line: 0>()
      2 
      3 
----> 4 predictions = mymodel.predict(np.asarray(test_df), 
      5                batch_size = 128,
      6                verbose = 1)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    225                     None,
    226                 }:
--> 227                     raise ValueError(
    228                         f'Input {input_index} of layer "{layer_name}" is '
    229                         f"incompatible with the layer: expected axis {axis} "

ValueError: Exception encountered when calling Sequential.call().

Input 0 of layer "dense" is incompatible with the layer: expected axis -1 of input shape to have value 57, but received input with shape (128, 1)

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(128, 1), dtype=int8)
  • training=False
  • mask=None

## === cell 28
pred_df = pd.DataFrame({'pred_val' :  np.argmax(predictions, axis = 1)})



mapping = {0: 1,
           1: 2,
           2: 3,
           3: 4,
           4: 6,
           5: 7}


pred_df['pred_val'].replace(mapping, inplace=True)



submission_df = pd.DataFrame(data = {'Id' : ids, 'Cover_Type' : pred_df.values.reshape(-1,)}).set_index('Cover_Type')

submission_df.to_csv('submission.csv', header=True)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2022105446.py in <cell line: 0>()
----> 1 pred_df = pd.DataFrame({'pred_val' :  np.argmax(predictions, axis = 1)})
      2 
      3 
      4 
      5 mapping = {0: 1,

NameError: name 'predictions' is not defined
