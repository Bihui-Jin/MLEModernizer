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

3.8

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
missingno==0.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            if hasattr(_message_factory, "GetMessageClass"):
                return _message_factory.GetMessageClass(descriptor)
            raise AttributeError(
                "MessageFactory has no GetPrototype/GetMessageClass available"
            )

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

import missingno as msno
import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda

import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")


## === cell 1
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


## === cell 2
train_df.head()


## === cell 3
plt.subplot(121)
msno.bar(train_df)
plt.title("Missing values in Training Data", fontsize = 20, color = 'Red')

plt.subplot(122)
msno.bar(test_df)
plt.title("Missing values in Test Data", fontsize = 20, color = 'Blue')

plt.show()


## === cell 4
print(f'Total unique patients are {train_df.Patient.nunique()} out of total {len(train_df.Patient)} patients')


## === cell 5
go.Figure(go.Pie(labels = train_df.Sex.value_counts().keys().tolist(), 
                       values = train_df.Sex.value_counts().values.tolist(), 
                       marker = dict(colors=['red']), hoverinfo = "value", pull=[0, 0.1]), 
          layout = go.Layout(title = {'text':"Gender Distribution", 'x':0.5}, font=dict(family="Courier New, monospace",
                                                                                                size=18,
                                                                                                color="RebeccaPurple")))


## === cell 6
go.Figure(go.Pie(labels = train_df.SmokingStatus.value_counts().keys().tolist(), 
                       values = train_df.SmokingStatus.value_counts().values.tolist(), 
                       marker = dict(colors=['pink', 'blue', 'purple']), hoverinfo = "value", hole = 0.3), 
          layout = go.Layout(title = {'text':"Smoking Status", 'x':0.425}, font=dict(family="Courier New, monospace",
                                                                                                size=18,
                                                                                                color="RebeccaPurple")))


## === cell 7
train_df.groupby('Sex')['SmokingStatus'].value_counts()


## === cell 8
fig = go.Figure(data=[
    go.Bar(name='Smoker', x= train_df.Sex.unique(), y = [train_df.groupby('Sex')['SmokingStatus'].value_counts().values[5],
                                                         train_df.groupby('Sex')['SmokingStatus'].value_counts().values[2]]),
    go.Bar(name='Non-Smoker', x= train_df.Sex.unique(), y = [train_df.groupby('Sex')['SmokingStatus'].value_counts().values[4],
                                                             train_df.groupby('Sex')['SmokingStatus'].value_counts().values[0]]),
    go.Bar(name='Ex-Smoker', x= train_df.Sex.unique(), y = [train_df.groupby('Sex')['SmokingStatus'].value_counts().values[3],
                                                             train_df.groupby('Sex')['SmokingStatus'].value_counts().values[1]]),
])
fig.update_layout(title = {'text':"Smoking Distribution by Sex", 'x':0.5}, 
                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
                  barmode ='group')
fig.show()


## === cell 9
count_df = train_df["Patient"].value_counts().reset_index()
count_df.rename(
    columns={"Patient": "Patient ID", "count": "No of Images"}, inplace=True
)

fig = px.bar(count_df, x="Patient ID", y="No of Images", color="No of Images")
fig.update_xaxes(showticklabels=False)
fig.update_layout(
    title={"text": "Distribution of Images per Patient", "x": 0.5},
    font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
)
fig.show()


## === cell 10
dicom_ids = os.listdir('../input/osic-pulmonary-fibrosis-progression/train/')
patient_sizes = [len(os.listdir('../input/osic-pulmonary-fibrosis-progression/train/' + d)) for d in dicom_ids]
dicom_df = pd.DataFrame({'Dicom_ID':dicom_ids, 'Dicom Files':patient_sizes})

fig = px.bar(dicom_df, x='Dicom_ID',y ='Dicom Files',color='Dicom Files')
fig.update_xaxes(showticklabels=False)
fig.update_layout(title = {'text':"Distribution of Dicom Files per Dicom ID", 'x':0.5}, 
                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))
fig.show()


## === cell 11
fig = ff.create_distplot([train_df.Age.values], ['Age'], colors = ['red'])
fig.update_layout(title = {'text':"Age Distribution", 'x':0.5}, 
                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))
fig.show()

fig = ff.create_distplot([train_df.Weeks.values], ['Weeks'], colors = ['blue'])
fig.update_layout(title = {'text':"Weeks Distribution", 'x':0.5}, 
                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))
fig.show()


## === cell 12
fig = ff.create_distplot([train_df.Percent.values], ['Percent'], colors = ['purple'])
fig.update_layout(title = {'text':"Percent Distribution", 'x':0.5}, 
                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))
fig.show()

fig = ff.create_distplot([train_df.FVC.values], ['FVC'], colors = ['green'])
fig.update_layout(title = {'text':"FVC Distribution", 'x':0.5}, 
                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))
fig.show()


## === cell 13
fig = px.histogram(train_df, x='Age', color='SmokingStatus', marginal="box", 
                   color_discrete_map={'Ex-smoker':'green','Never smoked':'light green','Currently smokes':'orange'})
fig.update_traces(marker_line_color='cyan',marker_line_width=1, opacity=0.8)
fig.update_layout(title = {'text':"Smoking Status by Age", 'x':0.4}, 
                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))
fig.show()


## === cell 14
fig = px.histogram(train_df, x='Age', color='Sex',marginal="box", color_discrete_map={'Male':'blue','Female':'light green'})
fig.update_traces(marker_line_color='cyan',marker_line_width=1, opacity=0.8)
fig.update_layout(title = {'text':"Sex Distribution by Age", 'x':0.45}, 
                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))
fig.show()


## === cell 15
fig = px.histogram(train_df, x='FVC', color='Sex', marginal="rug", 
                   color_discrete_map={'Male':'DarkKhaki','Female':'MediumSpringGreen'})
fig.update_traces(marker_line_color='LightSlateGrey',marker_line_width=1, opacity=0.8)
fig.update_layout(title = {'text':"Gender Distribution in FVC", 'x':0.45}, 
                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))
fig.show()


## === cell 16
fig = px.histogram(train_df, x='FVC', color='SmokingStatus', marginal="box",
                   color_discrete_map={'Ex-smoker':'#393E46','Never smoked':'MediumTurquoise','Currently smokes':'Linen'})
fig.update_traces(marker_line_color = 'black',marker_line_width = 1, opacity = 0.8)
fig.update_layout(title = {'text':"SmokingStatus Distribution in FVC", 'x':0.45}, 
                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))
fig.show()


## === cell 17
train_df.shape


## === cell 18
train_df[train_df.duplicated(subset = ['Patient','Weeks'])]


## === cell 19
train_df.drop_duplicates(keep=False, inplace = True, subset=['Patient','Weeks'])


## === cell 20
submission_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')


## === cell 21
submission_df


## === cell 22
temp_sub_df = submission_df['Patient_Week'].str.split('_', expand = True)
temp_sub_df.rename(columns = {0: 'Patient', 1: 'Weeks'}, inplace = True)


## === cell 23
submission_df = pd.concat([submission_df, temp_sub_df], axis = 1)
submission_df = submission_df[['Patient','Weeks','Confidence','Patient_Week']]


## === cell 24
test_df.head()


## === cell 25
submission_df = submission_df.merge(test_df.drop('Weeks', axis = 1), on = 'Patient')


## === cell 26
train_df["data_type"] = "Train"
test_df["data_type"] = "Val"
submission_df["data_type"] = "Test"

combined_df = pd.concat(
    [train_df, test_df, submission_df], axis=0, ignore_index=False, sort=False
)


## === cell 27
data_type = ['Train', 'Val', 'Test']
for type in data_type:
    data = combined_df.query("data_type == @type")
    print(type, "shape in combined data is ", data.shape)


## === cell 28
combined_df['Min_Weeks'] = combined_df['Weeks']
combined_df.loc[combined_df.data_type == 'Test','Min_Weeks'] = np.nan
combined_df['Min_Weeks'] = combined_df.groupby('Patient')['Min_Weeks'].transform('min')


## === cell 29
base = combined_df.loc[combined_df.Weeks == combined_df.Min_Weeks]
base = base[['Patient','FVC']].rename(columns = {'FVC':'min_FVC'})
base.drop_duplicates(keep = 'first', inplace = True, subset = ['Patient'])


## === cell 30
combined_df.Weeks = combined_df.Weeks.astype(int)
combined_df.Min_Weeks = combined_df.Min_Weeks.astype(float)


## === cell 31
combined_df = combined_df.merge(base, on='Patient', how='left')
combined_df['Deviation_Weeks'] = combined_df['Weeks'] - combined_df['Min_Weeks']
del base


## === cell 32
combined_df = pd.concat([combined_df, pd.get_dummies(combined_df[['Sex','SmokingStatus']])], axis = 1)


## === cell 33
scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(combined_df[['Age','Percent','min_FVC','Deviation_Weeks']]), 
                      columns = ['scaled_Age', 'scaled_Percent', 'scaled_FVC', 'scaled_Deviation_Weeks'])
combined_df = pd.concat([combined_df, scaled], axis = 1)


## === cell 34
combined_df


## === cell 35
feature_columns = ['Sex_Male','Sex_Female','SmokingStatus_Ex-smoker','SmokingStatus_Never smoked','SmokingStatus_Currently smokes',
                   'scaled_Age','scaled_Percent','scaled_Deviation_Weeks','scaled_FVC']


## === cell 36
train_df = combined_df.loc[combined_df.data_type == 'Train']
test_df = combined_df.loc[combined_df.data_type == 'Val']
submission_df = combined_df.loc[combined_df.data_type == 'Test']
del combined_df


## === cell 37
train_df.shape, test_df.shape, submission_df.shape


## === cell 38
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)

def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss

def make_model():
    x1 = Input((9,), name="Patient")
    x2 = Dense(100, activation="relu", name="d1")(x1)
    x3 = Dense(100, activation="relu", name="d2")(x2)
    
    p1 = Dense(3, activation="relu", name="p1")(x3)
    p2 = Dense(3, activation="relu", name="p2")(x3)
    
    preds = Lambda(lambda x3: x3[0] + tf.cumsum(x3[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = Model(x1, preds, name="CNN")
   
    model.compile(loss = mloss(0.8), optimizer = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005,
                                                                          amsgrad=False), metrics=[score])
    return model


## === cell 39
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model():
    x1 = Input((9,), name="Patient")
    x2 = Dense(100, activation="relu", name="d1")(x1)
    x3 = Dense(100, activation="relu", name="d2")(x2)

    p1 = Dense(3, activation="relu", name="p1")(x3)
    p2 = Dense(3, activation="relu", name="p2")(x3)

    preds = Lambda(lambda x3: x3[0] + tf.cumsum(x3[1], axis=1), name="preds")([p1, p2])

    model = Model(x1, preds, name="CNN")

    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.005,
            amsgrad=False,
        ),
        metrics=[score],
    )
    return model


## === cell 40
y = train_df['FVC'].values
z = train_df[feature_columns].values
sub = submission_df[feature_columns].values
pe = np.zeros((sub.shape[0], 3))
pred = np.zeros((z.shape[0], 3))


## === cell 41
NFOLD = 5
BATCH_SIZE=128
kf = KFold(n_splits=NFOLD)


## === cell 42
z = np.asarray(z, dtype=np.float32)
sub = np.asarray(sub, dtype=np.float32)
y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

cnt = 0
for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    model = make_model()
    model.fit(
        z[tr_idx],
        y[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=800,
        validation_data=(z[val_idx], y[val_idx]),
        verbose=0,
    )  #
    print(
        "train", model.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE)
    )
    print(
        "val", model.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE)
    )
    print("predict val...")
    pred[val_idx] = model.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print("predict test...")
    pe += model.predict(sub, batch_size=BATCH_SIZE, verbose=0) / NFOLD


## --- ERROR in cell 42, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3580396891.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     10[0m     [0mprint[0m[0;34m([0m[0;34mf"FOLD {cnt}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m     [0mmodel[0m [0;34m=[0m [0mmake_model[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 12[0;31m     model.fit(
[0m[1;32m     13[0m         [0mz[0m[0;34m[[0m[0mtr_idx[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m         [0my[0m[0;34m[[0m[0mtr_idx[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py[0m in [0;36mconvert_to_tensor[0;34m(x, dtype, sparse)[0m
[1;32m    135[0m             [0mx[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mconvert_to_tensor[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    136[0m             [0;32mreturn[0m [0mtf[0m[0;34m.[0m[0mcast[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 137[0;31m         [0;32mreturn[0m [0mtf[0m[0;34m.[0m[0mconvert_to_tensor[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    138[0m     [0;32melif[0m [0mdtype[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32mand[0m [0;32mnot[0m [0mx[0m[0;34m.[0m[0mdtype[0m [0;34m==[0m [0mdtype[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mtf[0m[0;34m.[0m[0mSparseTensor[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: None values not supported.

## === cell 43
sigma_opt = mean_absolute_error(y, pred[:, 1])
unc = pred[:,2] - pred[:, 0]
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)
