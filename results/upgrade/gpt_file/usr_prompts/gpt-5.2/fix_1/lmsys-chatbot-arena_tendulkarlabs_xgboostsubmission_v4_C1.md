# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.12

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.08084

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
BASE_PATH = '/kaggle/input/lmsys-chatbot-arena'


## === cell 1
class CFG:
    seed = 42  # Random seed
    label2name = {0: 'winner_model_a', 1: 'winner_model_b', 2: 'winner_tie'}
    name2label = {v:k for k, v in label2name.items()}
    class_labels = list(label2name.keys())
    class_names = list(label2name.values())


## === cell 2
import pandas as pd

train_df = pd.read_csv(f'{BASE_PATH}/train.csv')

train_df["prompt"] = train_df.prompt.map(lambda x: eval(x)[0])
train_df["response_a"] = train_df.response_a.map(lambda x: eval(x.replace("null","''"))[0])
train_df["response_b"] = train_df.response_b.map(lambda x: eval(x.replace("null", "''"))[0])

train_df["class_name"] = train_df[["winner_model_a", "winner_model_b" , "winner_tie"]].idxmax(axis=1)
train_df["class_label"] = train_df.class_name.map(CFG.name2label)


## === cell 3
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def count_stop_words(text):
  """Counts the number of stop words in a text string."""
  words = text.lower().split()
  return sum(1 for word in words if word in ENGLISH_STOP_WORDS)

def count_chars(text):
  return len(text)

def is_response_a_longer(row):
  return len(row['response_a']) > len(row['response_b'])

def count_words(text):
  return len(text.split())

def is_response_a_more_words(row):
  return len(row['response_a'].split()) > len(row['response_b'].split())

def count_lines(text):
  return text.count('\n') + 1  # Add 1 to count the last line if it doesn't end with '\n'

def is_markdown(text):
  """Checks if text contains Markdown elements."""
  if re.search(r'(#+|\*\*|\[.*\]\(.*\)|`.*`)', text):
    return True
  return False

def is_html(text):
  """Checks if text contains HTML elements."""
  if re.search(r'(<[^>]+>)', text):
    return True
  return False

def count_bullet_points(text):
  """Counts the number of bullet points in a text string."""
  return text.count('- ') + text.count('* ') + text.count('+ ')

def count_headlines(text):
  """Counts the number of headlines in a text string (Markdown format)."""
  return text.count('# ')  # Assuming headlines are marked with '# '

def count_bold_tokens(text):
  """Counts the number of bold-faced tokens in a text string (Markdown format)."""
  return text.count('**') // 2  # Divide by 2 because each bold token has two **


## === cell 4
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity

text_feature_extraction_pipeline = Pipeline([
    ('hashing_vectorizer', HashingVectorizer(stop_words="english", n_features=50_000)),
    ('tfidf_transformer', TfidfTransformer()),
    ('svd', TruncatedSVD(n_components=50, random_state=CFG.seed)),
    ('normalizer', Normalizer(copy=False))
])

text_feature_extraction_pipeline.fit(
    train_df['prompt'] + train_df['response_a'] + train_df['response_b'])

def calculate_percentage_difference(value_a, value_b):
  """Calculates the percentage difference between two values."""
  return abs(value_a - value_b) / ((value_a + value_b + 1 ) / 2) * 100

def avg_line_length_chars(text):
  lines = text.splitlines()
  return sum(len(line) for line in lines) / len(lines) if lines else 0

def avg_line_length_words(text):
  lines = text.splitlines()
  return sum(len(line.split()) for line in lines) / len(lines) if lines else 0

def avg_line_length_non_stopwords(text):
  lines = text.splitlines()
  non_stopword_counts = [sum(1 for word in line.lower().split() if word not in ENGLISH_STOP_WORDS) for line in lines]
  return sum(non_stopword_counts) / len(lines) if lines else 0


## === cell 5
from sklearn.cluster import KMeans


def cluster_features(df, prompt_vectors, response_a_vectors, response_b_vectors):
    k = 64
    kmeans_a = KMeans(n_clusters=k, random_state=CFG.seed).fit(response_a_vectors)
    kmeans_b = KMeans(n_clusters=k, random_state=CFG.seed).fit(response_b_vectors)
    kmeans_prompts = KMeans(n_clusters=k, random_state=CFG.seed).fit(prompt_vectors)

    df['cluster_prompt'] = kmeans_prompts.labels_
    df['cluster_a'] = kmeans_a.labels_
    df['cluster_b'] = kmeans_b.labels_

    return df, kmeans_a, kmeans_b, kmeans_prompts

def predict_cluster(kmeans_model, x):
    return kmeans_model.predict(x)


## === cell 6
from sklearn.preprocessing import OneHotEncoder

def calculate_fieldwise_features(df, field):
    df[f"{field}_char_length"] = df[field].apply(count_chars)
    df[f"{field}_word_count"] = df[field].apply(count_words)
    df[f"{field}_line_count"] = df[field].apply(count_lines)
    df[f"{field}_stop_word_count"] = df[field].apply(count_stop_words)
    df[f"{field}_is_markdown"] = df[field].apply(is_markdown)
    df[f"{field}_is_html"] = df[field].apply(is_html)
    df[f"{field}_bullet_point_count"] = df[field].apply(count_bullet_points)
    df[f"{field}_headline_count"] = df[field].apply(count_headlines)
    df[f"{field}_bold_token_count"] = df[field].apply(count_bold_tokens)

    df[f"{field}_avg_line_length_chars"] = df[field].apply(avg_line_length_chars)
    df[f"{field}_avg_line_length_words"] = df[field].apply(avg_line_length_words)
    df[f"{field}_avg_line_length_non_stopwords"] = df[field].apply(avg_line_length_non_stopwords)
    return df

def diff_features(df, field1, field2):
    df['diff_length_ab'] = df[f"{field1}_char_length"] - df[f"{field2}_char_length"]
    df['diff_words_ab'] = df[f"{field1}_word_count"] - df[f"{field2}_word_count"]
    df['diff_lines_ab'] = df[f"{field1}_line_count"] - df[f"{field2}_line_count"]
    df['diff_stop_words_ab'] = df[f"{field1}_stop_word_count"] - df[f"{field2}_stop_word_count"]
    df['diff_avg_line_length_chars_ab'] = df[f"{field1}_avg_line_length_chars"] - df[f"{field2}_avg_line_length_chars"]
    df['diff_avg_line_length_words_ab'] = df[f"{field1}_avg_line_length_words"] - df[f"{field2}_avg_line_length_words"]
    df['diff_avg_line_length_non_stopwords_ab'] = df[f"{field1}_avg_line_length_non_stopwords"] - df[f"{field2}_avg_line_length_non_stopwords"] 

    df['both_markdown_ab'] = df[f"{field1}_is_markdown"] & df[f"{field2}_is_markdown"]
    df['both_html_ab'] =  df[f"{field1}_is_html"] & df[f"{field2}_is_html"]
    df['diff_bullet_points_ab'] = df[f"{field1}_bullet_point_count"] - df[f"{field2}_bullet_point_count"]
    df['diff_headlines_ab'] = df[f"{field1}_headline_count"] - df[f"{field2}_headline_count"]
    df['diff_bold_tokens_ab'] = df[f"{field1}_bold_token_count"] - df[f"{field2}_bold_token_count"]

    df['diff_percentage_length_ab'] = df.apply(lambda row: calculate_percentage_difference(row[f"{field1}_char_length"], row[f"{field2}_char_length"]), axis=1)
    df['diff_percentage_words_ab'] = df.apply(lambda row: calculate_percentage_difference(row[f"{field1}_word_count"], row[f"{field2}_word_count"]), axis=1)
    df['diff_percentage_lines_ab'] = df.apply(lambda row: calculate_percentage_difference(row[f"{field1}_line_count"], row[f"{field2}_line_count"]), axis=1)
    df['diff_percentage_stop_words_ab'] = df.apply(lambda row: calculate_percentage_difference(row[f"{field1}_stop_word_count"], row[f"{field2}_stop_word_count"]), axis=1) 
    df['diff_percentage_bullet_points_ab'] = df.apply(lambda row: calculate_percentage_difference(row[f"{field1}_bullet_point_count"], row[f"{field2}_bullet_point_count"]), axis=1)  
    df['diff_percentage_headlines_ab'] = df.apply(lambda row: calculate_percentage_difference(row[f"{field1}_headline_count"], row[f"{field2}_headline_count"]), axis=1)  
    df['diff_percentage_bold_tokens_ab'] = df.apply(lambda row: calculate_percentage_difference(row[f"{field1}_bold_token_count"], row[f"{field2}_bold_token_count"]), axis=1)  

    df['diff_avg_line_length_chars_ab'] = df[f"{field1}_avg_line_length_chars"] - df[f"{field2}_avg_line_length_chars"] 
    df['diff_avg_line_length_words_ab'] = df[f"{field1}_avg_line_length_words"] - df[f"{field2}_avg_line_length_words"] 
    df['diff_avg_line_length_non_stopwords_ab'] = df[f"{field1}_avg_line_length_non_stopwords"] - df[f"{field2}_avg_line_length_non_stopwords"] 
    return df

def ratio_features(df):
    df['ratio_words_a'] = df['response_a_word_count']/df['prompt_word_count']
    df['ratio_words_b'] = df['response_b_word_count']/df['prompt_word_count']

    df['ratio_lines_a'] = df['response_a_line_count']/df['prompt_line_count']
    df['ratio_lines_b'] = df['response_b_line_count']/df['prompt_line_count']

    df['ratio_len_a'] = df['response_a_char_length']/df['prompt_char_length']
    df['ratio_len_b'] = df['response_a_char_length']/df['prompt_char_length']
    return df

def generate_features(
    df, kmeans_prompt=None, kmeans_response_a=None, kmeans_response_b=None,
    train_flag=True):

    """Generates features for each row in the input DataFrame."""
    df = calculate_fieldwise_features(df, 'prompt')
    df = calculate_fieldwise_features(df, 'response_a')
    df = calculate_fieldwise_features(df, 'response_b')

    df = diff_features(df, 'response_a', 'response_b')
    df = ratio_features(df)

    prompt_vectors = compute_text_vectors(df, 'prompt')
    response_a_vectors = compute_text_vectors(df, 'response_a')
    response_b_vectors = compute_text_vectors(df, 'response_b')

    df = add_text_vectors_to_df(df, prompt_vectors, 'prompt')
    df = add_text_vectors_to_df(df, response_a_vectors, 'response_a')
    df = add_text_vectors_to_df(df, response_b_vectors, 'response_b')

    df = text_similarity(df, prompt_vectors, response_a_vectors, 'sim_a')
    df = text_similarity(df, prompt_vectors, response_b_vectors, 'sim_b')
    df = text_similarity(df, response_a_vectors, response_b_vectors, 'sim_ab')

    if train_flag:
      df, kmeans_prompt = cluster_feature(df, prompt_vectors, 'cluster_prompt')
      df, kmeans_response_a = cluster_feature(df, response_a_vectors, 'cluster_a')
      df, kmeans_response_b = cluster_feature(df, response_b_vectors, 'cluster_b')
    else:
      df = predict_cluster(df, kmeans_prompt, prompt_vectors, 'cluster_prompt')
      df = predict_cluster(df, kmeans_response_a, response_a_vectors, 'cluster_a')
      df = predict_cluster(df, kmeans_response_b, response_b_vectors, 'cluster_b')

    df['cluster_prompt_cat'] = df['cluster_prompt'].astype('category').cat.as_ordered()
    df['cluster_prompt_cat'] = df['cluster_prompt_cat'].cat.codes + 1

    df['cluster_a_cat'] = df['cluster_a'].astype('category').cat.as_ordered()
    df['cluster_a_cat'] = df['cluster_a_cat'].cat.codes + 1

    df['cluster_b_cat'] = df['cluster_b'].astype('category').cat.as_ordered()
    df['cluster_b_cat'] = df['cluster_b_cat'].cat.codes + 1



    return df, kmeans_prompt, kmeans_response_a, kmeans_response_b

def cluster_feature(df, text_vectors, output_field, k=64):
    kmeans = KMeans(n_clusters=k, random_state=CFG.seed).fit(text_vectors)
    df[output_field] = kmeans.labels_
    return df, kmeans

def predict_cluster(df, kmeans_model, x, output_field):
    df[output_field] = kmeans_model.predict(x)
    return df

def compute_text_vectors(df, field):
    return text_feature_extraction_pipeline.transform(df[field])

def text_similarity(df, text1, text2, output_field):
    df[output_field] = [cosine_similarity([text1[i]], [text2[i]])[0][0] for i in range(len(df))]
    return df

def add_text_vectors_to_df(df, text_vectors, prefix):
    """Adds pre-computed text vectors to the DataFrame."""
    vector_df = pd.DataFrame(text_vectors,
                             columns=[f"{prefix}_vector_{i}" for i in range(text_vectors.shape[1])])
    df = pd.concat([df, vector_df], axis=1)
    return df


## === cell 7
train_df, kmeans_prompt, kmeans_response_a, kmeans_response_b = generate_features(train_df)


## === cell 8
label = ['class_label']

def get_feature_list():
    exclude_cols = ['id', 'prompt', 'response_a', 'response_b', 'model_a', 'model_b',
                'winner_model_a', 'winner_model_b', 'winner_tie', 'class_name',
                'class_label', 'cluster_prompt', 'cluster_a', 'cluster_b',
                'model_a_cat', 'model_b_cat',
                ]
    return [col for col in train_df.columns if col not in exclude_cols]
    


## === cell 9
feature_list = get_feature_list()
training_features = train_df[feature_list]
training_labels = train_df[label]


## === cell 10
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    training_features, training_labels, test_size=0.2, random_state=CFG.seed
)


## === cell 11
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200],
    'max_depth': [3],
    'learning_rate': [0.2]
}

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=CFG.seed)


grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

best_xgb_model = grid_search.best_estimator_


## === cell 12
from xgboost import plot_importance
from matplotlib import pyplot

ax = plot_importance(best_xgb_model)
ax.figure.set_size_inches(20,30)
pyplot.show()


## === cell 13
from sklearn.metrics import classification_report, log_loss

y_pred = best_xgb_model.predict(X_val)

print(classification_report(y_val, y_pred, target_names=CFG.class_names))

y_pred_proba = best_xgb_model.predict_proba(X_val)
nll = log_loss(y_val, y_pred_proba)
print("Negative Log Likelihood:", nll)


## === cell 14
test_df = pd.read_csv(f'{BASE_PATH}/test.csv')
test_df, _, _, _ = generate_features(test_df, kmeans_prompt, kmeans_response_a, kmeans_response_b, train_flag=False)


## === cell 15
feature_list = get_feature_list()
test_features = test_df[feature_list]

test_predictions = (best_xgb_model.predict_proba(test_features) + [0.33, 0.33, 0.33])/2
sub_df = test_df[["id"]].copy()
sub_df[CFG.class_names] = test_predictions
sub_df.to_csv("/kaggle/working/submission.csv", index=False)
sub_df.head()
