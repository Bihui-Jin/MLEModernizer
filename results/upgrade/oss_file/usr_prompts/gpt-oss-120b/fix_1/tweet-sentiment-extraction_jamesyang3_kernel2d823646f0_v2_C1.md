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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
nltk==3.9.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.64524

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import sys
import string
import re
from nltk.stem import WordNetLemmatizer 

inputs = 0

def jaccard(A, B): 
    a = set(A.lower().split()) 
    b = set(B.lower().split())
    c = a.intersection(b)

    j_value = float(len(c)) / (len(a) + len(b) - len(c))
    return j_value

def total_number_of_sentiments():
	df_reviews = pd.read_csv("train.csv")
	sentiments = df_reviews.groupby('sentiment')['textID'].nunique()
	
	total_pos = sentiments[0]
	total_neutral = sentiments[1]
	total_neg = sentiments[2]
	return total_pos, total_neutral, total_neg


def load_and_train(train_size=None, random_state=None, max_feat=None, max_d=None, min_d=None):
	if train_size is None:
		train_size = 0.90
	if random_state is None:
		random_state = 0
	if max_feat is None:
		max_feat = 10000
	if max_d is None:
		max_d= 0.95
	if min_d is None:
		min_d = 2
	train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
	test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
	sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
	train[train['text'].isna()]
	train = train.dropna()
	train[train['text'].isna()]




	train['text'] = train['text'].map(lambda x: re.sub('\\n',' ',str(x)))
    
	train['text'] = train['text'].map(lambda x: re.sub("\[\[User.*",'',str(x)))
	    
	train['text'] = train['text'].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
	    
	train['text'] = train['text'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))


	train['text'] = train['text'].str.lower()
	test['text'] = test['text'].str.lower()
	
	X_train, X_val = train_test_split(train, train_size = train_size, random_state = random_state)

	pos_train = X_train[X_train['sentiment'] == 'positive']
	neutral_train = X_train[X_train['sentiment'] == 'neutral']
	neg_train = X_train[X_train['sentiment'] == 'negative']

	cv = CountVectorizer(max_df=max_d, min_df=min_d, max_features=max_feat, stop_words='english')

	X_train_cv = cv.fit_transform(X_train['text'].values.astype('U'))

	X_pos = cv.transform(pos_train['text'])
	X_neutral = cv.transform(neutral_train['text'])
	X_neg = cv.transform(neg_train['text'])

	pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
	neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
	neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())

	pos_words = {}
	neutral_words = {}
	neg_words = {}

	for k in cv.get_feature_names():
	    pos = pos_count_df[k].sum()
	    neutral = neutral_count_df[k].sum()
	    neg = neg_count_df[k].sum()
	    
	    pos_words[k] = pos/pos_train.shape[0]
	    neutral_words[k] = neutral/neutral_train.shape[0]
	    neg_words[k] = neg/neg_train.shape[0]
	    

	neg_words_adj = {}
	pos_words_adj = {}
	neutral_words_adj = {}

	for key, value in neg_words.items():
	    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])

	for key, value in pos_words.items():
	    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
	    
	for key, value in neutral_words.items():
	    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])

	return X_val, pos_words_adj, neg_words_adj, neutral_words_adj, train, test, sample
def calculate_selected_text(df_row, pos_words_adj, neg_words_adj, neutral_words_adj, tol):
    
    tweet = df_row['text']
    sentiment = df_row['sentiment']
    
    if(sentiment == 'neutral'):
        return tweet
    
    elif(sentiment == 'positive'):
        dict_to_use = pos_words_adj # Calculate word weights using the pos_words dictionary
    elif(sentiment == 'negative'):
        dict_to_use = neg_words_adj # Calculate word weights using the neg_words dictionary
        
    words = tweet.split()
    words_len = len(words)
    subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]
    
    score = 0
    selection_str = '' # This will be our choice
    lst = sorted(subsets, key = len) # Sort candidates by length
    
    for i in range(len(subsets)):
        
        new_sum = 0 # Sum for the current substring
        
        for p in range(len(lst[i])):
            if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):
            	new_sum += dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]
            	after_character = 0
            	before_character = 0
            if (inputs == 1):  # If there are inputs.

                if (p + 1 < len(lst[i]) and p - 1 > 0):


                    after_character = lst[i][p + 1].translate(
                        str.maketrans('', ''.string.punctuation)) in dict_to_use.keys()
                    before_character = lst[i][p - 1].translate(
                        str.maketrans('', ''.string.punctuation)) in dict_to_use.keys()

                    if (abs(after_character) > abs(before_character)):  # If the weight of the word is more
                        new_sum += dict_to_use[lst[i][p].translate(
                            str.maketrans('', '', string.punctuation))] + after_character

                    elif (abs(after_characvter) < abs(before_character)):  # If the weight of the other word is more
                        new_sum += dict_to_use[lst[i][p].translate(
                            str.maketrans('', '', string.punctuation))] + before_character

                    else:  # if the weight is 0 (edge words in the tweet) #otherwise its on the edge
                        new_sum += dict_to_use[lst[i][p].translate(str.maketrans('', '', string.punctuation))]
        if(new_sum > score + tol):
            score = new_sum
            selection_str = lst[i]
            tol = tol*5 # Increase the tolerance a bit each time we choose a selection

    if(len(selection_str) == 0):
        selection_str = words
        
    return ' '.join(selection_str)


def calculation(train_size=None, random_state=None, max_features=None, max_df=None, min_df=None):
	if train_size is None:
		train_size = 0.90
	if random_state is None:
		random_state = 0
	if max_features is None:
		max_feat = 10000
	if max_df is None:
		max_df= 0.95
	if min_df is None:
		min_df = 2
	X_val, pos_words_adj, neg_words_adj, neutral_words_adj = load_and_train(train_size, random_state, max_features, max_df, min_df)

	pd.options.mode.chained_assignment = None
	tol = 0.001

	X_val['predicted_selection'] = ''

	for index, row in X_val.iterrows():
	    
	    selected_text = calculate_selected_text(row, pos_words_adj, neg_words_adj, neutral_words_adj, 3)
	    
	    X_val.loc[X_val['textID'] == row['textID'], ['predicted_selection']] = selected_text

	X_val['jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)
	print("For train_size: ", train_size)
	print('The jaccard score for the validation set is:', np.mean(X_val['jaccard']))
	return np.mean(X_val['jaccard'])

def run_output(train, test, sample):
	pos_tr = train[train['sentiment'] == 'positive']
	neutral_tr = train[train['sentiment'] == 'neutral']
	neg_tr = train[train['sentiment'] == 'negative']
	cv = CountVectorizer(max_df=0.95, min_df=2,
                                     max_features=10000,
                                     stop_words='english')
	final_cv = cv.fit_transform(train['text'])

	X_pos = cv.transform(pos_tr['text'])
	X_neutral = cv.transform(neutral_tr['text'])
	X_neg = cv.transform(neg_tr['text'])

	pos_final_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
	neutral_final_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
	neg_final_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())

	pos_words = {}
	neutral_words = {}
	neg_words = {}

	for k in cv.get_feature_names():
	    pos = pos_final_count_df[k].sum()
	    neutral = neutral_final_count_df[k].sum()
	    neg = neg_final_count_df[k].sum()
	    
	    pos_words[k] = pos/(pos_tr.shape[0])
	    neutral_words[k] = neutral/(neutral_tr.shape[0])
	    neg_words[k] = neg/(neg_tr.shape[0])

	neg_words_adj = {}
	pos_words_adj = {}
	neutral_words_adj = {}

	for key, value in neg_words.items():
	    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])
	    
	for key, value in pos_words.items():
	    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
	    
	for key, value in neutral_words.items():
	    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])

	tol = 0.001

	for index, row in test.iterrows():
	    
	    selected_text = calculate_selected_text(row, pos_words_adj, neg_words_adj, neutral_words_adj, tol)
	    
	    sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text


	sample.to_csv('submission.csv', index = False)

def run_multiple_times():
	best = 0
	X_val, pos_words_adj, neg_words_adj, neutral_words_adj, train, test, sample = load_and_train()
	run_output(train, test, sample)
run_multiple_times()


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3423754553.py in <cell line: 0>()
    329         X_val, pos_words_adj, neg_words_adj, neutral_words_adj, train, test, sample = load_and_train()
    330         run_output(train, test, sample)
--> 331 run_multiple_times()
    332 # load_and_split_data()

/tmp/ipykernel_11/3423754553.py in run_multiple_times()
    327         # print("best max_feature size is: ", best)
    328         # calculation()
--> 329         X_val, pos_words_adj, neg_words_adj, neutral_words_adj, train, test, sample = load_and_train()
    330         run_output(train, test, sample)
    331 run_multiple_times()

/tmp/ipykernel_11/3423754553.py in load_and_train(train_size, random_state, max_feat, max_d, min_d)
    117         X_neg = cv.transform(neg_train['text'])
    118 
--> 119         pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
    120         neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
    121         neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())

AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names'
