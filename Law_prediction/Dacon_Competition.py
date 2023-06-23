import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler

# Prepare data
train = pd.read_csv('open/train.csv')
test = pd.read_csv('open/test.csv')

# Configure vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))  # use both unigrams and bigrams

# Configure classifier
clf = LogisticRegression(solver='liblinear', penalty='l1')  # L1 regularization

# Create pipeline
pipe = make_pipeline(vectorizer, clf)

# Set random search params
param_distributions = {
    'logisticregression__C': loguniform(1e-4, 1e4),  # log-uniform distribution from 1e-4 to 1e4
    'tfidfvectorizer__max_df': [0.85, 0.9, 0.95, 1.0]  # maximum document frequency for the CountVectorizer
}

# Create random search
random_search = RandomizedSearchCV(pipe, param_distributions, n_iter=20, cv=5, random_state=42, n_jobs=-1)

# Prepare train data
X_train = train[['first_party', 'second_party', 'facts']].apply(lambda x: ' '.join(x), axis=1)
Y_train = train["first_party_winner"]

# Perform undersampling
rus = RandomUnderSampler(random_state=42)
X_train_res, Y_train_res = rus.fit_resample(X_train.values.reshape(-1, 1), Y_train)

# Train the model
random_search.fit(X_train_res.ravel(), Y_train_res)

# Split train data into train and validation sets
X_train_full = train[['first_party', 'second_party', 'facts']].apply(lambda x: ' '.join(x), axis=1)
Y_train_full = train["first_party_winner"]
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.33, random_state=42)

# Train the model
random_search.fit(X_train, Y_train)

# Make predictions on the validation set
val_pred = random_search.predict(X_val)

# Calculate and print macro-f1 score
macro_f1 = f1_score(Y_val, val_pred, average='macro')
print('Macro F1 Score:', macro_f1)

# Make predictions
X_test = test[['first_party', 'second_party', 'facts']].apply(lambda x: ' '.join(x), axis=1)
pred = random_search.predict(X_test)

# Save predictions
submit = pd.read_csv('open/sample_submission.csv')
submit['first_party_winner'] = pred
submit.to_csv('./submit.csv', index=False)
print('Done')