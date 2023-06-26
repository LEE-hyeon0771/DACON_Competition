import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from transformers import BertModel, BertTokenizer

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# Prepare data
train = pd.read_csv('open/train.csv')
test = pd.read_csv('open/test.csv')

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Encode data
X_train_facts = bert_encode(train.facts.values, tokenizer, max_len=160)
X_test_facts = bert_encode(test.facts.values, tokenizer, max_len=160)
X_train_party1 = bert_encode(train.first_party.values, tokenizer, max_len=160)
X_test_party1 = bert_encode(test.first_party.values, tokenizer, max_len=160)
X_train_party2 = bert_encode(train.second_party.values, tokenizer, max_len=160)
X_test_party2 = bert_encode(test.second_party.values, tokenizer, max_len=160)

# Prepare labels
Y_train = train["first_party_winner"]

# Concatenate encoded data
X_train = np.concatenate([X_train_party1[0], X_train_party2[0], X_train_facts[0]], axis=1)
X_test = np.concatenate([X_test_party1[0], X_test_party2[0], X_test_facts[0]], axis=1)

# Split the data into train and validation sets in a 2:1 ratio
X_train_full, X_val_full, Y_train_full, Y_val_full = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)

# Parameter grid for LGBM
param_grid = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'num_leaves': [50, 100, 200],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'min_split_gain': [0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9],
    'subsample_freq': [20]
}

# Train the model on the training set
model = LGBMClassifier()

rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3)
rsearch.fit(X_train_full, Y_train_full)

# Print out the best parameters
print('Best parameters:', rsearch.best_params_)

# Make predictions on the validation set
val_pred = rsearch.predict(X_val_full)

# Calculate and print macro-f1 score
macro_f1 = f1_score(Y_val_full, val_pred, average='macro')
print('Macro F1 Score:', macro_f1)

# Make predictions on the test set
pred = rsearch.predict(X_test)

# Prepare submission
submit = pd.read_csv('open/sample_submission.csv')
submit['first_party_winner'] = pred
submit.to_csv('./submit1.csv', index=False)
print('Done')