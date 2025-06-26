# Part 2:Feature Extraction and Classification
"""
Read the hansard40000.csv dataset in the texts directory into a dataframe. 
Sub-set and rename the dataframe as follows:
"""
import pandas as pd
import spacy 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("/Users/elenadelafuente/Desktop/MASTER/2 trimestre/Natural Lenguage Processing/Assesment/p2-texts/hansard40000.csv")

# Rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’
df = df.replace({"Labour (Co-op)": "Labour"})

# Remove the ‘Speaker’ value
df = df[df["party"]!= "Speaker"]

# Remove any rows where the value of the ‘party’ column is not one of the 4 most common 
parties_4 = list(df['party'].value_counts()[:4].index)
df = df[df['party'].isin(parties_4)]

# Remove any rows where the value in the ‘speech_class’ column is not ‘Speech’.
df = df[df['speech_class'] == 'Speech']

# Remove any rows where the text in the ‘speech’ column is less than 1000 characters long.
final_df = df[df["speech"].str.len() >= 1000]

# Print the dimensions of the resulting dataframe using the shape method.
print('The shape of the final dataframes is: ', final_df.shape)

"""
Vectorise the speeches using TfidfVectorizer from scikit-learn. Use the default
parameters, except for omitting English stopwords and setting max_features to
3000.
Split the data into a train and test set, using stratified sampling, with a
random seed of 26.
"""

vectorizer = TfidfVectorizer(stop_words = "english", max_features = 3000)

X = vectorizer.fit_transform(final_df["speech"])
y = final_df["party"] # Since we are trying to predict the political party that said certain speech

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=26, stratify=y)

"""
Train RandomForest (with n_estimators=300) and SVM with linear kernel 
classifiers on the training set, and print the scikit-learn macro-average f1 score and
classification report for each classifier on the test set. The label that you are
trying to predict is the ‘party’ value.
"""
# Train RandomForest (with n_estimators=300)
r_Forest = RandomForestClassifier(n_estimators=300, class_weight='balanced')
r_Forest.fit(X_train, y_train)
y_rF_predict = r_Forest.predict(X_test)

# Train the SVM with linear kernel classifiers
svm = SVC(kernel= 'linear')
svm.fit(X_train, y_train)
y_svm_predict = svm.predict(X_test)

# Print the scikit-learn macro-average f1 score
print(f" The F1 score for the random forest is:{f1_score(y_test,y_rF_predict, average = 'macro')}")
print(f" The F1 score for the svm is:{f1_score(y_test,y_svm_predict, average = 'macro')}")

# Classification report
print("Classification report Random Forest")
class_rf_1 = classification_report(y_test,y_rF_predict)
print(class_rf_1)  

print("Classification report svm")
class_svm_1 = classification_report(y_test,y_svm_predict)
print(class_svm_1)

"""
Adjust the parameters of the Tfidfvectorizer so that unigrams, 
bi-grams and tri-grams will be considered as features, 
limiting the total number of features to 3000. 
Print the classification report as in 2(c) again using these parameters.
"""
# Adjust the parameters of the Tfidfvectorizer so that unigrams, bi-grams and tri-grams will be considered as features
vectorizer = TfidfVectorizer(stop_words = "english", max_features = 3000, ngram_range=(1,3))

X = vectorizer.fit_transform(final_df["speech"])
y = final_df["party"]

# Print the classification report as in 2(c) again using these parameters.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=26, stratify=y)
# Train RandomForest (with n_estimators=300)
r_Forest = RandomForestClassifier(n_estimators=300, class_weight='balanced')
r_Forest.fit(X_train, y_train)
y_rF_predict = r_Forest.predict(X_test)

# Train the SVM with linear kernel classifiers
svm = SVC(kernel= 'linear')
svm.fit(X_train, y_train)
y_svm_predict = svm.predict(X_test)

# Print the classification
print("Classification report Random Forest(2)")
# Aqui hay un error
class_rf_2 = classification_report(y_test,y_rF_predict)
print(class_rf_2)  

print("Classification report svm(2)")
class_svm_2 = classification_report(y_test,y_svm_predict)
print(class_svm_2)

"""
Implement a new custom tokenizer and pass it to the tokenizer argument of
Tfidfvectorizer. 
"""

def custom_tokenizer(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        # We skip the stopword, punctuation, or numbers
        if token.is_stop:
            continue
        if token.is_punct:
            continue
        if token.like_num:
            continue

        # We make a lemmatization 
        new_token = token.lemma_
        tokens.append(new_token.lower().strip())
    return tokens

"""
You can use this function in any way you like to try to achieve
the best classification performance while keeping the number of features to no
more than 3000, and using the same three classifiers as above. 
"""
vectorizer =  TfidfVectorizer(tokenizer = custom_tokenizer, max_features = 3000)
X_custom = vectorizer.fit_transform(final_df["speech"])
y_custom = final_df["party"]

X_train_cust, X_test_cust, y_train_cust, y_test_cust = train_test_split(X_custom, y_custom, stratify=y_custom)

# Train the random Forest
r_Forest_cust = RandomForestClassifier(n_estimators=300, class_weight='balanced')
r_Forest_cust.fit(X_train_cust, y_train_cust)
y_rF_predict_cust = r_Forest_cust.predict(X_test_cust)

# Train svm 
svm_cust = SVC(kernel= 'linear')
svm_cust.fit(X_train, y_train)
y_svm_predict_cust = svm_cust.predict(X_test)

# F1 scores
rf_f1 = f1_score(y_test_cust, y_rF_predict_cust, average='macro')
print(f"F1 of the random forest trained with the custom tokenizer is {rf_f1}")

svm_f1 = f1_score(y_test_cust, y_svm_predict_cust, average='macro')
print(f"F1 of the svm trained with the custom tokenizer is {svm_f1}")

# Classifications
# Print the classification report for the best performing classifier using your tokenizer.
if rf_f1 > svm_f1:
    print("Classification report Random Forest custom tokenizer")
    class_rf_3 = classification_report(y_test,y_rF_predict_cust)
    print(class_rf_3)  
else:
    print("Classification report svm custom tokenizer")
    class_svm_3 = classification_report(y_test,y_svm_predict_cust)
    print(class_svm_3)

