import nltk

## This code downloads the required packages.
## You can run `nltk.download('all')` to download everything.
nltk_packages = [
    ("reuters", "corpora/reuters.zip")
]

for pid, fid in nltk_packages:
    try:
        nltk.data.find(fid)
    except LookupError:
        nltk.download(pid)

# Setting up corpus
print("Setting up corpus 'routers' .. ", end='')
from nltk.corpus import reuters
print("Done!")

# Setting up train/test data
print("Setting up train/test data .. ", end='')
train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])
test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])
print("Done!")

# Setting up categories
print("Setting up text categories .. ", end='')
all_categories = sorted(list(set(reuters.categories())))
print("Done!")
print("The following categories are available:")
print(all_categories)

# Check the length of the training set
# First let's check how much training data is available.
# This will help up to chose the appropriate classifier
print("Cheking the size of the training set .. ", end='')
trainingset_length = len(train_documents)
print("Done!")
print("Training set contains " + str(trainingset_length) + " instances")
# Since there are 7769 documents in the training set, we can go with Linear Support Vector Classification
# This model scored the best, compared with MultinomialNB, and Logistics Regression

# Preparing the model
print("Preparing the model .. ", end='')

# At first we will tokenize the words
train_tokens = [nltk.word_tokenize(text) for text in train_documents]
test_tokens = [nltk.word_tokenize(text) for text in test_documents]

# Intuitivelly, the meaning of the text hides in nouns and (possibly) punctuation.
# Other words may add noise to our data
# Therefore, let's tag the word with POS labels
def getTaggedList(tokens_list):
    return [nltk.pos_tag(x) for x in tokens_list]

pos_tagged = getTaggedList(train_tokens)

# And now remove everything but Nouns and End of the sentense punctuation signs
def removeExtraPOS(data):
    result = []
    for i in range(len(data)):
        document = []
        for j in range(len(data[i])):
            if (data[i][j][1].startswith('N') or data[i][j][1]=='.'):
                document.append(data[i][j])
        result.append(document)
    return result

clean_data = removeExtraPOS(pos_tagged)

# To improve performance, let's upweight some words.
# The first sentence of the document can often tell us a lot about the topic. Therefore
def upweightWords(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i].append(data[i][j])
            if data[i][j][1]=='.':
                break
                
upweightWords(clean_data)

# Also, for noise reduction, let's collapse case of words
# We will keep the case for all uppercase words, because they may be important abbreviation
def removeCaps(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if not data[i][j][0].isupper():
                data[i][j][0].lower()
                
removeCaps(clean_data)

# We are concerned with frequences in the context, instead of the document
# Therefore, we will use TF-IDF to calculate

def prepTrainingSet(data):
    training_set = []
    for i in range(len(data)):
        document = []
        for j in range(len(data[i])):
            document.append(str(data[i][j][0]))
        training_set.append(document)
    return training_set

# Pandas DataFrame Operations to prepare text and abels for train/test input

import pandas as pd

final_train_data = [' '.join(words) for words in prepTrainingSet(clean_data)]
final_train_categories = [''.join(words) for words in train_categories]

training_df_texts = pd.DataFrame({'data':final_train_data})
training_df_labels = pd.DataFrame({'categories':final_train_categories})

X_train = training_df_texts['data']
y_train = training_df_labels['categories']

final_test_categories = [''.join(words) for words in test_categories]

test_df_texts = pd.DataFrame({'data':test_documents})
test_df_categories = pd.DataFrame({'categories':final_test_categories})

X_test = test_df_texts['data']
y_test = test_df_categories['categories']

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Creating a pipeline
# Using a TF-IDF Vectorizer to count frequencies in the whole context
# After that, select the 10,000 best features

pipeline = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True, lowercase=False)),
    ('chi', SelectKBest(chi2, k=10000)),
    ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))
])

print("Done!")

# Training the model
print("Training the model .. ", end='')
model = pipeline.fit(X_train, y_train)
print("Done!")

# Classification

# Input text
input_text = [input("Enter text to classify: ")]

# Input preparation
pred_tokens = [nltk.word_tokenize(text) for text in input_text]
pred_pos_tagged = getTaggedList(pred_tokens)
pred_clean_data = removeExtraPOS(pred_pos_tagged)
upweightWords(pred_clean_data)
removeCaps(pred_clean_data)
pred_train_data = [' '.join(words) for words in prepTrainingSet(pred_clean_data)]

# Prediction
print(pipeline.predict(pred_train_data))