import itertools

import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy
from sklearn.naive_bayes import MultinomialNB

dataset = 'fake_or_real_news.csv'


def loadPanda(dataset):
    loaded_panda = pd.read_csv(dataset)
    print(loaded_panda.shape)
    loaded_panda = loaded_panda.set_index("Unnamed: 0")
    print(loaded_panda.head())
    loaded_panda.title = loaded_panda.title.str.lower()
    loaded_panda.text = loaded_panda.text.str.lower()
    print(loaded_panda.head())
    # remove the URL's present
    loaded_panda.title = loaded_panda.title.str.replace(r'http[\w:/\.]+', '<URL>')
    loaded_panda.text = loaded_panda.text.str.replace(r'http[\w:/\.]+', '<URL>')

    # remove everything except for the characters and the punctuation
    loaded_panda.title = loaded_panda.title.str.replace(r'[^\.\w\s]', '')
    loaded_panda.text = loaded_panda.text.str.replace(r'[^\.\w\s]', '')

    # replacing multiple . with one .
    loaded_panda.title = loaded_panda.title.str.replace(r'[^\.\w\s]', '')
    loaded_panda.text = loaded_panda.text.str.replace(r'[^\.\w\s]', '')

    # adds spaces before and after each .
    loaded_panda.title = loaded_panda.title.str.replace(r'\.', ' . ')
    loaded_panda.text = loaded_panda.text.str.replace(r'\.', ' . ')

    # replaces multiple spaces with single spaces
    loaded_panda.title = loaded_panda.title.str.replace(r'\s\s+', ' ')
    loaded_panda.text = loaded_panda.text.str.replace(r'\s\s+', ' ')

    loaded_panda.title = loaded_panda.title.str.strip()
    loaded_panda.text = loaded_panda.text.str.strip()
    print(loaded_panda.shape)
    print(loaded_panda.head())
    return loaded_panda


fakeRealPanda = loadPanda(dataset)

fakeRealPanda.head()

# Set `y`
y = fakeRealPanda.label

# Drop the `label` column
fakeRealPanda.drop("label", axis=1)


def count_fe(train, test):
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    print(count_vectorizer.get_feature_names()[:10])


# Make training and test sets
X_train, X_test, y_train, y_test = train_test_split(fakeRealPanda['text'], y, test_size=0.33, random_state=53)

# Initialize the `count_vectorizer`
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test set
count_test = count_vectorizer.transform(X_test)

# Initialize the `tfidf_vectorizer`
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test set
tfidf_test = tfidf_vectorizer.transform(X_test)

# Get the feature names of `tfidf_vectorizer`
print(tfidf_vectorizer.get_feature_names()[-10:])

# Get the feature names of `count_vectorizer`
print(count_vectorizer.get_feature_names()[:10])

count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

difference = set(count_df.columns) - set(tfidf_df.columns)
difference

print(count_df.equals(tfidf_df))

count_df.head()
tfidf_df.head()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


clf = MultinomialNB()

clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
cmfass = plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
tfi_f1 = metrics.f1_score(y_test, pred, average='macro')
print('f1 score: ', tfi_f1)
tfi_acc = metrics.accuracy_score(y_test, pred)
print('accuracy: ', tfi_acc)
tfi_prec = metrics.precision_score(y_test, pred, average='micro')
print('precision: ', tfi_prec)

print(metrics.classification_report(y_test,pred, labels=['FAKE', 'REAL']))

clf = MultinomialNB()

clf.fit(count_train, y_train)
pred = clf.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

sxx = fakeRealPanda["title"]
print(sxx)
