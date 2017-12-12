### Machine Learning Techniques: NB, SVM and DT

#### 1. Naive Bayes

```python
!pip install scikit-learn
!pip install nltk
```

```python
#!/usr/bin/python
#start up codes for downloading data files
print
print "checking for nltk"
try:
    import nltk
except ImportError:
    print "you should install nltk before continuing"

print "checking for numpy"
try:
    import numpy
except ImportError:
    print "you should install numpy before continuing"

print "checking for scipy"
try:
    import scipy
except:
    print "you should install scipy before continuing"

print "checking for sklearn"
try:
    import sklearn
except:
    print "you should install sklearn before continuing"

print
print "downloading the Enron dataset (this may take a while)"
print "to check on progress, you can cd up one level, then execute <ls -lthr>"
print "Enron dataset should be last item on the list, along with its current size"
print "download will complete at about 423 MB"
import urllib
url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"
urllib.urlretrieve(url, filename="../enron_mail_20150507.tar.gz")
print "download complete!"


print
print "unzipping Enron dataset (this may take a while)"
import tarfile
import os
os.chdir("..")
tfile = tarfile.open("enron_mail_20150507.tar.gz", "r:gz")
tfile.extractall(".")

print "you're ready to go!"
```

```python
#!/usr/bin/python
"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.
    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")  #download the folder "tools" and it is in the upper folder "DAND7"
from email_preprocess import preprocess
```


```python
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

```
    no. of Chris training emails: 7936
    no. of Sara training emails: 7884

```python
len(features_train),len(labels_train),len(features_test),len(labels_test)
```
    (15820, 15820, 1758, 1758)

```python
#########################################################
### your code goes here ###
t0 = time()
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf = GaussianNB()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
```

    training time: 0.833 s
    predicting time: 0.955 s

```python
clf.score(features_test,labels_test)
```
    0.97326507394766781

#### 2. Support Vector Machine
```python
#!/usr/bin/python
"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
```
    no. of Chris training emails: 7936
    no. of Sara training emails: 7884    
```python
len(features_train), len(features_test), len(labels_train), len(labels_test)
```
    (15820, 1758, 15820, 1758)

```python
from sklearn import svm
t0 = time()
clf = svm.SVC(kernel = 'linear') # Create classifier, using a linear kernel to avoid too long time for fitting
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
clf.score(features_test, labels_test)
```
    training time: 155.521 s
    predicting time: 171.834 s

    0.98407281001137659

```python
#To speed up an algorithm, we can train it on a smaller training dataset.
#The tradeoff is that the accuracy almost goes down when doing this.
#It is important for real application to have more timely training and prediction, for example, credit fraud, voice control.
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]
t0 = time()
clf = svm.SVC(kernel = 'linear') # Create classifier, using a linear kernel to avoid too long time for fitting
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
clf.score(features_test, labels_test)
```

    training time: 0.101 s
    predicting time: 1.032 s

    0.88452787258248011

```python
#rbf: radial basis functions
t0 = time()
clf = svm.SVC(kernel = 'rbf') # Create classifier, using a linear kernel to avoid too long time for fitting
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
clf.score(features_test, labels_test)
```
    training time: 0.102 s
    predicting time: 1.128 s

    0.61604095563139927

```python
#The C parameter trades off misclassification of training examples against simplicity of the decision surface.
#A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly
# by giving the model freedom to select more samples as support vectors.
# Try different C
t0 = time()
clf = svm.SVC(C= 10.0, kernel = 'rbf') # Create classifier, using a linear kernel to avoid too long time for fitting
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
clf.score(features_test, labels_test)
```

    training time: 0.101 s
    predicting time: 1.126 s

    0.61604095563139927

```python
t0 = time()
clf = svm.SVC(C= 100.0, kernel = 'rbf') # Create classifier, using a linear kernel to avoid too long time for fitting
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
clf.score(features_test, labels_test)
```

    training time: 0.102 s
    predicting time: 1.123 s

    0.61604095563139927

```python
t0 = time()
clf = svm.SVC(C= 1000.0, kernel = 'rbf') # Create classifier, using a linear kernel to avoid too long time for fitting
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
clf.score(features_test, labels_test)
```
    training time: 0.096 s
    predicting time: 1.071 s

    0.82138794084186573

```python
t0 = time()
clf = svm.SVC(C= 10000.0, kernel = 'rbf') # Create classifier, using a linear kernel to avoid too long time for fitting
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
clf.score(features_test, labels_test)
# the larger C means more complexity at the boundary
```
    training time: 0.094 s
    predicting time: 0.919 s

    0.89249146757679176

```python
# train the original data set
t0 = time()
clf = svm.SVC(C= 10000.0, kernel = 'rbf')  
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
clf.score(features_test, labels_test)
```

    training time: 102.054 s
    predicting time: 112.441 s

    0.99089874857792948

```python
clf.predict([features_test[10]])
```

```python
import numpy as np
np.shape(np.array([features_test[10]])) # create an array, because [[features_test[10]]] is a list
```

```python
clf.predict(np.array([features_test[26]]))
```
```python
clf.predict(np.array([features_test[50]]))
```

```python
np.count_nonzero(pred)
```
    877

It’s becoming clearer what Sebastian meant when he said Naive Bayes is great for text--it’s faster and generally gives better performance than an SVM for this particular problem. Of course, there are plenty of other problems where an SVM might work better. Knowing which one to try when you’re tackling a problem for the first time is part of the art and science of machine learning. In addition to picking your algorithm, depending on which one you try, there are parameter tunes to worry about as well, and the possibility of overfitting (especially if you don’t have lots of training data).

Our general suggestion is to try a few different algorithms for each problem. Tuning the parameters can be a lot of work, but just sit tight for now--toward the end of the class we will introduce you to GridCV, a great sklearn tool that can find an optimal parameter tune almost automatically.

#### 3.Decision Trees

In this project, we will again try to classify emails, this time using a decision tree.

Part 1: Get the Decision Tree Running
Get the decision tree up and running as a classifier, setting min_samples_split=40.  It will probably take a while to train.  What’s the accuracy?

Part 2: Speed It Up
You found in the SVM mini-project that the parameter tune can significantly speed up the training time of a machine learning algorithm.  A general rule is that the parameters can tune the complexity of the algorithm, with more complex algorithms generally running more slowly.  

Another way to control the complexity of an algorithm is via the number of features that you use in training/testing.  The more features the algorithm has available, the more potential there is for a complex fit.  We will explore this in detail in the “Feature Selection” lesson, but you’ll get a sneak preview now.

Find the number of features in your data.  The data is organized into a numpy array where the number of rows is the number of data points and the number of columns is the number of features; so to extract this number, use a line of code like len(features_train[0])

Go into tools/email_preprocess.py, and find the line of code that looks like this:     selector = SelectPercentile(f_classif, percentile=1)  Change percentile from 10 to 1.

What’s the number of features now?
What do you think SelectPercentile is doing?  Would a large value for percentile lead to a more complex or less complex decision tree, all other things being equal?
Note the difference in training time depending on the number of features.  
What’s the accuracy when percentile = 1?

```python
#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.
    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

```
    no. of Chris training emails: 7936
    no. of Sara training emails: 7884

```python
from sklearn import tree
t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split = 40)
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
```
    training time: 33.273 s
    predicting time: 33.294 s

```python
clf.score(features_test, labels_test)
```
    0.97724687144482369

You found in the SVM mini-project that the parameter tune can significantly speed up the training time of a machine learning algorithm. A general rule is that the parameters can tune the complexity of the algorithm, with more complex algorithms generally running more slowly.

Another way to control the complexity of an algorithm is via the number of features that you use in training/testing. The more features the algorithm has available, the more potential there is for a complex fit. We will explore this in detail in the “Feature Selection” lesson, but you’ll get a sneak preview now.

What's the number of features in your data? (Hint: the data is organized into a numpy array where the number of rows is the number of data points and the number of columns is the number of features; so to extract this number, use a line of code like len(features_train[0]).)

```python
len(features_train),len(features_train[0])
```
    (15820, 3785)

Go into ../tools/email_preprocess.py, and find the line of code that looks like this:

selector = SelectPercentile(f_classif, percentile=10)

Change percentile from 10 to 1, and rerun dt_author_id.py. What’s the number of features now?

SelectPercentile

Select features according to a percentile of the highest scores.

A large value for percentile lead to a more complex decision tree.

```python

import pickle
import cPickle
import numpy

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif


def preprocess(words_file = "../tools/word_data.pkl", authors_file="../tools/email_authors.pkl"):
    """
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

    """

    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project
    authors_file_handler = open(authors_file, "r")
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, "r")
    word_data = cPickle.load(words_file_handler)
    words_file_handler.close()

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)


    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)


    ### feature selection, because text is super high dimensional and
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=1)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    ### info on the data
    print "no. of Chris training emails:", sum(labels_train)
    print "no. of Sara training emails:", len(labels_train)-sum(labels_train)

    return features_train_transformed, features_test_transformed, labels_train, labels_test
```


```python
features_train, features_test, labels_train, labels_test = preprocess()

```
    no. of Chris training emails: 7936
    no. of Sara training emails: 7884

```python
len(features_train[0])
```
    379

```python
from sklearn import tree
t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split = 40)
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
```

    training time: 2.93 s
    predicting time: 2.932 s

```python
clf.score(features_test, labels_test)
```

    0.9670079635949943
