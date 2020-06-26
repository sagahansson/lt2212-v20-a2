## pca dim reduction

import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
import numpy as np
from nltk.corpus import stopwords
import collections
#from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
#from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier


stopwords = stopwords.words('english')
random.seed(42)


###### PART 1
#DONT CHANGE THIS FUNCTION
def part1(samples):
    #extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X

def tokenize_to_list(onesample):
    # creates lists of words that are lowercase and alphabetiacl 
    lower = onesample.lower()
    word_strings = lower.split()
    only_alpha = [word for word in word_strings if word.isalpha() and word not in stopwords]
    return only_alpha

def word_count(alist, no_of_words, index_dict):
    # produces word counts for a doc, puts counts in the right place in a vector of the same length as the total number of unique words in all docs
    # "the right place" = all wc for word "cat" are placed at the same index, independent of doc 
    count_dict = collections.Counter(alist)
    vector = np.zeros(no_of_words)
    
    for word in alist:
        i = index_dict[word] # fetches the index of the current word from index_dict aka word_idx 
        vector[i] = count_dict[word] # puts the word count in the right place w the help of the index
    return vector



def extract_features(samples):
    print("Extracting features ...")
    
    sample_list = [] # list of lists where each inner list is a doc
    idx = 0 # needed to place the words in the array
    word_idx = {} # dict where each word gets an index 
    for sample in samples:
        sample_words = tokenize_to_list(sample) # creating list of all tokens for each file
        sample_list.append(sample_words) 
        for word in sample_words:
            if word not in word_idx:
                word_idx[word] = idx  # giving each word an index in word_idx
                idx += 1
    no_of_words = len(word_idx)
    no_of_docs = len(samples)
    
    big_array = np.zeros((no_of_docs, no_of_words)) # creating array with as many columns as there are words, and as many rows as there are docs

    sample_no = 0 # keeps track of which row we're on
    for sam in sample_list: # i e for every doc
        big_array[sample_no] = word_count(sam, no_of_words, word_idx) # gets word counts of file, puts them at the right place in the array by looking at word_idx
        sample_no += 1
    print('Features extracted.')
    print('Number of words before frequency filtering: ', big_array.shape[1]) 
    
    big_arr_sum = np.sum(big_array, axis =0) # gets total word counts
    array_filter = big_arr_sum > 10 # boolean filter -- everything in big_array_sum that is over 10 gets True, everything else False
    filtered_arr = big_array[:, array_filter] # filters out words that occur less than 10 times in total
    
    print('Number of words after filtering: ', filtered_arr.shape[1])
        
    return filtered_arr



##### PART 2
#DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X,n=10):  
    pca = PCA(n_components=n)
    print('Fitting and transforming')
    fit_transformed = pca.fit_transform(X)
    return fit_transformed 



##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = DecisionTreeClassifier()
    elif clf_id == 2:
        clf = GaussianNB()
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf

#DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    #PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)


    # evalute model
    print("Evaluating classcifier ...")
    evalute_classifier(clf, X_test, y_test)


def shuffle_split(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_classifer(clf, X, y):
    assert is_classifier(clf)
    clf.fit(X, y)
    

def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    pred = clf.predict(X)
    acc = metrics.accuracy_score(y, pred)
    prec = metrics.precision_score(y, pred, average='weighted')
    rec = metrics.recall_score(y, pred, average='weighted')
    f1 = metrics.f1_score(y, pred, average='weighted')
    print('Accuracy: ', acc)
    print('Precision: ', prec)
    print('Recall: ', rec)
    print('F1-measure: ', f1)

######
#DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names

#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(   
            model_id=args.model_id, 
            n_dim=args.number_dim_reduce
            )
