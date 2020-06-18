import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
import numpy as np
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
    only_alpha = [word for word in word_strings if word.isalpha()]
    return only_alpha

def word_count(alist, words, index_dict):
    #puts counts instead of zeroes in np.zeroes
    count_dict = collections.Counter(alist)
    vector = np.zeros(words)
    
    for word in alist:
        i = index_dict[word]
        vector[i] = count_dict[word]
    return vector




def extract_features(samples):
    print("Extracting features ...")
    
    sample_list = []
    idx = 0
    word_idx = {}
    # gör en 0-array med lika många "lists" i sig som det finns samples i samples OCH lika många nollor i varje "list" som det finns tokens totalt
    # -> np.zeros(bredd, höjd) : bredd = antal tokens = len(word_idx), höjd = antal "dokument" = len(samples)
    for sample in samples:
        sample_words = tokenize_to_list(sample)
        sample_list.append(sample_words) # list of lists where each inner list is a doc
        for word in sample_words:
            if word not in word_idx:
                word_idx[word] = idx
                idx += 1
    antal_words = len(word_idx)
    antal_docs = len(samples)
    
    big_array = np.zeros((antal_docs, antal_words))

    sample_no = 0
    for sam in sample_list:
        big_array[sample_no] = word_count(sam, antal_words, word_idx) # call counter funktion som räknar ord i varje sample + lägger deras värden på rätt plats i rätt arr i nparray
        sample_no += 1
        
    return big_array



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
    #fill this in
    pass



##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = "" # <--- REPLACE THIS WITH A SKLEARN MODEL
    elif clf_id == 2:
        clf = "" # <--- REPLACE THIS WITH A SKLEARN MODEL
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
    pass # Fill in this


def train_classifer(clf, X, y):
    assert is_classifier(clf)
    ## fill in this


def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    #Fill this in


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
