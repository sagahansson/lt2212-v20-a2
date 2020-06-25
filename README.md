# LT2212 V20 Assignment 2

Part 1 - Tokenization and helper functinos

The function extract_features calls the helper function tokenize_to_list for each document. In tokenize_to_list, the email is lowercased and split into tokens. The tokens are then put into a list if they are alphabetical AND if they do not appear in the list stopwords from NLTK. The alphabetical filter is used to remove punctuation, email addresses etc. The list of stopwords is used to remove function words that are of little use when classifying. This puts the number of (unique) tokens at 67492. Back in the extract_features function, all words that occur 10 times or less in total are removed in order to accelerate the processing of the data. Through this process, the number tokens were reduced to 14660.

Another helper function is called in extract_features: word_count. Word_count simply counts the words in a document and places them in a vector of the same length as the lexicon.

Part 2 - Results

Below is a table of the results obtained when running both classifier algorithms on the testing data. In order to be able to reference it later, I would like to note that I also tried a few other classifiers, including SVC and LinearSVC Both SVC algorithms produced results, but also raised warnings that I was unable figure out entirely. For these reasons, I will not include SVC or LinearSVC in the table.  

I find it interesting that both classifiers handled the dimensions of the features being reduced quite badly. LinearSVC did seem to handle the dimensionality reduction with both unreduced and 50%-reduced dimensions attaining scores of around 0.80, but seeing as the dimensionality reduction increased the execution time with approximately 45 minutes, it does not seem like a reasonable option. This indicates that there is no issue with the dimensionality reduction, but rather the method might not be optimal for the Decision Tree Classifier or Gaussian Naïve Bayes. The best performing classifier for the unreduced features, of the two in the table below, Gaussian Naïve Bayes, is interestingly also the one that has the worst performance with the reduced features. The Decision Tree Classifier exhibits the, to me, unusual behaviour of dropping in perfromance at 50% reduced dimensionality, and then rising (even if it's only with 2 percentage points) at 25% reduced dimensionality. A further interesting observation is that the different metrics seem to differ a great deal for Gaussian Naïve Bayes (for each of the different feature dimensionalities), whereas for the Decision Tree Classifier, the metrics are very similar (in the table, they are the exact same due to rounding).  


| Features       | Decision Tree Classifier  | Gaussian Naïve Bayes      | 
|----------------|---------------------------|---------------------------| 
| -------------  | A ---- P ---- R ---- F1   | A ---- P ---- R ---- F1   | 
| 100% - 14660   | 0.63 - 0.63 - 0.63 - 0.63 | 0.77 - 0.77 - 0.77 - 0.77 | 
| 50%  - 7330    | 0.30 - 0.30 - 0.30 - 0.30 | 0.13 - 0.30 - 0.13 - 0.11 |
| 25%  - 3665    | 0.32 - 0.32 - 0.32 - 0.32 | 0.13 - 0.31 - 0.13 - 0.11 |
| 10%  - 1446    | 0.32 - 0.32 - 0.32 - 0.32 | 0.16 - 0.33 - 0.16 - 0.13 |
| 5%   - 733     | 0.32 - 0.33 - 0.32 - 0.33 | 0.15 - 0.35 - 0.15 - 0.15 |


