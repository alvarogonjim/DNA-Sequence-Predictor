__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__date__ = "2020 January"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org, laurendeau.matthieu@gmail.com"
__status__ = "Submitted"
__brief_compute_kmer_feature__ = "Create the k-mer features for the DNA data \
given on kaggle. K-mer are subsequences of length k contained in a DNA sequence \
(or more generally in a string). Thus, for each sequences in the datasets, it \
calculates the number of times where a k-mer appear in this sequence. We allow \
that the subsequence mismatch with the k-mer label with a number of m \
dissimilarity."

############ Imports ############
"""
Libraries necessary to run this file alone.
"""
from multiprocessing import Pool # for multiprocess the code
from functools import partial # for create partial object
from itertools import combinations, product # for list operations

def count_occurence(s, ss):
    """
    Count occurence of substring in a string (taking care of overlapping)
    @param: s: string - source string
            ss: string - substring
    """
    # Initialize count and start to 0 
    count = 0
    start = 0
  
    # Search through the string till 
    # we reach the end of it 
    while start < len(s): 
  
        # Check if a substring is present from 
        # 'start' position till the end 
        flag = s.find(ss, start) 
  
        if flag != -1: 
            # If a substring is present, move 'start' to 
            # the next position from start of the substring 
            start = flag + 1
  
            # Increment the count 
            count += 1
        else: 
            # If no further substring is present 
            # return the value of count 
            return count 

def list_mismatch(kmer, letters, m=0):
    """
    Construct the list of all possible m-mismatch of the string kmer
    @param: kmer: string - source word
            letters: list of characters - avalaible letters
            m: int - number of mismatch allowed. Default = 0
    """
    N = len(kmer)
    source = list(kmer)

    for indices in combinations(range(N), m):
        for replacements in product(letters, repeat=m):
            # Skip if current letter to change is equal to replacement letter
            skip = False
            for i, r in zip(indices, replacements):
                if source[i] == r: skip = True
            if skip: continue

            # Possible letters to use for replacement
            rep = dict(zip(indices, replacements))
            
            # Replace a letter and then fill with the source word
            yield ''.join([source[i] if i not in indices else rep[i] 
                           for i in range(N)])


def apply_k_mer_features(k, letters, m, keep_zeros, s):
    """
    Apply k-mer features of one sequence s. It constructs a list of dictionaries
    where each dictionary represent the number of occurence for one k-mer and 
    all its neighbours.
    @param: k: int - length of k-mer
            letters: list of characters - avalaible letters
            m: int - number of mismatch allowed. Default = 0
            s: string - current sequence on which we apply k-mer feature
            keep_zeros: bool - whether keep neighbours of a k-mer even if it 
            does not appear in the sequence. Default = False.
    """
    occurences = {}
    for i in range(len(s) - k + 1):
        kmer = s[i:i+k]
        mismatch = list(list_mismatch(kmer,letters,m))
        words = [kmer] + mismatch
        for w in words:
            if w in occurences:
                occurences[w] += count_occurence(s,w)
            else:
                occurences[w] = count_occurence(s,w)
    if not keep_zeros:
        occurences = {k: v for k, v in occurences.items() if v != 0}
    return [occurences]


def list_all_k_mer(features, letters, feature, n, k, init_k):
    """
    Construct the full list of possible k-mer. It is like 
    provide all the words possible of a given length and knowing the available 
    letters. We do it recursively.
    @param: features: list of string - result
            letters: list of characters - avalaible letters
            feature: string - the current feature constructing during the 
                    recursivity
            n: int - number of letters
            k: int - length of the k-mer / words. It is this parameter 
                    that we are changing (= decreasing) each iteration of 
                    the recursivity.
            init_k: int - initial k used for the final return
    """
    # Finishing the word and put in the features list
    if (k == 0):
        features += [feature]
        return features
    
    for i in range(n):
        # Adding letter in the current word
        new_feature = feature + letters[i]
        # Recursivity
        full_features = list_all_k_mer(features, letters, new_feature, n, k - 1, init_k)
        # Final return
        if full_features is not None:
            if full_features[-1] == init_k*letters[-1]:
                return features


def compute_kmer_feature(train_set, \
                        validation_set, test_set, k, m=0, keep_zeros=False):
    """
    Create the k-mer feature of length k for each set
    @param: train_set: pandas dataframe - training set
            validation_set: pandas dataframe - validation set
            test_set: pandas dataframe - test set
            k: integer - length of the k-mer
            m: interger - number of mismatch allowed in the sequences. 
            Default = 0 --> exact k-mer subsequences
            keep_zeros: bool - whether keep neighbours of a k-mer even if it 
            does not appear in the sequence. Default = False.
    """
    # Get the full k-mer list of length k
    features = []
    letters = ['A', 'T', 'C', 'G']
    n = len(letters)
    feature = ""
    # List all possible k-mer of length k according to the 4 DNA letters
    features = list_all_k_mer(features, letters, feature, n, k, k)
    # # Convert in a dictionary
    # dict_features = { i : features[i] for i in range(0, len(features) ) }
    print("\t\t" + str(len(features)) + " possible k-mers.")

    pool = Pool(processes=8)

    # Apply k-mer features to train_set
    train_set_k_mer = []
    for result in pool.imap_unordered(partial(apply_k_mer_features, k, \
        letters, m, keep_zeros), train_set["seq"]):
        train_set_k_mer += result
    print("\t\tFeature k-mer map applied on training data.")


    # Apply k-mer features to validation_set
    validation_set_k_mer = []
    if not validation_set.empty:
        for result in pool.imap_unordered(partial(apply_k_mer_features, k, \
            letters, m, keep_zeros), validation_set["seq"]):
            validation_set_k_mer += result
        print("\t\tFeature k-mer map applied on validation data.")

    # Apply k-mer features to test_set
    test_set_k_mer = []
    for result in pool.imap_unordered(partial(apply_k_mer_features, k, \
        letters, m, keep_zeros), test_set["seq"]):
        test_set_k_mer += result
    print("\t\tFeature k-mer map applied on test data.")

    return train_set_k_mer, validation_set_k_mer, test_set_k_mer