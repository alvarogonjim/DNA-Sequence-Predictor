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
import pandas as pd  # for manipulate the dataframe
from multiprocessing import Pool  # for multiprocess the code
from functools import partial  # for create partial object
from itertools import combinations, product  # for list operations

# def count_occurence(s, ss):
#     """
#     Count occurence of substring in a string (taking care of overlapping)
#     @param: s: string - source string
#             ss: string - substring
#     """
#     # Initialize count and start to 0
#     count = 0
#     start = 0

#     # Search through the string till
#     # we reach the end of it
#     while start < len(s):

#         # Check if a substring is present from
#         # 'start' position till the end
#         flag = s.find(ss, start)

#         if flag != -1:
#             # If a substring is present, move 'start' to
#             # the next position from start of the substring
#             start = flag + 1

#             # Increment the count
#             count += 1
#         else:
#             # If no further substring is present
#             # return the value of count
#             return count


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
                if source[i] == r:
                    skip = True
            if skip:
                continue

            # Possible letters to use for replacement
            rep = dict(zip(indices, replacements))

            # Replace a letter and then fill with the source word
            yield "".join([source[i] if i not in indices else rep[i] for i in range(N)])


def apply_kmer_features(data, k, m, letters, full_active_kmer, full_active_mismatch):
    """
    Apply k-mer features on all datasets. It constructs a list of dictionaries
    where each dictionary represent the number of occurence for one k-mer and 
    all its neighbours (= mismatch according to m).
    @param: data: pandas dataframe - dataset on which we apply the kmer feature
            k: int - length of k-mer
            m: int - number of mismatch allowed. Default = 0
            letters: list of characters - avalaible letters
            full_active_kmer: list of string - list of all active kmers in the 
            datasets
            full_active_mismatch: dictionary - saving all mismatch for one 
            kmer which appear in the datasets. It is for avoir repetitive 
            calculations
    """
    data_kmer = []
    counter = 0  # for see the progress
    for s in data:
        seq_kmer = {}
        for cursor in range(len(s) - k + 1):
            counter += 1
            print(
                str(round(counter / (len(data) * (len(s) - k + 1)) * 100)).zfill(3)
                + "%",
                end="\b\b\b\b",
            )

            current_kmer = s[cursor : cursor + k]

            # Add new entry in the dictionary
            if current_kmer not in full_active_mismatch:
                current_mismatch = list(list_mismatch(current_kmer, letters, m))
                if m != 0:
                    current_mismatch += [current_kmer]
                active_mismatch = list(
                    set(current_mismatch).intersection(full_active_kmer)
                )
                full_active_mismatch[current_kmer] = active_mismatch

            for kmer in full_active_mismatch[current_kmer]:
                if kmer not in seq_kmer:
                    seq_kmer[kmer] = 1
                else:
                    seq_kmer[kmer] += 1

        data_kmer += [seq_kmer]

    return data_kmer


# def apply_kmer_features(k, letters, m, keep_zeros, full_kmer, s):
#     """
#     Apply k-mer features of one sequence s. It constructs a list of dictionaries
#     where each dictionary represent the number of occurence for one k-mer and
#     all its neighbours.
#     @param: k: int - length of k-mer
#             letters: list of characters - avalaible letters
#             m: int - number of mismatch allowed. Default = 0
#             s: string - current sequence on which we apply k-mer feature
#             keep_zeros: bool - whether keep neighbours of a k-mer even if it
#             does not appear in the sequence. Default = False.
#     """
#     # full_kmer = [s[i:i+k] for i in range(len(s) - k + 1)]
#     occurences = {}
#     # print(s)
#     for i in range(len(s) - k + 1):
#         current_kmer = s[i:i+k]
#         # if current_kmer in occurences:
#         #     continue
#         mismatch = list(list_mismatch(current_kmer,letters,m))
#         if m != 0:
#             mismatch += [current_kmer]

#         # print(mismatch, full_kmer)

#         mismatch_appeared = list(set(mismatch).intersection(full_kmer))
#         # print(mismatch, len(mismatch))
#         print(mismatch_appeared, len(mismatch_appeared))
#         # print(full_kmer, len(full_kmer))
#         assert(False)
#         # mismatch_appeared = [x for ele in mismatch for x in full_kmer if x in ele]

#         print(mismatch_appeared, len(mismatch_appeared))
#         assert(i <6)
#         # words = mismatch
#         # print(words)
#         # assert False, mismatch_appeared
#         for w in mismatch_appeared:
#             if w in occurences:
#                 occurences[w] += 1 #count_occurence(s,w)
#                 # continue
#             else:
#                 occurences[w] = 1 #count_occurence(s,w)

#     if not keep_zeros:
#         occurences = {k: v for k, v in occurences.items() if v != 0}

#     # print(occurences)
#     # if s == "GCCTCCCTTGGCACCACGGGAGACCAGTTTTGGAGGGGCGGGGCTGCAGGGGGGCGAGCCCCACTGTCAGGAAGGCTGAAGTTTCGGGGCAGAGTGCTAAA":
#     #     assert(False)
#     # print(occurences)
#     # print(len(occurences))
#     # assert(False)
#     return [occurences]

# def apply_kmer_features(k, letters, m, keep_zeros, full_kmer, oui, s):
#     """
#     Apply k-mer features of one sequence s. It constructs a list of dictionaries
#     where each dictionary represent the number of occurence for one k-mer and
#     all its neighbours.
#     @param: k: int - length of k-mer
#             letters: list of characters - avalaible letters
#             m: int - number of mismatch allowed. Default = 0
#             s: string - current sequence on which we apply k-mer feature
#             keep_zeros: bool - whether keep neighbours of a k-mer even if it
#             does not appear in the sequence. Default = False.
#     """
#     # full_kmer = [s[i:i+k] for i in range(len(s) - k + 1)]
#     occurences = {}
#     # print(s)
#     for i in range(len(s) - k + 1):
#         current_kmer = s[i:i+k]

#         mismatch_appeared = oui[current_kmer]
#         # print(mismatch)
#         # print(full_kmer, len(full_kmer))
#         # assert(False)
#         # mismatch_appeared = [x for ele in mismatch for x in full_kmer if x in ele]

#         # print(mismatch_appeared, len(mismatch_appeared))
#         # assert(False)
#         # words = mismatch
#         # print(words)
#         # assert False, mismatch_appeared
#         for w in mismatch_appeared:
#             if w in occurences:
#                 occurences[w] += 1 #count_occurence(s,w)
#                 # continue
#             else:
#                 occurences[w] = 1 #count_occurence(s,w)

#     if not keep_zeros:
#         occurences = {k: v for k, v in occurences.items() if v != 0}

#     # print(occurences)
#     # if s == "GCCTCCCTTGGCACCACGGGAGACCAGTTTTGGAGGGGCGGGGCTGCAGGGGGGCGAGCCCCACTGTCAGGAAGGCTGAAGTTTCGGGGCAGAGTGCTAAA":
#     #     assert(False)
#     print(occurences)
#     print(len(occurences))
#     assert(False)
#     return [occurences]


def list_all_kmer(features, letters, feature, n, k, init_k):
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
    if k == 0:
        features += [feature]
        return features

    for i in range(n):
        # Adding letter in the current word
        new_feature = feature + letters[i]
        # Recursivity
        full_features = list_all_kmer(features, letters, new_feature, n, k - 1, init_k)
        # Final return
        if full_features is not None:
            if full_features[-1] == init_k * letters[-1]:
                return features


def active_kmer(train_set, validation_set, test_set, k):
    """
    Give the list of all the kmers of length k inside all datasets
    @param: train_set: pandas dataframe - training set
            validation_set: pandas dataframe - validation set
            test_set: pandas dataframe - test set
            k: int - length of the k-mer
    """
    data = pd.concat([train_set, validation_set, test_set], axis=0, ignore_index=True)
    active_kmer = []

    for s in data["seq"]:
        for l in range(len(s) - k + 1):
            current_kmer = s[l : l + k]
            # if current_kmer not in active_kmer:
            active_kmer += [current_kmer]

    return list(dict.fromkeys(active_kmer))


# def active_mismatch_kmer(full_active_kmer, letters, m):
#     active_mismatch_kmer = {}
#     for kmer in full_active_kmer:
#         mismatch = list(list_mismatch(kmer,letters,m))
#         if m != 0:
#             mismatch += [kmer]

#         active_mismatch_kmer[kmer] = list(set(mismatch).intersection(full_active_kmer))
#     return active_mismatch_kmer

# def active_mismatch_kmer(letters, m, full_active_kmer, kmer):
#     mismatch = list(list_mismatch(kmer,letters,m))
#     if m != 0:
#         mismatch += [kmer]

#     return kmer, list(set(mismatch).intersection(full_active_kmer))


def compute_kmer_feature(train_set, validation_set, test_set, k, m=0):
    """
    Create the k-mer feature of length k for each set
    @param: train_set: pandas dataframe - training set
            validation_set: pandas dataframe - validation set
            test_set: pandas dataframe - test set
            k: integer - length of the k-mer
            m: interger - number of mismatch allowed in the sequences. 
            Default = 0 --> exact k-mer subsequences
    """
    letters = ["A", "T", "C", "G"]  # DNA letters
    # List all possible k-mer of length k according to the 4 DNA letters
    full_kmer = []
    n = len(letters)
    feature = ""
    full_kmer = list_all_kmer(full_kmer, letters, feature, n, k, k)
    # List all actives kmer over all datasets
    full_active_kmer = active_kmer(train_set, validation_set, test_set, k)
    print(
        "\t\t"
        + str(len(full_kmer))
        + " possible k-mers | "
        + str(len(full_active_kmer))
        + " are actives."
    )

    full_active_mismatch = {}

    # Apply k-mer features to train set
    print("\t\tApplying feature kmer map on training data: ", end="")
    train_set_kmer = apply_kmer_features(
        train_set["seq"], k, m, letters, full_active_kmer, full_active_mismatch
    )
    print("\r\t\tFeature kmer map applied on training data.      ")

    # Apply k-mer features to validation set
    validation_set_kmer = []
    if not validation_set.empty:
        print("\t\tApplying feature kmer map on validation data: ", end="")
        validation_set_kmer = apply_kmer_features(
            validation_set["seq"], k, m, letters, full_active_kmer, full_active_mismatch
        )
        print("\r\t\tFeature kmer map applied on validation data.      ")

        # Apply k-mer features to test set
    print("\t\tApplying feature kmer map on test data: ", end="")
    test_set_kmer = apply_kmer_features(
        test_set["seq"], k, m, letters, full_active_kmer, full_active_mismatch
    )
    print("\r\t\tFeature kmer map applied on test data.      ")

    return train_set_kmer, validation_set_kmer, test_set_kmer
