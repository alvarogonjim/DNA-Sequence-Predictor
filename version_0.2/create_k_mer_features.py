"""
@authors: GONZALEZ Alvaro: alvarogonjim95@gmail.com
        & LAURENDEAU Matthieu: laurendeau.matthieu@gmail.com
@date: 2020 January
@brief: Create the k-mer features for the DNA data given on kaggle
"""

def apply_k_mer_features(data, features, out_print=""):
    """
    Apply k-mer features to the data. It constructs the distribution matrix of 
    size len(data) x len(features). Each cell say how many times
    the sequence 'i' contains the k-mer 'j' normalized by the number of 
    occurences over the sequences.
    @param: data: pandas dataframe - data on whcih we apply the k-mer features
            letters: list of characters - avalaible letter
            features: list of string - full list of k-mer = label in the matrix
            out_print: string - line to print for see the progress. Default = ""
    """
    c = 0 # counter to print the progress
    for i in features:
        c += 1
        print(out_print + str(round(c*100/len(features),3)) + "%", end="\r")
        data[i] = data.seq.str.count(i) # count occurences
    print("")
    data = data.drop("seq", axis=1) 
    data=(data-data.mean())/data.std() # normalization
    # data=(data-data.min())/(data.max()-data.min()) # normalization
    return data

def list_all_k_mer(features, letters, feature, n, k, init_k):
    """
    Construct the full list of possible k-mer. It is like 
    provide all the words possible of a given length and knowing the available 
    letters. We do it recursively.
    @param: features: list of string - result
            letters: list of characters - avalaible letter
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


def create_k_mer_features(train_set, \
                        validation_set, test_set, k):
    """
    Create the k-mer features of length k for each set
    @param: train_set: pandas dataframe - training set
            validation_set: pandas dataframe - validation set
            test_set: pandas dataframe - test set
            k: integer - length of the k-mer
    """
    print("Create k-mer features.")
    features = []
    letters = ['A', 'T', 'C', 'G']
    n = len(letters)
    feature = ""
    features = list_all_k_mer(features, letters, feature, n, k, k)

    # Apply k-mer features to train_set
    out_print = "Apply k-mer features on train data set: "
    train_set = apply_k_mer_features(train_set, features, out_print)

    # Apply k-mer features to validation_set
    out_print = "Apply k-mer features on validation data set: "
    validation_set = apply_k_mer_features(validation_set, features, out_print)

    # Apply k-mer features to test_set
    out_print = "Apply k-mer features on test data set: "
    test_set = apply_k_mer_features(test_set, features, out_print)

    return train_set, validation_set, test_set

############ Main ############
''' If the file is executed separetely '''
if __name__ == "__main__":
    print("This file need train data and test data set at least to be run.")