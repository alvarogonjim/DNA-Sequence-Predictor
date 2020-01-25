"""
@authors: GONZALEZ Alvaro: alvarogonjim95@gmail.com
        & LAURENDEAU Matthieu: laurendeau.matthieu@gmail.com
@date: 2020 January
@brief: Save submission for the kaggle challenge under a file called: 
        'Yte_GONZALEZ_LAURENDEAU_kaggle_submission.csv'
"""

############ Imports ############
"""
Libraries necessary to run this file alone.
"""
import pandas as pd # for data management

### TO DELETE
def reverse_predictions(test_predictions):
    """
    Reverse prediction to don't show on the leaderboard our best result
    @param: test_predictions: numpy array - predictions
    """
    return ~test_predictions+2
### TO DELETE

def post_procesing(test_predictions):
    """
    Post processing of the data after taining and before submitted.
    @param: test_predictions: numpy array - predictions
    """
    return test_predictions # nothing to do

def save_submission(test_predictions, title=""):
    """
    Save submission in the kaggle challenge format
    @param: test_predictions: numpy array - predictions to submit
            title: string - title of the submitted file. Default = ""
    """
    test_predictions = pd.DataFrame({"Bound": test_predictions}) # convert in pandas df
    test_predictions.index.name = 'Id'
    test_predictions = reverse_predictions(test_predictions) # TO DELETE
    test_predictions = post_procesing(test_predictions)
    test_predictions.to_csv("Yte_"+title+".csv", sep=",", encoding="utf-8", index=True)

############ Main ############
''' If the file is executed separetely '''
if __name__ == "__main__":
    print("This file need prediciton on test data set to be run.")