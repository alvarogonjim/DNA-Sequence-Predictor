# Advanced-Learning-Models
Kaggle data challenge for the course "Advanced Learning Models" for the master 
programs MSIAM and MoSIG.

The purpose of this challenge is to  predict  whether DNA  sequences  is  a  
binding site for 3 transcription factor. No machine learning libraries  were  allowed
 and the competition is on the Kaggle platform (https://www.kaggle.com/c/advanced-learning-models-2019/overview).
Our final results achieve an accuracy in the Public Leaderboard and Private Leaderboard of 0.69400%

The structure of the code is as following:
    * main.py is the main file which has the functions to predict according the given 
    hyper-parameters and save the result

    * start.py the file to reproduce the result submitted on Kaggle platform

    * get_dataset.py, compute_kmer_feature.py, kernels.py and svm.py are the 
    files necessary to read, process the data and build the model for predictions

    * requirements.txt regroup the necessary package to run the scripts 
    (main.py or start.py)

    * Yte.csv is the Kaggle submission

    * Report_GONZALEZ_LAURENDEAU_kaggle_submission.pdf is the report for the 
    Data Challenge project

    * score.txt is the benchmark of the report

    * the folder precomputedi for i=0,1,2 is here to save the Gram matrices 
    already calulated (given empty).
