# Advanced learning models 2019-2020
Kaggle data challenge for the course "Advanced Learning Models" for the master 
programs MSIAM and MoSIG.

The purpose of this challenge is to  predict  whether DNA  sequences is a binding site for 3 transcription factor. No machine learning libraries  were  allowed
and the competition is on the Kaggle platform (https://www.kaggle.com/c/advanced-learning-models-2019/overview).
Our solution got in the rank 6th/45th (Top 14%) with an accuracy of 0.69400 in the Private Leaderboard.


## Approach

Our solution combines the K-mer with a mismatch kernel and a Support Vector Machine (SVM), the details of the approach and the parameters can be found in the report folder.

### Installing

First you need to get the repository and install the dependencies, use the following commands:

```
$git clone https://github.com/alvarogonjim/Advanced-Learning-Models.git
$cd Advanced-Learning-Models
$pip install -t requirements.txt
```

Then you can execute the main program and wait for the results that will be stored as 'Yte_GONZALEZ_LAURENDEAU_kaggle_submission.csv', use the following command:

```
$python main.py
```

### Structure:
The structure of the code is as following:
- *main.py* is the main file which has the functions to predict according the given 
    hyper-parameters and save the result

- *start.py* the file to reproduce the result submitted on Kaggle platform

- *get_dataset.py*, compute_kmer_feature.py, kernels.py and svm.py are the 
    files necessary to read, process the data and build the model for predictions

- *requirements.txt* regroup the necessary package to run the scripts 
    (main.py or start.py)

- *Yte.csv* is the Kaggle submission

- *Report_GONZALEZ_LAURENDEAU_kaggle_submission.pdf* is the report for the 
    Data Challenge project

- *score.txt* is the benchmark of the report

- The folder precomputedi for i=0,1,2 is here to save the Gram matrices 
    already calulated (given empty).


### Coding style

We have followed the PEP-8 styles and used the Black library to format our code.

## Built With

- [Numpy](https://numpy.org) - Work with arrays and matrices
- [Pandas](https://pandas.pydata.org) - Manipulate the dataframes
- [Cvxopt](https://cvxopt.org/) - Solve quadratic programming problems.

## Authors

- **Álvaro González Jiménez** - _alvarogonjim95@gmail.com_ - [Website](http://alvarogj.sytes.net/)
- **Matthieu Laurendeau** - _laurendeau.matthieu@gmail.com_ - [Github](https://github.com/LAURENDEAU)

## References

- E. Eskin, J. Weston, W. S. Noble, and C. S. Leslie. Mismatch string kernels for svm protein classication. In Advances in neural information processing systems
- C. Leslie, E. Eskin, and W. S. Noble. The spectrum kernel: A string kernel for svm protein classication.

