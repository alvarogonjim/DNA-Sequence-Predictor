# Advanced learning models 2019-2020

This is the repository that contains our soulution for Advanced Learning Models's data challenge for the master programs MSIAM and MoSIG. Our solution got in the rank 6th/45th (Top 14%) with an accuracy of 0.69400.

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
