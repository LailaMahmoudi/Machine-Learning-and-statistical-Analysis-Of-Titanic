# Machine-Learning-and-statistical-Analysis-Of-Titanic

## Goal: 

- The mean goal of this repository is to predict if a passenger survived the sinking of the Titanic or not, based on multiple features such as sex, age ,...etc.

## Steps :

- EDA
- Feature Engineering
- Modeling
- Hyperparameter Tunning
- Predictions


## Requirements:

- __Python__
- [__JupyterNotebook__](https://jupyter.org/install)
- [__Sklearn__](https://scikit-learn.org/stable/install.html)
- [__Numpy__](https://numpy.org/install/)
- [__Pandas__](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
- [__Matplotlib__](https://matplotlib.org/3.3.2/users/installing.html).
- [__Seaborn__](https://seaborn.pydata.org/installing.html)


## DataSet :

- __Kaggle__ (https://www.kaggle.com/c/titanic/data).

# Step 1 : Exploratory Data Analysis

In this phase we will extract the dataset and explore it, and we will do some descriptive statistics, and visualize our data.

 - __Data Extraction__
 - __Viewing the data__
 - __Descriptive statistics__
 - __Data Visualization__
 
  ![](https://www.kaggleusercontent.com/kf/45328398/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..RYRkrmeMA1tPmW4_EhoXNA.2xvrqL1jyjnFThTkGo8td9BpC4bzGs55TNYb5Ko7bP1N-g3_WGRq4WuYjZ6pdrmWbQS3193g9HL1pjWg7ki66SWE-z5tunnysy41V7m7F094D--PIJl7-OjVr1aSUgBzkH_CFbOYW2Zc6MBvgy1Xrh1xwARn8G7sabU6y2drErclMRZXM-ZXwnThhOmj_yJH6GM1_338RxKqmpZlbKDeLwdj2q6u9jDjlGsjLPn625ksMS-BPcDqQl74x8Y8t8FQjHP2oL9WD9lO_2rD4qo3iKwDAwhGj-CTG4lyCf7wRKWgTq0dvosjHeXyqqzcUKecyeE8i7bxVuCxoHEJvH1oVncZTVt06L_fPTpKmaguM60bdtbEWXqeM4KTDEH99oC3R7127ARol_NCaUbyld4QQV5Srq6WAwk8vJOGA1yqP8l513ge7esXdBuuCKXBKuSCJkf2cePkRamWYddb359lDLQIGGN2Zm_1V5YLIwhcIpilWLj8qyj9x1E5GIwgjeRQd_3KlBiJpBiFbG8HYVBQgUe0vNvAWUbzrSYnBRzwlxwIYR76rOU-OP4-iDKSONfGrfRq3j60fEtIFkyhThUnjxDAEfaX_dE-7grzLhkM_NTzXAYpV_4iErcbfuSK6MZJxadPrRmDiZKfKGcAx07whsVUdLnQG8-qSvZA7yTNc4SYFEV1TAKtp_SctUic8tWe.ATpeN3FaSOZOH_sPsDZuUg/__results___files/__results___26_1.png)
  
  
  ![](https://www.kaggleusercontent.com/kf/45328398/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..ozUvT0OQ6up7FnzalgqK2w.qmRBeybR_4EAu7ALz8TCmeyzt5KxLgMQWZeO2n_nf7HQ8Tjk0lgyk-cRf3ffaRtzHwjKdV5_YY0P9Yac0ATUHjxGwK9wiPBy3rPjICUwTQ3OGadj_d83wehZbtBhFUNTkvuI5BPRaTQkL--xVf_YjyEU6l7miworc3M63-s1PU4_RbzUZeRfuyaBrAPHbe4fK3a8zMM2-Xvk1wqr3RXVCsXhNtngMgDt1Dx66nhXMM_JMSDkiWdWZmVWGEHmfut99Wd8Kj4zOo1VmYDZR4uTRi1JtfLiRDq3Rto3R3SxdHiYvVxsLRoxrotlmdLitA13mO7GmmQtc_5GDFpnxN9AtGdT45I7-gTVI5-h1Fk_QeaQJfsaC7UM6u2XHEXN8br90NoimEH-n5-sdD76gFi2DiiYJASaEFXNcBjVc5qN9L-5I0huiVFUrCoeXgzEwbQocfA0lQknWLG_CulrEbVsVW2e7KXHaRpYs0Tn4wedo_r2H5eVAZFj3ypGt9JaDLgi7EfwSI__wT5n2YyxzTFvGXtcBeRbPgUMM5NY5wMA_9dfAVCjbja415obZFjFUa4adLgiQEC4lSn0P4jMwoE0Glog01akw9hklHlNGf6B4Zen_odf72t4w0tWJbAZXS5-yGIIzRS_kvknn3Xqvf77BiuPqyD6GX8KdU8O5NYXEgQ2JOIjzfTBeosFc24TlUxK.k6hHdA7NvvRlLAzVHS0Gog/__results___files/__results___40_0.png)

# Step 2 : Feature Engineering

A critical part of the success of a Machine Learning Project is Feature Engineering. 

__Feature Engineering__ is a process of transforming the data into data which is easier to interept and also, to increase the predictive power of learning algorithm

In Feature Engineering we will create a new features that could improve predictions such as if the passenger is alone or not,
and combining existing features to produce a more useful one, and dropping the columns doesn't improve predictions.


# Step 3 : Pre-Modeling Tasks

- Separating the independant and the dependant variable.
- Splitting the training data.
- Feature Scaling 

# Step 4 : Modeling

In this part we'll try differents models of Machine learning: Logistic Regression, Random Forest, Support Vector Machine, Decision tree, GaussianNB Model and KNeighbors Model

# Step 5 : Performance Metrics

Evaluating the machine learning model is a crucial part in any data science project. There are many metrics that helps us to evaluate our model accuracy.

-Classification Accuracy
-Confusion matrix
-Precision Score
-Recall Score
-AUC & ROC Curve


# Step 6 : Hyperparameter Tuning using Grid Search

- HyperParameter is a configuration external which the value can not be estimated by the given data, , Tuning hyperparameters is an important part of building a machine learning system.

- Some examples of hyperparameters:

  - Number of leaves or depth of a tree
  - Number of latent factors in a matrix factorization
  - Learning rate (in many models)
  - Number of hidden layers in a deep neural network
  - Number of clusters in a k-means clustering

- There are several ways to tune a hyperparameter:
    
 - Manually   : we select the hyperparameter based on intuition, or experience.

- Grid Search: is a commonly method used to tuning a hyperparameter, it takes the model we would to train and differents values of the hyperparameters, it then       calculate the mean square error or R-squared for various hyperparameters values, allowing you to choose the best values, so we use differents hyperparameters and   we continue the process until we have the differents free parameters values and then each model produces an error and we select the hyperparameter that minimizes   the error.
    
- Random Search:  we set up a grid of hyperparameters and select a random combinations to train the model and the score.

- Automated Hyperparameter Tuning: we use methods like Bayesian Optimization, gradient descent..

 - In this notebook, We will tune our models using Grid Search CV 
 
 # Step 7 : Predictions and submit the results
 
 

# Useful Resources :

- [Fundamental Techniques of Feature Engineering for Machine Learning](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114).

- [Ensemble Machine Learning Algorithms in Python with scikit-learn](https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/).

- [ZacharyJWyman/ML-Techniques](https://github.com/ZacharyJWyman/ML-Techniques).

- [How to Build a Machine Learning Model](https://towardsdatascience.com/how-to-build-a-machine-learning-model-439ab8fb3fb1).

- [Ensemble methods: bagging, boosting and stacking](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205).

- [How to find optimal parameters using GridSearchCV?](https://www.dezyre.com/recipes/find-optimal-parameters-using-gridsearchcv).

- [Gradient Boosting Classifiers in Python with Scikit-Learn](https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/).
