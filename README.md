# Machine-Learning-and-statistical-Analysis-Of-Titanic
The mean goal of this repository is to predict if a passenger survived the sinking of the Titanic or not, based on multiple features such as sex, age ,...etc.


![ ](https://scontent-arn2-2.xx.fbcdn.net/v/t1.0-9/57429716_2030860760555937_2750062083545497600_n.jpg?_nc_cat=100&ccb=2&_nc_sid=8bfeb9&_nc_ohc=y5d_WudgJy0AX9PNi5o&_nc_ht=scontent-arn2-2.xx&oh=a630ca82a715594c37e77c981b2f0b12&oe=6004F26F)

## Steps :

- STEP 1: Exploratory Data Analysis
- STEP 2: Feature Engineering
- STEP 3: Pre-Modeling Tasks
- STEP 4: Modeling
- STEP 5: Evaluating the performance of the model
- STEP 6: Predictions and submission


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

 ### - __Data Extraction__
 ### - __Viewing the data__
 ### - __Descriptive statistics__
 
 ###### Correlation Map
   
![](https://www.kaggleusercontent.com/kf/50302331/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..61eQA5kAS7Vbn9dgdXA6Fg.owQrqdb6Qrfo8YL3sSlSFUnZJGj387fy-Dx2e7k3LNZg_3L4eLuodEz6tttCI3ySa7NL0G-AbhQn5A85X23o3imDuQ3UDt-OprWqsaaLZLM3rEJLvBQyVQsUKHOiCOYPT58dA8NpSXduTOviWCFhx_Bcstiv35JdLAbQdf5wzvem1v_t5IeWmq0PoRR7dC-yFq7YDIO-hayrpmFVd-_Dkvut-2qDF1QL1jRRJu_dL8KczHHM5L4zxBiZw38MrsikBQ07wB6yVl9fyr7o_p-Agn-6qwpT6UU_qepb4DPmsGS7HTkLecEasIpspz-deRTXSg2FEO5Iiis0ERfjdIlf6LzC4Ovy469iiu9j8rwRh6k8bRi9JGrQIsb_DEIE1DJc76PVPqxffVp3okPsNvxkeWJ92L4hrjkLjTXDox1HFHifn8eoKGEewfb6JePnX6u9bn1fxqlVwxfO3yuUwW5jWKGZ70nv1Rg5FuFaT-f0hBQMQOsI_KzVDpyXKGAAZFZn52PX7_xcDJyTb52tqiu4n9kygYb1rPNF32AbN1TvdkwF-b775VX9oJDGSoV3o-GZxtKLtHdH9HoMo9LUgeWkZlUktKrVOWpBgh70hxdHZGq-t7E9RNDOYqvLjSqrFwztDnHkQMBlQDOFstv7f0nUHn1S28UDRJWRxfOarA3mLWLfaz3ASoUnADcsbWEEBdwy.HarOxfOP0TerXCwJvDZ--g/__results___files/__results___22_0.png)
       
## - __Data Visualization__
 
 ###### - Distribution of Age
 
  ![](https://www.kaggleusercontent.com/kf/50302331/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..61eQA5kAS7Vbn9dgdXA6Fg.owQrqdb6Qrfo8YL3sSlSFUnZJGj387fy-Dx2e7k3LNZg_3L4eLuodEz6tttCI3ySa7NL0G-AbhQn5A85X23o3imDuQ3UDt-OprWqsaaLZLM3rEJLvBQyVQsUKHOiCOYPT58dA8NpSXduTOviWCFhx_Bcstiv35JdLAbQdf5wzvem1v_t5IeWmq0PoRR7dC-yFq7YDIO-hayrpmFVd-_Dkvut-2qDF1QL1jRRJu_dL8KczHHM5L4zxBiZw38MrsikBQ07wB6yVl9fyr7o_p-Agn-6qwpT6UU_qepb4DPmsGS7HTkLecEasIpspz-deRTXSg2FEO5Iiis0ERfjdIlf6LzC4Ovy469iiu9j8rwRh6k8bRi9JGrQIsb_DEIE1DJc76PVPqxffVp3okPsNvxkeWJ92L4hrjkLjTXDox1HFHifn8eoKGEewfb6JePnX6u9bn1fxqlVwxfO3yuUwW5jWKGZ70nv1Rg5FuFaT-f0hBQMQOsI_KzVDpyXKGAAZFZn52PX7_xcDJyTb52tqiu4n9kygYb1rPNF32AbN1TvdkwF-b775VX9oJDGSoV3o-GZxtKLtHdH9HoMo9LUgeWkZlUktKrVOWpBgh70hxdHZGq-t7E9RNDOYqvLjSqrFwztDnHkQMBlQDOFstv7f0nUHn1S28UDRJWRxfOarA3mLWLfaz3ASoUnADcsbWEEBdwy.HarOxfOP0TerXCwJvDZ--g/__results___files/__results___35_0.png)
  
  ###### - Sex feature vs Survived feature
  
![](https://www.kaggleusercontent.com/kf/50302331/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..61eQA5kAS7Vbn9dgdXA6Fg.owQrqdb6Qrfo8YL3sSlSFUnZJGj387fy-Dx2e7k3LNZg_3L4eLuodEz6tttCI3ySa7NL0G-AbhQn5A85X23o3imDuQ3UDt-OprWqsaaLZLM3rEJLvBQyVQsUKHOiCOYPT58dA8NpSXduTOviWCFhx_Bcstiv35JdLAbQdf5wzvem1v_t5IeWmq0PoRR7dC-yFq7YDIO-hayrpmFVd-_Dkvut-2qDF1QL1jRRJu_dL8KczHHM5L4zxBiZw38MrsikBQ07wB6yVl9fyr7o_p-Agn-6qwpT6UU_qepb4DPmsGS7HTkLecEasIpspz-deRTXSg2FEO5Iiis0ERfjdIlf6LzC4Ovy469iiu9j8rwRh6k8bRi9JGrQIsb_DEIE1DJc76PVPqxffVp3okPsNvxkeWJ92L4hrjkLjTXDox1HFHifn8eoKGEewfb6JePnX6u9bn1fxqlVwxfO3yuUwW5jWKGZ70nv1Rg5FuFaT-f0hBQMQOsI_KzVDpyXKGAAZFZn52PX7_xcDJyTb52tqiu4n9kygYb1rPNF32AbN1TvdkwF-b775VX9oJDGSoV3o-GZxtKLtHdH9HoMo9LUgeWkZlUktKrVOWpBgh70hxdHZGq-t7E9RNDOYqvLjSqrFwztDnHkQMBlQDOFstv7f0nUHn1S28UDRJWRxfOarA3mLWLfaz3ASoUnADcsbWEEBdwy.HarOxfOP0TerXCwJvDZ--g/__results___files/__results___40_1.png)

  ###### - Embarked vs Survived

![](https://www.kaggleusercontent.com/kf/50302331/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..61eQA5kAS7Vbn9dgdXA6Fg.owQrqdb6Qrfo8YL3sSlSFUnZJGj387fy-Dx2e7k3LNZg_3L4eLuodEz6tttCI3ySa7NL0G-AbhQn5A85X23o3imDuQ3UDt-OprWqsaaLZLM3rEJLvBQyVQsUKHOiCOYPT58dA8NpSXduTOviWCFhx_Bcstiv35JdLAbQdf5wzvem1v_t5IeWmq0PoRR7dC-yFq7YDIO-hayrpmFVd-_Dkvut-2qDF1QL1jRRJu_dL8KczHHM5L4zxBiZw38MrsikBQ07wB6yVl9fyr7o_p-Agn-6qwpT6UU_qepb4DPmsGS7HTkLecEasIpspz-deRTXSg2FEO5Iiis0ERfjdIlf6LzC4Ovy469iiu9j8rwRh6k8bRi9JGrQIsb_DEIE1DJc76PVPqxffVp3okPsNvxkeWJ92L4hrjkLjTXDox1HFHifn8eoKGEewfb6JePnX6u9bn1fxqlVwxfO3yuUwW5jWKGZ70nv1Rg5FuFaT-f0hBQMQOsI_KzVDpyXKGAAZFZn52PX7_xcDJyTb52tqiu4n9kygYb1rPNF32AbN1TvdkwF-b775VX9oJDGSoV3o-GZxtKLtHdH9HoMo9LUgeWkZlUktKrVOWpBgh70hxdHZGq-t7E9RNDOYqvLjSqrFwztDnHkQMBlQDOFstv7f0nUHn1S28UDRJWRxfOarA3mLWLfaz3ASoUnADcsbWEEBdwy.HarOxfOP0TerXCwJvDZ--g/__results___files/__results___50_0.png)

 
# STEP 2: Feature Engineering 
 
- Feature Engineering is a process of transforming the data into data which is easier to interept and also, to increase the predictive power of learning             algorithm.

- In this part we will create a new features that could improve predictions such as if the passenger is alone or not,
  and combining existing features to produce a more useful one, and dropping the columns doesn't improve predictions.


# Step 3 : Pre-Modeling Tasks

- Separating the independant and the dependant variable.
- Splitting the training data.

# Step 4 : Modeling the Random Forest Model.

 - In this part we'll try to build a Random Forest Model and then tunning the hyperparameters using the GridSearcCV.
 
# Step 5 : Evaluating the performance of Random Forest using the performance metrics.


 - Evaluating the machine learning model is a crucial part in any data science project. There are many metrics that helps us to evaluate our model accuracy.

- __Classification Accuracy__
- __Classification Report__
- __Precision Score__
- __Recall Score__
- __Confusion matrix__

 ![](https://www.kaggleusercontent.com/kf/50303687/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..T-lvH9KJMdlxLaZkae0Odg.MPvHcfIuIccgOrbpS6222zyedTVz9_oqTG7tjalcXZ70RFMivUm1bsYt2vzNE9sWLfRnVd1Nmirtmqy527-vstPVezZ-BbyJYhoziN0TZE1Bz95GT1n00G8x-YNhRVHLc_bn0YOMuuQT22kFEjvZdJfsUGc9DJwRKMNw9cR9JPxMSilKzkML9xhCDN-jtqtgr5rhVsTq0vBp34vKiiywDfVMlXvEWazqGljW9fw0LWOq9b8oyB92VxZsyGSnOVjfSwhztiD2XTuV-fVzzfIDpR5sduih8i6cErUT0GeTB8fgL4egf_xsKjoL-hvu-klc19iCq0c7e3vFtRVCL5ikOtwAd9kIRuQ0imBiCGVwYp4UsmR8eDJjpHsuucrtXa_5y3xMfvTdSxVAnF5oM-5xXEQzd3I5SGRBBxLJBXep90viQNfITkf559E0UsTHGCxuTNcJs9KNIHpOUPX2ngDWVkhTwN61nL561SEcODrI8YLh-_k7plT5sRIaEE8IIgh4tAY0_qqz-xdWGUzggnDUMnT9s4To9UPF8mielAVAXGkVJC-2lKbe0z0m8YCrv0dLwBGYGxE5i81vY-RhioVQ-HEBh_iLHYp1NbgAYY8tYDIySJuh08aiKP0R9d0QhE8c9X0zXapHZ9iLsmhJGZ9A4Dxptu61qDeow42IaUE9TeIlfvA1SPeoqZKreL-nXhWs.0T4hLxOSamzJE5BuHmWDgg/__results___files/__results___102_1.png)
 
- __AUC & ROC Curve__

 ![](https://www.kaggleusercontent.com/kf/50303687/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..T-lvH9KJMdlxLaZkae0Odg.MPvHcfIuIccgOrbpS6222zyedTVz9_oqTG7tjalcXZ70RFMivUm1bsYt2vzNE9sWLfRnVd1Nmirtmqy527-vstPVezZ-BbyJYhoziN0TZE1Bz95GT1n00G8x-YNhRVHLc_bn0YOMuuQT22kFEjvZdJfsUGc9DJwRKMNw9cR9JPxMSilKzkML9xhCDN-jtqtgr5rhVsTq0vBp34vKiiywDfVMlXvEWazqGljW9fw0LWOq9b8oyB92VxZsyGSnOVjfSwhztiD2XTuV-fVzzfIDpR5sduih8i6cErUT0GeTB8fgL4egf_xsKjoL-hvu-klc19iCq0c7e3vFtRVCL5ikOtwAd9kIRuQ0imBiCGVwYp4UsmR8eDJjpHsuucrtXa_5y3xMfvTdSxVAnF5oM-5xXEQzd3I5SGRBBxLJBXep90viQNfITkf559E0UsTHGCxuTNcJs9KNIHpOUPX2ngDWVkhTwN61nL561SEcODrI8YLh-_k7plT5sRIaEE8IIgh4tAY0_qqz-xdWGUzggnDUMnT9s4To9UPF8mielAVAXGkVJC-2lKbe0z0m8YCrv0dLwBGYGxE5i81vY-RhioVQ-HEBh_iLHYp1NbgAYY8tYDIySJuh08aiKP0R9d0QhE8c9X0zXapHZ9iLsmhJGZ9A4Dxptu61qDeow42IaUE9TeIlfvA1SPeoqZKreL-nXhWs.0T4hLxOSamzJE5BuHmWDgg/__results___files/__results___112_0.png)



# Step 6 : Predictions and submit the results

 
 

# Useful Resources :

- [Fundamental Techniques of Feature Engineering for Machine Learning](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114).

- [Ensemble Machine Learning Algorithms in Python with scikit-learn](https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/).

- [ZacharyJWyman/ML-Techniques](https://github.com/ZacharyJWyman/ML-Techniques).

- [How to Build a Machine Learning Model](https://towardsdatascience.com/how-to-build-a-machine-learning-model-439ab8fb3fb1).

- [How to find optimal parameters using GridSearchCV?](https://www.dezyre.com/recipes/find-optimal-parameters-using-gridsearchcv).

