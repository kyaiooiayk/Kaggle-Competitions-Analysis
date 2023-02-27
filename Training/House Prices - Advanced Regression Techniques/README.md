# House Prices - Advanced Regression Techniques
*Predict sales prices and practice feature engineering, RFs, and gradient boosting*
***

Category? Getting Started

[Competition overview](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)
[Link to dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)


## A quick-and-dirty (and by no means complete) step-by-step approach to submit your solution to Kaggle:

- Build three different notebooks:
    - Are the train and test set representative of the same pool of data? *If not* which features show a net deviation?
    - Build a comprehensive EDA where the main goal is to understand and explain the data
    - Model the problem, train a model and predict the output using the providede test set.

The third one is then
- **Start-off:
    - Step #: Read-in the train and test data as provided by Kaggle.
    - Step #: Visualise the train and test data as a table.
    - Step #: Visualise your target variable and establish if it is normally distributed or not.
- **Explorat ory Design Analysis (EDA)**:
    - Step #: Get the correlation matrix even if this captures only linear relationship.
    - Step #: Change the type of some feature if necessary. It is important you do this before the other steps.
    - Step #: Drop outliers on the training set only. Do not touch the test set otherwise Kaggle will not accept your submission.
    - Step #: Merge train and test sets.
    - Step #: Data imputation.
    - Step #: Deal with skew features.
    - Step #: Create new feature via feature engineering.
    - Step #: Encode categorical features.
    - Step #: Recreate train and test data ready for the training part.
- **Training Part**:
    - Step #: Get a baseline model with zero feature engineering apart those necessary. The main goal here is to get something off the ground with a baseline model.
    - Step # - Spot check several methods. The main goal here is is to downselect some od the best performing models.
    - Step # -  Select one method and see how the model behaves as you add one features at the time. The goal is to see how the model responds to the addition of new features.
    - Step # - Add new features and see how the model responds
    - Step # - Optimise the most promising methods and see how much can you squeeze out of it.
***

## Notable notebooks (secret nuggets)
- [How I made top 0.3% on a Kaggle competition](https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition)
Comments: I found interesting to see how the author modified the final submission to take into account the outliers in the final submssion. In particulars, the author used the quantile to trim the outliers. I suspect this is just a trick to get up in the public leaderboard, rather than a result of some theoretical analysis.
- [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python/notebook#Last-but-not-the-least,-dummy-variables) There is a nice discussion on the homoscedasticity and how to use the log trasnform in the presence of feature that have entries equal to zero.
- [Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard#Modelling). Nice introduction on stacking and how to create your own stack method which can be interfaces in ScikitLearn CV methods.
- [XGBoost + Lasso](https://www.kaggle.com/humananalog/xgboost-lasso) Interesting discussion about outliers in the train and test sets. In particular it was important to note that outliers cannot be removed from the test set otehrwise you submission will be invalid, at least in the public board. On top of it, there is quite a lot of interesting extra features.
- [Regularized Linear Models](https://www.kaggle.com/apapiu/regularized-linear-models) Nice discussion on how to use Lasso for feature find out which features are important.
- [A study on Regression applied to the Ames dataset](https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset/notebook) Nice discussion on how to create polynomial of any degree. Also it shows how to combine feature together.
- [A Detailed Regression Guide with House-pricing](https://www.kaggle.com/masumrumi/a-detailed-regression-guide-with-house-pricing/notebook) A nice discussion about linear model and the concept ot homo- and Heteroscedasticity. No code is provided!
- [House Prices EDA](https://www.kaggle.com/dgawlik/house-prices-eda) Nice discussion about price degmentation and ANOVA.
- [Data Science Workflow TOP 2% (with Tuning)](https://www.kaggle.com/angqx95/data-science-workflow-top-2-with-tuning) Nice descriotion of ensembling methods.
- [Stacking House Prices - Walkthrough to Top 5%](https://www.kaggle.com/agodwinp/stacking-house-prices-walkthrough-to-top-5) Very nice discussion on EDA and feature engineering, probavly the best I've seen for this dataset.  
- [Feature Engineering for House Prices](https://www.kaggle.com/ryanholbrook/feature-engineering-for-house-prices) Interesting discussion on how to use PCA for fature engineering.
- [House Prices Tutorial with Catboost](https://www.kaggle.com/allunia/house-prices-tutorial-with-catboost) Interesting discussion about numerical and categorical features. Also there is an interesting discussion about features that are not present in the two datasets (train and test).
- [Top 10 (0.10943): stacking, MICE and brutal force](https://www.kaggle.com/agehsbarg/top-10-0-10943-stacking-mice-and-brutal-force) Interesting brute force trick to deal with edge cases.
- [House Prices || Useful Regression Techniques ](https://www.kaggle.com/janiobachmann/house-prices-useful-regression-techniques) Interesting analysis especially in form of data statistics.
- [Some lessons (3rd place in leaderboard)](https://www.kaggle.com/roee286/some-lessons-3rd-place-in-leaderboard) I was particularly interested in this comment (quoting) *"clean data according to what makes sense. Worry less about CV score if the change makes sense"*.
- [House Prices - EDA Stacking XGB LGB](https://www.kaggle.com/squarex/house-prices-eda-stacking-xgb-lgb) Nice EDA with some interesting way of using pandas markup.
- [# 3.2 House Prices](https://www.kaggle.com/dmkravtsov/3-2-house-prices/notebook) Interesting idea on how to divide the feature in smaller groups.
- [Beginners_Prediction_Top3%🌃💲](https://www.kaggle.com/marto24/beginners-prediction-top3) This notebooks uses the idea the test set was actually taken from another data set -> The Amesa dataset. By doing we bring the test set into the training set and leak these information. It is not advisable, but it is good to know.
- [House Prices - EDA Stacking XGB LGB](https://www.kaggle.com/squarex/house-prices-eda-stacking-xgb-lgb#Baseline-model) Very interesting analysis on how to compute the correlation for numerical, categorical and the mix of the two.
- [Techniques for Handling the Missing Data](https://www.kaggle.com/srivignesh/techniques-for-handling-the-missing-data) Comprehensive guide of the option one has to impute missing data.
- [House Price Prediction: Systematic EDA](https://www.kaggle.com/ar2017/house-price-prediction-systematic-eda) How a systematic EDA should be structured.
- [Object oriented programming for Data Science](https://www.kaggle.com/alaasedeeq/object-oriented-programming-for-data-science) A nice example how to keep your code clean and re-usable.
***
