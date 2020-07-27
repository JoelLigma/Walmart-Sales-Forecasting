# Predicting Walmart Store Sales with Python

Date: July 2020 - July 2020

**Context**

The data provided for this project contains historical sales data for 45 Walmart stores located in different regions of the US. Each store contains a number of departments, and the goal for this project is to predict the department-wide sales for each store.

In addition, Walmart runs several promotional markdown events throughout the year. These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks.

For convenience, the four holidays fall within the following weeks in the dataset (not all holidays are in the data):

- Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
- Labor Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
- Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
- Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13

Dataset Link: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/overview/description

## Table of Contents
**1) Data Preprocessing and Merging Multiple Datasets** 
- Data Cleaning
- Dealing with Missing Values through Imputation
- Merging
<br>

**2) Exploratory Data Analysis (EDA)**
- Summary Statistics
- Univariate Analysis
- Bivariate Analysis
<br>

**3) Data Preparation**
- Dealing with Outliers
- Feature Correlation
- Dummy Coding
- Identifying and Removing NZV/ ZV Predictors
- Training, Validation and Test Split
- Scaling Data
<br>

**4) Model Building & Validation**
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Model Comparison
<br>

**5) Saving the Model**
- Saving the Best Performing Model for Future Deployment

# Results

**Model Evaluation Explanation**

Due to limited computational resources we will simply go with the default model results. Ideally, we would go back and tune the model hyperparameters to further improve the model performance by reducing the prediction error.

To compare the 3 models, we will use adjusted R2 and RMSE metrics.

**Why Adj R2?**

Adjusted Rsquared = 1 - (1 - R2) * ((n - 1) / (n - p - 1))

where p is the total number of explanatory variables in the model (not including the constant term), and n is the sample size. It can also be written as:

Why is adjusted Rsquared used? The adjusted R-squared compensates for the addition of variables and only increases if the new predictor enhances the model above what would be obtained by probability.

R-square test is used to determine the goodness of fit in regression analysis. Goodness of fit implies how better regression model is fitted to the data points. More is the value of r-square near to 1, better is the model. But the problem lies in the fact that the value of r-square always increases as new variables(attributes) are added to the model, no matter that the newly added attributes have a positive impact on the model or not. also, it can lead to over fitting of the model if there are large no. of variables.

Adjusted r-square is a modified form of r-square whose value increases if new predictors tend to improve model’s performance and decreases if new predictors does not improve performance as expected.

Source= https://www.geeksforgeeks.org/ml-adjusted-r-square-in-regression-analysis/

**Why RMSE?**

Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit.

We will use RMSE because this metric has some advantages over MAE and MSE. The MAE fails to punish large prediction errors and might therefore be misleading. MSE is a way to punish these outlier prediction errors by squaring each error and then taking the mean, however, it is difficult to compare this metric with the actual outcome value since its unit is squared. That's when RMSE comes into play. It entails simply taking the square root of the MSE metric which means it still punishes outlier but is now in the same units as the outcome value which makes it easy to evaluate.

The lower the RMSE value, the better the fit.

Source: https://www.google.com/search?q=rmse&rlz=1C1CHBF_enUS902US902&oq=RMSE&aqs=chrome.0.0l8.575j0j4&sourceid=chrome&ie=UTF-8

**Model Results**

Linear Regression
- RMSE: 8810.42
- adj R2: 0.6689

Random Forest
- RMSE: 2219.03
- adj R2: 0.9789

XGBoost
- RMSE: 4005.47
- adj R2: 0.9315

We can clearly see that Linear Regression did not perform well on our data. Presumably, because this model fails at finding relationships that are non-linear in nature. Our dataset probably has these types of relationships. On the other side, the Random Forest model performed very well. It showed an adj R2 of 0.9789 which means the model can explain around 97% of the variance in the data. Further, it displays an RMSE of 2219.03, which is the lowest error among the 3 models. Therefore, the Random Forest model is selected as best performing model.

# Next Steps - Ways to potentially improve the model(s)

- PCA to reduce dimensionality
- Re-assess feature selection
- Hyperparameter tuning
- Create a function to measure time while training RF (It takes quite long)
- Try AutoRegressive Integrated Moving Average model
