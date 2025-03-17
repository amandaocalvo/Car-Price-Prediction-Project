# BEHIND THE WHEEL 
# AI Model for Used Car Price Prediction 

## Introduction

### Objective

The goal of this project is to develop an AI model capable of predicting used car prices based on key influencing factors. Pricing used cars accurately is a complex challenge due to varying attributes such as make, model, year, mileage, condition, and location. Our approach involves training a model on Armenian used car data and evaluating its adaptability for the Canadian market.

### Business Problem

How do key factors such as production year, mileage, vehicle model, condition, and engine volume influence used car prices, and how can these relationships be leveraged to accurately predict market values over different countries?

In today’s competitive automotive market, determining the fair market value of used vehicles can be a challenge and it's an important factor for buyers and sellers. The pricing of used cars depends on a variety of factors, such as the car’s make, model, age, mileage, condition, and regional economic conditions, and this can create a complex environment for price estimation.

This project aims to develop an AI-driven model capable of predicting used car prices with greater accuracy. The problem originates from a Kaggle competition, utilizing a dataset sourced from auto.am, an Armenian used car marketplace.

Beyond price prediction, the AI model seeks to identify key factors that drive price fluctuations, whether increasing or decreasing a vehicle's value. Also, the project will evaluate the model’s adaptability to the Canadian used car market, assessing whether the same predictive approach can be effectively applied across different regions.

## Data Information

### Data Sources

There are two data sources for this project:

#### Armenian Dataset
- **Source**: 1,642 records from auto.am, an Armenian used car website. (Kaggle dataset)
- **Models**: Toyota, Mercedes-Benz, Kia, Nissan, Hyundai
- **Motor Types**: Petrol, Gas, Petrol & Gas, Diesel, Hybrid
- **Currency**: All prices are converted to US dollars
- **Status**: Sedan, SUV, Universal, Coupe, Pickup, Hatchback, Minivan/Minibus
- **Condition Categories**: Excellent, Good, Crashed, Normal, New
- **Data Quality**: No null values, but outliers are present and will be removed during data cleaning.

#### Canadian Dataset
- **Source**: 9 columns, 18 rows from autotrader.ca, a Canadian used car website.
- **Currency**: All prices are converted to US dollars
- **Car Types**: Sedan, SUV, Coupe, Pickup
- **Status**: Excellent, Good
- **Models**: Toyota, Mercedes-Benz, Kia, Nissan, Hyundai
- **Motor Types**: Gas, Diesel
- **Data Quality**: No null values.

### Data Cleaning

The Kaggle dataset contained a significant number of outliers, which were removed using the Interquartile Range (IQR) method. The lower and upper bounds were calculated as follows:
- **Lower Bound**: `Q1 - 1.5 * IQR`
- **Upper Bound**: `Q3 + 1.5 * IQR`

This preprocessing step improved the accuracy and reliability of the machine learning model.

The running column contained mixed units (kilometers and miles), so a function was implemented to convert miles to kilometers for consistency.

The wheel column had only one unique value (left), providing no variance, so it was removed to streamline the dataset.

## AI Model Selection

The **Random Forest** model will be used to address the problem, and we will implement it using Python. This model is a popular choice for regression problems like used car price prediction because it has several key advantages, such as:

- **Capturing Complex Relationships**: Used car pricing is influenced by many factors with non-linear relationships. The Random Forest model can combine multiple decision trees, helping to capture these complex, non-linear interactions without requiring extensive feature engineering.
- **Handling Noise and Outliers**: Random Forests average the predictions of many trees, making them less sensitive to noise and outliers compared to single decision trees. This is particularly beneficial when dealing with real-world datasets.
- **Reducing Overfitting**: This approach helps reduce overfitting by averaging the results of many trees, each trained on different subsets of the data. This is especially important when adapting the model to different markets (e.g., from Armenia to Canada).
- **Feature Importance**: Random Forests provide insights into feature importance, helping us understand which variables most influence car prices.
- **Minimal Data Preprocessing**: Random Forests are less affected by the scale or distribution of features, meaning we do not need to perform normalization or transformation, unlike with some other algorithms.

## AI Model Training and Validation

### Handling Categorical Data

The Random Forest model does not natively process categorical data; one-hot encoding was applied to convert categorical variables (model, motor type, color, type, and status) into numerical format.

### Splitting Data for Training and Testing

The dataset was divided into features (independent variables) and the target (price). This separation ensures that the model learns from the correct inputs.

### Cross-Validation for Performance Evaluation

5-Fold Cross-Validation was used to improve model reliability. The dataset was split into five equal parts:
- Four folds used for training.
- One fold used for validation, rotating through all folds.

This approach reduces overfitting and provides a more robust performance estimate.

### Model Training with Optimized Parameters

The model was trained using Random Forest Regressor with:
- `random_state = 2025`, which ensures reproducibility.
- `max_depth = 10`, which prevents overfitting by limiting tree depth.

This configuration provided better accuracy with reduced overfitting.

## AI Model Performance

To assess the effectiveness of our model, we used two key performance metrics:

### Mean Absolute Error (MAE)

MAE calculates the average absolute difference between the predicted and actual values. Expressed in the same units as the target variable (price), making it easy to interpret. Lower MAE values indicate better model accuracy, as smaller deviations from the actual prices reflect stronger predictive performance.

### R² Score (Coefficient of Determination)

R² measures the proportion of variance in the dependent variable (price) that is explained by the independent variables. The scale ranges from 0 to 1, where:
- **R² close to 1**: The model captures most of the variability in price.
- **R² close to 0**: The model fails to explain the data pattern effectively.

A higher R² suggests a stronger relationship between the input features and price.

### Model Performance Results

The results are shown below. Considering that our target value has a mean of 15,982, a mean absolute error (MAE) of 1,731 indicates that the model is performing well. Also, an average test R² score of 0.78 suggests that the model effectively captures much of the underlying pattern.

## Feature Selection

One of the advantages of using the Random Forest model is that it allows for feature importance selection. We analyzed the most important features for our final model and selected the top 20 that had the greatest impact. We tested the model with fewer features, but the results would not be better.

### Top Features:
- year
- running_km
- model_mercedes-benz
- model_toyota
- model_kia
- model_hyundai
- model_nissan
- status_crashed
- motor_volume
- type_sedan
- status_excellent
- color_black
- color_white
- color_silver
- type_Universal
- color_blue
- status_normal
- color_cherry
- motor_type_petrol and gas
- status_good

## Testing on Canadian Dataset

The same processes of data cleaning, hot-encoding, and feature selection were applied to the Canadian dataset. It is possible to see that there is a high error percentage when predicting Canadian car prices using the same machine learning model trained with the Armenian dataset.

The poor performance in the testing dataset also shows that the model fails to fit Canadian data into its parameters.

## Key Insights

To better answer the business question, we performed the SHAP (SHapley Additive exPlanations) plot, which is a powerful visualization that helps explain the output of the model by showing the contribution of each feature to the final prediction.

From our analysis, it is evident that:
- Newer cars increase the predicted price.
- Being a Mercedes-Benz is associated with higher car prices.
- More kilometers driven leads to a lower price.
- Damaged cars lower the predicted price.

## Limitations

Our project faces several limitations, particularly due to the restricted set of features available in our dataset. The data includes key variables such as car model, production year, motor type, color, vehicle type, status, motor volume, and running kilometers. While these features capture essential aspects of a vehicle’s value, many other factors can influence market price.

### Other Factors:
- **Transmission Type**: Whether the car is automatic or manual.
- **Advanced Features**: Information about electronic devices, infotainment systems, and safety technologies.
- **Fuel Economy**: Efficiency ratings that impact vehicle desirability.
- **Service History**: Records of maintenance, repairs, and past accidents.

Incorporating these variables could enhance the predictive accuracy of the model. Also, applying a model developed for one market to another presents challenges. Our dataset is sourced from an Armenian used car platform, but market dynamics in Armenia differ significantly from those in Canada. Key differences include:

### Market Differences:
- **Consumer Preferences**: Buyers in different countries prioritize different features and attributes, affecting how certain factors influence pricing.
- **Economic Factors**: Currency differences, tax structures, import duties, and overall economic conditions all impact car prices.
- **Market Conditions**: Regional supply and demand, as well as cultural factors, are not always reflected in datasets from different markets.

## Recommendations

To make the car price prediction model more scalable and accurate, we recommend the following improvements:

### Enhancing Data Collection
- Expand the feature set by incorporating additional relevant attributes such as transmission type, technology features, fuel economy ratings, and accident records.
- Gather more data specific to the target market (e.g., Canadian data) to better capture local consumer preferences and regional pricing trends.

### Improving Machine Learning Techniques
- Explore alternative machine learning models, such as Gradient Boosting Machines (GBM), XGBoost, and LightGBM for enhanced performance.
- Use ensemble methods to compare model performance and reduce overfitting.
- Apply hyperparameter tuning using techniques like Grid Search or Bayesian Optimization to fine-tune model parameters for optimal accuracy.

