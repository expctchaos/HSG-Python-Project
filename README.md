# HSG-Python-Project
Programming with Advanced Computer Languages Group Project


Short Description:

Use NHL players data set to build a model (e.g. multivariate regression or machine learning) in Python, based on which we predict the player's salary (as a proxy for the player's value). The analysis is  based on independent variables we choose from a data set retrieved from Kaggle. The prediction model provides the basis for a "scouting program" that lets users query values for player names. The goal is to help NHL scouts understand the effective value of players. oreover, the data-based determination of player values should enable scouts to identify under and overpaid players.


Program Requirements:

The program should let users query for a player name with the output being the actual salary they earn, the value of the player predicted by the model (as a salary value) and the % difference between the two (to indicate if the player is over/undervalued).

Hence, the program needs two parts:

- Reading Data / Predicting Values: Regression or machine learning model based on the data set that is fed into the program. The model should be able to capture complex interactions, account for non-linear relationships in the data set and not be prone to overfitting.

- User Interface to input Player Name: Simple UI with text input field where user enters the full name (first name + last name) of the player they want to analyze. For each query, the output interface should display player values (actual vs. predicted) and the difference.


User Instructions:

1) Download & install Anaconda Navigator

2) Start Anaconda Navigator & launch Jupyter Notebook

3) Open the Notebook Final Value Search.ipynb from the repository or copy the code to a new Notebook

4) Run the code to upload & read the data

5) Run the code for the scouting program (user interface)

6) in the pop-up window, enter a player name in the following format: FIRST NAME + LAST NAME

7) Submit & results should be displayed on-screen


In-Depth Project Description & Model Explanation:


Data processing:

To predict the value of the players we use data from Hockey Abstract. The consolidated dataset we downloaded from Kaggle (Owner: Cam Nugent).

Hockey Abstract: http://www.hockeyabstract.com

Kaggle (Cam Nugent): https://www.kaggle.com/datasets/camnugent/predict-nhl-player-salaries

The dataset contains 154 different statistics (including salary) for 874 NHL players during the 2016/2017 season. For the description of the different statistics listed in the dataset, please refer to the document "Dataset Column Legend".

In order to make the statistics comparable, 25 additional columns were added to the 154 existing ones. The values within the added columns are based on calculations performed on the existing data in the initial columns. For example, the absolute goals scored were extended by the goals scored per game played. For the creation of the value prediction model, 23 columns were determined, which have a direct connection to the specific player, influence the prediction of the player's value, and does not have a correlation of more than 0.7 with other selected columns. The wages were logrythmized and the dataset split into a training (75%) and a test dataset (25%) to optimize the prediction of the players' values.


Choice of Regression Model:

Aside linear regression models, potential choices include more sophisticated regressions like a support vector machine or a k-nearest neighbors model and machine learning models like single decision trees, neural networks or random forest models.

A random forest is an ensemble machine learning model that is made up of multiple decision trees. Each decision tree is created using a different subset of data and a different set of features. The predictions made by each decision tree are combined to form the final prediction of the random forest model.

Generally, both single decision trees and random forests are good choices for predicting non-linear relationships between features and the target variable (which I think is fair to assume we have in our data). Additionally, they can also handle a large number of features and can provide insights into which features are most important for predicting the target variable (suits our use case well).

Moreover, they are also less prone to overfitting than other models such as linear regression and hence likely to be a good overall choice in our case. It’s also simpler to implement than a neural network.

Finally, it seems to me that in our case a random forest approach may be a better choice than a single decision tree. This is because a random forest is an ensemble of decision trees, which allows the model to capture more complex interactions between the features and make more accurate predictions while being less prone to overfitting than a decision tree.

The main disadvantage of a random forest is that it is more computationally expensive than a decision tree, which might be a concern if we were working with a large dataset (which I don’t think we are).


Code Description:

The code uses the Random Forest Regressor method from the scikit-learn library to train a model that predicts player value based on performance statistics. To do this, the code first imports several libraries, including pandas for data manipulation and sklearn for machine learning algorithms. Next, the code loads a dataset containing information about players in the NHL, including their performance statistics.

The code then begins preparing the data for analysis by removing rows that contain missing data, adding new columns to the dataset, and cleaning the data to ensure that it is in a suitable format for analysis. For example, the code adds new columns that contain the number of goals and assists per game, the player's age, and their body mass index (BMI). The code also calculates several new metrics, such as the player's adjusted plus-minus and penalties per game, which are derived from the original data.

Once the data has been prepared, the code uses the Random Forest Regressor method to train a model that predicts player value. To do this, the code first splits the data into a training and testing set using the train_test_split method from the scikit-learn library. The training set is used to train the model, while the testing set is used to evaluate the performance of the model.

Next, the code trains a Random Forest Regressor model on the training set. This type of model is an ensemble method that uses multiple decision trees to make predictions. In this case, the model uses the player's performance statistics as input features and their estimated salary as the target variable. The model is trained using a process called bootstrapping, where multiple subsets of the training data are used to train multiple decision trees. The predictions made by these decision trees are then combined to make a final prediction.

Once the model has been trained, the code uses it to make predictions on the testing set. The predictions made by the model are then compared to the actual salaries of the players in the testing set to evaluate the model’s accuracy.

The metrics used in the code to evaluate the random forest model are mean absolute error root mean squared error, r squared and feature importance. These are all commonly used metrics for evaluating the performance of regression models, which make predictions on a continuous numerical scale.

Mean absolute error (MAE) is a measure of the average magnitude of the errors in the model's predictions. It is calculated by taking the absolute value of the difference between the predicted value and the actual value for each data point, and then taking the average of those differences. A smaller mean absolute error indicates that the model's predictions are closer to the true values.

Root mean squared error (RMSE) is another measure of the average magnitude of the errors in the model's predictions. It is calculated by taking the square of the difference between the predicted value and the actual value for each data point, taking the average of those squared differences, and then taking the square root of that average. Like MAE, a smaller RMSE indicates that the model's predictions are closer to the true values.

Both of these metrics are important because they help us understand how well the model is performing. In the context of this code, they can help us determine how accurately the random forest model is able to predict player salaries based on various factors in the dataset. A lower value for these metrics indicates that the model is making more accurate predictions, which can be useful for making decisions about player salaries.

R-squared is a measure of how well the model fits the data. It is calculated by taking the ratio of the sum of the squared differences between the predicted values and the mean of the actual values, to the sum of the squared differences between the actual values and the mean of the actual values. This ratio is then expressed as a percentage, with values ranging from 0 to 100. A higher R-squared value indicates that the model explains more of the variation in the data, and is therefore a better fit.

Feature importance is a measure of how much each feature in the dataset contributes to the predictions made by the model. In the context of a random forest model, feature importance is calculated by looking at how much each feature reduces the impurity in the model. The impurity of a node in a decision tree is a measure of how mixed the data at that node is. For example, a node that contains only data points from a single class has zero impurity, while a node that contains an equal number of data points from each class has maximum impurity. By looking at how much each feature reduces the impurity in the model, we can determine which features are most important for making accurate predictions.

Both R-squared and feature importance are important because they help us understand the performance of the model and the relative importance of each feature in the dataset. In the context of this code, they can help us determine how well the random forest model is able to explain the variation in player salaries, and which factors are most important for predicting those salaries. This information can be useful for making decisions about player salaries and improving the performance of the model.

Model Performance / Result Interpretation:

The performance statistics indicate that the random forest model has a relatively high level of accuracy in predicting player salaries. The RMSE of 0.5253143777537517 and MAE of 0.4087 indicate that the model's predictions are close to the true values, on average. The R-squared value of 0.6407 indicates that the model explains about 64% of the variation in the data. This is a relatively high R-squared value, which indicates that the model is a good fit for the data.

The feature importance values show that some factors are more important for predicting player salaries than others. For example, the "A/GP" feature, which represents the number of assists per game, has a relatively high feature importance of 0.246306. This indicates that this factor has a significant impact on player salaries. On the other hand, the "Hand_R" feature, which represents whether a player is right-handed, has a relatively low feature importance of 0.002571. This indicates that this factor has a relatively small impact on player salaries.

Overall, the performance statistics indicate that the random forest model is performing well in predicting player salaries based on the factors in the dataset. The model has a relatively high level of accuracy and a good fit for the data, and the feature importance values indicate which factors are most important for predicting player salaries.

