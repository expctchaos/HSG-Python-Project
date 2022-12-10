# NHL - Predicting the value of players (salary)

##Import data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor

#import data
nhl = pd.read_excel("NHL data.xlsx")
nhl.info()

##Prepare data

#Eliminate rows in dataset with missing data
nhl=nhl.dropna()
nhl.info()

#add Goals per Game
nhl["G/GP"] = nhl["G"]/nhl["GP"]

#add Assist per Game
nhl["A/GP"] = nhl["A"]/nhl["GP"]

#add the year the players were born
nhl['BornInt'] = nhl['Born'].str.replace('\W', '', regex=True)
nhl=nhl.dropna()
nhl = nhl.astype({'BornInt':'int'})
conditions = [
    (nhl['BornInt'] < 10000),
    (nhl['BornInt'] >= 10000) & (nhl['BornInt'] < 100000),
    (nhl['BornInt'] >= 100000)]
choices = [2000,nhl['BornInt']//10000+2000, nhl['BornInt']//10000+1900]
nhl['YearBorn'] = np.select(conditions, choices)
del nhl['BornInt']

#add players age
nhl["age"] = 2022 - nhl['YearBorn']

#add BMI (Body Mass Index)
nhl = nhl.astype({'Ht':'int'})
nhl = nhl.astype({'Wt':'int'})
nhl["WtInKg"] = nhl["Wt"]*0.453592
nhl["HtInCm"] = nhl["Ht"]*2.54
nhl["HtInCm^2"] = nhl["HtInCm"]*nhl["HtInCm"]
nhl["BMI"] = nhl["Wt"]*nhl["HtInCm^2"]
del nhl['WtInKg']
del nhl['HtInCm']
del nhl['HtInCm^2']

#adding adjusted +/-
nhl["Adjusted +/-"] = nhl["+/-"]-nhl["E+/-"]

#adding penatlties (in minutes) per game
nhl["PIM/GP"] = nhl["PIM"]/nhl["GP"]

#adding unblocked shot attempts (Fenwick, USAT) taken by this individual per game
nhl["iFF/GP"] = nhl["iFF"]/nhl["GP"]

#adding shots on goal taken by this individual per game
nhl["iSF/GP"] = nhl["iSF"]/nhl["GP"]

#adding expected goals (weighted shots) for this individual per game
nhl["ixG/GP"] = nhl["ixG"]/nhl["GP"]

#adding all scoring chances taken by this individual per game
nhl["iSCF/GP"] = nhl["iSCF"]/nhl["GP"]

#adding an estimate of the player's setup passes (passes that result in a shot attempt) per game
nhl["Pass/GP"] = nhl["Pass"]/nhl["GP"]

#adding hits thrown by this individual per game
nhl["iHF/GP"] = nhl["iHF"]/nhl["GP"]

#adding hits taken by this individual per game
nhl["iHA/GP"] = nhl["iHA"]/nhl["GP"]

#adding individual shots taken that missed the net per game
nhl["iMiss/GP"] = nhl["iMiss"]/nhl["GP"]

#adding giveaways by this individual per game
nhl["iGVA/GP"] = nhl["iGVA"]/nhl["GP"]

#adding takeaways by this individual per game
nhl["iTKA/GP"] = nhl["iTKA"]/nhl["GP"]

#adding shots blocked by this individual per game
nhl["iBLK/GP"] = nhl["iBLK"]/nhl["GP"]

#adding penalties drawn by this individual per game
nhl["iPEND/GP"] = nhl["iPEND"]/nhl["GP"]

#adding penalties taken by this individual per game
nhl["iPENT/GP"] = nhl["iPENT"]/nhl["GP"]

#adding the team's scoring chances while this player was on the ice per game
nhl["SCF/GP"] = nhl["SCF"]/nhl["GP"]

#adding scoring chances allowed while this player was on the ice per game
nhl["SCA/GP"] = nhl["SCA"]/nhl["GP"]

#adding the team's goals while this player was on the ice
nhl["GF/GP"] = nhl["GF"]/nhl["GP"]

#adding goals allowed while this player was on the ice
nhl["GA/GP"] = nhl["GA"]/nhl["GP"]

#overview of the dataframe
nhl.info

##Prepare data for predicting the value of players

#selecting values of the dataset relevant to the prediction for the player's value
nhlSalaryPrediction = nhl[['Salary', 'age', 'BMI', 'Hand', 'Position', 'GP', 'G/GP', 'A/GP', '+/-', 'Adjusted +/-', 'PIM/GP', 'SH%', 'SV%', 'iFF/GP', 'iSF/GP', 'ixG/GP', 'iSCF/GP', 'Pass/GP', 'iHF/GP', 'iHA/GP', 'iMiss/GP', 'iGVA/GP', 'iTKA/GP', 'iBLK/GP', 'BLK%', 'FO%', '%FOT', 'iPENT/GP', 'iPEND/GP', 'SCF/GP', 'SCA/GP', 'GF/GP', 'GA/GP']]

#ckecking correlation between the selected stats for the valuation of the players
nhlSalaryPrediction.corr()

#checking wich stats have a correlation >0.7 (risk to distort the prediction model)
c = nhlSalaryPrediction.corr().abs()
s = c.unstack()
s = s[s<1]
s = s[s>0.7]
s.sort_values().drop_duplicates()

#Eliminate too strongly correlated stats to optimize the prediction model
del nhlSalaryPrediction['SCF/GP']
del nhlSalaryPrediction['SCA/GP']
del nhlSalaryPrediction['Adjusted +/-']
del nhlSalaryPrediction['iSF/GP']
del nhlSalaryPrediction['Pass/GP']
del nhlSalaryPrediction['GF/GP']
del nhlSalaryPrediction['ixG/GP']
del nhlSalaryPrediction['iBLK/GP']
del nhlSalaryPrediction['iPENT/GP']
del nhlSalaryPrediction['iFF/GP']

#Convert "Hand" and "Position" to dummy variable
nhlSalaryPrediction = pd.get_dummies(nhlSalaryPrediction, columns=['Hand'], drop_first=True)
nhlSalaryPrediction = pd.get_dummies(nhlSalaryPrediction, columns=['Position'], drop_first=True)

##RANDOM FOREST model (on log salary)

#Create a variable for the logarithmized salary
nhlSalaryPrediction_log = nhlSalaryPrediction.copy()
nhlSalaryPrediction_log["logSalary"] = np.log(nhlSalaryPrediction["Salary"])
del nhlSalaryPrediction_log["Salary"]

#splitting dataset to training and test dataset
nhl_train_log = nhlSalaryPrediction_log.loc[0:instance,:]
nhl_test_log = nhlSalaryPrediction_log.loc[instance+1:,:]

#checking training dataset
nhl_train_log

#Random forest model to predict the salary (with logarithm of salary)
y = nhl_train_log["logSalary"]
X = nhl_train_log.loc[:, nhl_train_log.columns != 'logSalary']

# Create an instance of the RandomForestRegressor class
model = RandomForestRegressor(n_estimators=100, max_depth=5)

# Fit the model to the training data
model.fit(X, y)

#Predicting the salary for the test dataset
y_test_log = nhl_test_log["logSalary"]
X_test_log = nhl_test_log.loc[:, nhl_test_log.columns != 'logSalary']

# Generate predictions on the test data
predictions_logSalary = model.predict(X_test_log)
predictions_Salary_with_log = np.exp(predictions_logSalary)
predictions_Salary_with_log

#calculate RMSE
rmse = sqrt(mean_squared_error(y_test, predictions_Salary_with_log))

#Creating an Actual vs. Predicted Plot
def actual_vs_predicted_plot(y_test, prediction_Salary_test):
  min_value=np.array([y_test.min(), predictions_Salary_with_log.min()]).min()
  max_value=np.array([y_test.max(), predictions_Salary_with_log.max()]).max()
  fig, ax = plt.subplots(figsize=(10,5))
  ax.scatter(y_test,predictions_Salary_with_log, color="blue")
  ax.plot([min_value,max_value], [min_value, max_value], lw=4, color="green")
  ax.set_title("Actual vs Predicted Plot")
  ax.set_xlabel('Actual')
  ax.set_ylabel('Predicted')
  plt.show()

#Displaying Actual vs. Predicted Plot
actual_vs_predicted_plot(y_test, predictions_Salary_with_log)

##Determine the effective values of the players (wages they should receive based on their statistics)

#Creating variable for logarithm of salary
nhlFinalSalaryPrediction = nhlSalaryPrediction.copy()
nhlFinalSalaryPrediction["logSalary"] = np.log(nhlFinalSalaryPrediction["Salary"])

#checking dataset
nhlFinalSalaryPrediction.head()

#Predicting the salary for the entire dataset / for all players (Linear regression with logarythmic salary)

#creating varibale for logarithm of salary
nhlFinalSalaryPrediction = nhlSalaryPrediction.copy()
nhlFinalSalaryPrediction["logSalary"] = np.log(nhlFinalSalaryPrediction["Salary"])
del nhlFinalSalaryPrediction["Salary"]

#Predicting salary (value of players)
y = nhlFinalSalaryPrediction["logSalary"]
X = nhlFinalSalaryPrediction.loc[:, nhlFinalSalaryPrediction.columns != 'logSalary']
X = sm.add_constant(X, prepend=False)
predictions_Salary = model_Salary_log.predict(X)
predictions_Salary_final = np.exp(predictions_Salary)
predictions_Salary_final

#adding Value of Players (predicted Salaries based on their stats) and the difference in valuation (overpaid or underpaid) to the main dataset
nhlFinal=nhl.copy()
nhlFinal["Player Value"] = predictions_Salary_final
nhlFinal["Difference in valuation"] = nhlFinal["Player Value"] - nhlFinal["Salary"]

#Overview of final dataset
nhlFinal.head()
