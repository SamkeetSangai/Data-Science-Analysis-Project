#------------------------------------------------------------
# Importing all libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import style
from keras.models import Sequential
from keras.layers import Dense, LSTM
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense, Dropout
#------------------------------------------------------------
print("\n")
print("="*50)
print("Data Science Analysis Project ", "\n")
#------------------------------------------------------------
# Supressing Pandas Future Warning
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
#------------------------------------------------------------
# Reading the DataSet
df = pd.read_csv("E:\Data Science Analysis\Project\Data\AEP_hourly.csv")
# Printing the first five row
print("="*50)
print("First five rows ", "\n")
print(df.head(5),"\n")
# Printing the last five rows
print("="*50)
print("Last five rows ", "\n")
print(df.tail(5), "\n")
# Printing the information about Dataset
print("="*50)
print("Information about Dataset", "\n")
print(df.info(), "\n")
# Describing the Dataset
print("="*50)
print("Describing the Dataset ","\n")
print(df.describe(), "\n")
# Printing the null Values
print("="*50)
print("Null Values ", "\n")
print(df.isnull().sum(), "\n")
#------------------------------------------------------------
# Extracting all Data like year, month, day and time etc
dataset = df
dataset["Year"] = pd.to_datetime(df["Datetime"]).dt.year
dataset["Date"] = pd.to_datetime(df["Datetime"]).dt.date
dataset["Time"] = pd.to_datetime(df["Datetime"]).dt.time
#------------------------------------------------------------
# When was the highest and lowest energy consumption and in which year
# Printing the highest energy consumption
print("="*50)
print("Highest energy consumption in MW ", "\n")
print(dataset[dataset["AEP_MW"] == dataset["AEP_MW"].max()], "\n")
# Printing the lowest energy consumption
print("="*50)
print("Lowest energy consumption in MW ", "\n")
print(dataset[dataset["AEP_MW"] == df["AEP_MW"].min()], "\n")
# Conclusion
print("-> Conclusion : We can say that maximum energy was consumed during 2008-
10-20 at 14:00:00 and it was 25695.0 MW and minimum energy was consumed during 
2016-10-02 at 05:00:00 and it was 9581.0 MW ", "\n")
#------------------------------------------------------------
# Checking how many years are unique
print("="*50)
print("Number of unique years ", "\n")
print(dataset["Year"].unique(), "\n")
print("-> Conclusion : There are 15 unique years from 2004 to 2018 ", "\n")
# Style
style.use('ggplot')
#------------------------------------------------------------
# Energy Distribution in MW
ax = sns.distplot(dataset["AEP_MW"], kde_kws={"color": 'green', "lw": 3})
ax.set(xlabel = 'Energy consumption in MW', ylabel = 'Density')
plt.title('Energy Distribution in MW')
plt.grid(True)
plt.show()
#------------------------------------------------------------
# Energy consumption according to Year
ax = sns.lineplot(x = dataset["Year"], y = dataset["AEP_MW"], data = dataset)
ax.set(xlabel = 'Year', ylabel = 'Energy consumption in MW')
plt.title('Energy consumption according to Year')
plt.grid(True)
plt.show()
#------------------------------------------------------------
# Regression
# Logistic regression
ax = sns.jointplot(x = dataset["Year"], y = dataset["AEP_MW"], data = dataset, 
kind = "reg")
# label
plt.title('Logistic regression (Year Vs Energy consumption)')
plt.show()
# Average energy consumption in MW
ax = sns.jointplot(x = dataset["Year"], y = dataset["AEP_MW"], data = dataset, 
kind = "kde", marginal_kws = {"lw":3, "color":'blue'})
# label
plt.title('Average energy consumption in MW')
plt.show()
# Energy consumption in MW vs Time
fig = plt.figure()
ax1 = fig.add_subplot(111)
sns.lineplot(x = dataset["Time"].astype(str),y = dataset["AEP_MW"], data = df, 
color = 'blue')
plt.title('Energy Consumption vs Time')
plt.xlabel('Time')
plt.ylabel('Energy Consumption in MW')
plt.grid(True)
for label in ax1.xaxis.get_ticklabels():
label.set_rotation(90)
plt.show()
# Dataset indexing
dataset = df.set_index('Datetime')
dataset.index = pd.to_datetime(dataset.index)
#------------------------------------------------------------
# Plotting the Energy consumption
fig = plt.figure()
# Subplot location
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
# 2004
y_2004 = dataset["2004"]["AEP_MW"].to_list()
x_2004 = dataset["2004"]["Date"].to_list()
ax1.plot(x_2004, y_2004, color = "red", linewidth = 1, label = '2004')
ax1.legend(loc="upper right")
ax1.grid(True)
# 2005
y_2005 = dataset["2005"]["AEP_MW"].to_list()
x_2005 = dataset["2005"]["Date"].to_list()
ax2.plot(x_2005, y_2005, color = "blue", linewidth = 1, label = '2005')
ax2.legend(loc="upper right")
ax2.grid(True)
# 2006
y_2006 = dataset["2006"]["AEP_MW"].to_list()
x_2006 = dataset["2006"]["Date"].to_list()
ax3.plot(x_2006, y_2006, color = 'green', linewidth = 1, label = '2006')
ax3.legend(loc = "upper right")
ax3.grid(True)
# Plot
plt.rcParams["figure.figsize"] = (18,8)
plt.rcParams["figure.autolayout"] = True
plt.suptitle('Energy consumption of 2004-06 in MW')
plt.xlabel('Date')
ax2.set_ylabel('Energy consumption in MW')
plt.show()
#------------------------------------------------------------
# Resampling Data
NewDataSet = dataset.resample('D').mean()
# Printing the Old Dataset
print("="*50)
print("Old Dataset Shape ", dataset.shape, "\n")
# Printing the New Dataset
print("="*50)
print("New Dataset Shape ", NewDataSet.shape, "\n")
# Testing and Training Dataset
TestData = NewDataSet.tail(100)
Training_Set = NewDataSet.iloc[:,0:1]
Training_Set = Training_Set[:-60]
# Printing the Testing Dataset Shape
print("="*50)
print("Test Dataset Shape ", TestData.shape, "\n")
# Printing the Training Dataset Shape
print("="*50)
print("Training Dataset Shape ", Training_Set.shape, "\n")
# Training Dataset 
Training_Set = Training_Set
sc = MinMaxScaler(feature_range = (0,1))
Train = sc.fit_transform(Training_Set)
X_Train = []
Y_Train = []
# Range should be fromm 60 Values to END 
for i in range(60, Train.shape[0]):
# X Train 0-59 
X_Train.append(Train[i-60:i])
# Y Would be 60 th Value based on past 60 Values 
Y_Train.append(Train[i])
# Convert into Numpy Array
X_Train = np.array(X_Train)
Y_Train = np.array(Y_Train)
# Printing the X Train Shape
print("="*50)
print("X Train Shape ", X_Train.shape, "\n")
# Printing the Y Train Shape
print("="*50)
print("Y Train Shape ", Y_Train.shape, "\n")
# Converting into 3-d Vector
X_Train = np.reshape(X_Train, newshape = (X_Train.shape[0], X_Train.shape[1], 1))
#------------------------------------------------------------
# Model
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = 
(X_Train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_Train, Y_Train, epochs = 50, batch_size = 32)
Df_Total = pd.concat((NewDataSet[["AEP_MW"]], TestData[["AEP_MW"]]), axis=0)
inputs = Df_Total[len(Df_Total) - len(TestData) - 60:].values
inputs = Df_Total[len(Df_Total) - len(TestData) - 60:].values
# Reshaping the input
inputs = inputs.reshape(-1,1)
# Normalizing the Dataset
inputs = sc.transform(inputs)
# X Test array appending
X_test = []
for i in range(60, 160):
X_test.append(inputs[i-60:i])
# Converting into Numpy array
X_test = np.array(X_test)
# Reshaping before passing to network
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Passing to Model 
predicted_stock_price = regressor.predict(X_test)
# Inverse Transformation 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
True_MegaWatt = TestData["AEP_MW"].to_list()
Predicted_MegaWatt = predicted_stock_price
dates = TestData.index.to_list()
Machine_Df = pd.DataFrame(data = {"Date": dates, "TrueMegaWatt": True_MegaWatt, 
"PredictedMeagWatt":[x[0] for x in Predicted_MegaWatt]})
#------------------------------------------------------------
# Plotting the Energy consumption
True_MegaWatt = TestData["AEP_MW"].to_list()
Predicted_MegaWatt = [x[0] for x in Predicted_MegaWatt ]
dates = TestData.index.to_list()
fig = plt.figure()
# Variables
x = dates
yt = True_MegaWatt
yp = Predicted_MegaWatt
# Plot
plt.plot(x, yt, color = 'blue', label = 'Actual energy consumption in MW')
plt.plot(x, yp, color = 'green', label = 'Predicted energy consumption in MW')
plt.gcf().autofmt_xdate()
plt.xlabel('Dates')
plt.ylabel('Energy consumption in MW')
plt.title('Predicting future energy consumption')
plt.legend()
plt.show()
