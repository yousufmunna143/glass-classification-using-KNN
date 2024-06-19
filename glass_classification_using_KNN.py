"""
Problem Statement: A glass manufacturing plant uses different earth elements to design new glass materials 
based on customer requirements. For that, they would like to automate the process of classification as it’s 
a tedious job to manually classify them. Help the company achieve its objective by correctly classifying 
the glass type based on the other features using KNN algorithm
"""
# dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
import pylab
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score


# 1a. Business Understanding
# Objectives:
# -> To automate the process of classification of glass types to minimize time and manual effort

# 1b. Data Understanding
# 'glass.csv' is the our required data set (214 rows * 10 columns)
# We have 10 columns in our data set. They are,
# RI: refractive index
# Na: Sodium 
# Mg: Magnesium
# Al: Aluminum
# Si: Silicon
# K: Potassium
# Ca: Calcium
# Ba: Barium
# Fe: Iron
# Type of glass (Type 1, Type 2, Type 3, Type 5, Type 6, Type 7) 

# 2. Data Preparation
# 2a. Exploratory data analysis (EDA) 

# loading the data set
glass = pd.read_csv(r"C:\Users\yousu\Desktop\ML projects\glass classification using KNN\glass.csv")


# i) First Moment business decision (Measures of central tendency) 
# MEAN:
# Also called average value. 
# Useful in understanding typical value of a attribute
# Influenced by outliers
glass.RI.mean()
glass.Na.mean()
glass.Mg.mean()
glass.Al.mean()
glass.Si.mean()
glass.K.mean()
glass.Ca.mean()
glass.Ba.mean()
glass.Fe.mean()

# MEDIAN:
# Middle most value of sorted data.
# Not influenced by presence of outliers.
glass.RI.median()
glass.Na.median()
glass.Mg.median()
glass.Al.median()
glass.Si.median()
glass.K.median()
glass.Ca.median()
glass.Ba.median()
glass.Fe.median()

# MODE:
# Most repeated value
# Can also be used for non-numeric data
glass.RI.mode()
glass.Na.mode()
glass.Mg.mode() 
glass.Al.mode()
glass.Si.mode()
glass.K.mode()
glass.Ca.mode()
glass.Ba.mode()
glass.Fe.mode()

# ii) Second Moment business decision (Measures of Dispersion) 
# VARIANCE:
# It measures how much the values in a dataset vary from the mean.
# high variance indicates that the data points are spread out widely from the mean.
glass.RI.var()
glass.Na.var()
glass.Mg.var()
glass.Al.var()
glass.Si.var()
glass.K.var()
glass.Ca.var()
glass.Ba.var()
glass.Fe.var()

# STANDARD DEVIATION:
# Square root of variance.
# Advantage is that we can get back our original inputs from squated units.
glass.RI.std()
glass.Na.std()
glass.Mg.std()
glass.Al.std()
glass.Si.std()
glass.K.std()
glass.Ca.std()
glass.Ba.std()
glass.Fe.std()

# iii) Third Moment business decision (Skewness) 
# Indicates the extent to which the data deviates from symmetry around the mean.
# Skewness can be positive, negative, or zero
# Positive skewness: mean is typically greater than the median.
# Negative skewness: mean is typically less than the median.
# Zero skewness: distribution is perfectly symmetrical, and the mean and median are equal.
glass.RI.skew()
glass.Na.skew()
glass.Mg.skew()
glass.Al.skew()
glass.Si.skew()
glass.K.skew()
glass.Ca.skew()
glass.Ba.skew()
glass.Fe.skew()

# iv) Fourth Moment business decision (Kurtosis) 
# positive kurtosis: If kurtosis is greater than 3,that the distribution has more extreme values or outliers than would be expected under a normal distribution.
# zero kurtosis: If kurtosis is equal to 3, it indicates that the distribution has similar tails and peak as the normal distribution. In other words, it has neither unusually heavy nor light tails compared to the normal distribution.
# negative kurtosis: If kurtosis is less than 3, it indicates that the distribution has lighter tails and a lower peak than the normal distribution. This means that the distribution has fewer extreme values or outliers than would be expected under a normal distribution.
glass.RI.kurt()
glass.Na.kurt()
glass.Mg.kurt()
glass.Al.kurt()
glass.Si.kurt()
glass.K.kurt()
glass.Ca.kurt()
glass.Ba.kurt()
glass.Fe.kurt()

# v) Graphical Representation:
# As there are many attributes to consider, let's go with heat map
# finding corelation between attributes
corr = glass.corr()
sns.heatmap(corr, cmap = "RdBu", vmax = 1, vmin = -1)

# Analysis from heat map 
# There is no corelation between Ca,K and Type. We can drop Ca, K columns 
glass.drop(columns=['Ca', 'K'], inplace=True)

# 2b. Data pre-processing
# i) Handling Duplicates:
sum(glass.duplicated())
glass = glass.drop_duplicates()
sum(glass.duplicated())

# ii) Outlier Ananlysis
# box-plot helps in identifying outliers
plt.boxplot(glass.RI)
plt.boxplot(glass.Na)
plt.boxplot(glass.Mg)
plt.boxplot(glass.Al)
plt.boxplot(glass.Si)
plt.boxplot(glass.Ba)
plt.boxplot(glass.Fe)
# All attributes have outliers except 'Mg'
# To handle outliers winsorizer function can be used
columns_to_winsorize = ['RI', 'Na', 'Al', 'Si', 'Fe']
winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=columns_to_winsorize)
glass = pd.DataFrame(glass)
winsorizer.fit(glass)
glass = winsorizer.transform(glass)
plt.boxplot(glass.RI)
plt.boxplot(glass.Na)
plt.boxplot(glass.Mg)
plt.boxplot(glass.Al)
plt.boxplot(glass.Si)
plt.boxplot(glass.Ba)
plt.boxplot(glass.Fe)

# iii) Missing values
glass.isna().sum()
# no missing values in our data set 

# iv) Transformation
# check if data is normal, or else apply suitable transformation
# For RI coloumn
stats.probplot(glass.RI, dist='norm', plot = pylab)
# For Na column
stats.probplot(glass.Na, dist='norm', plot = pylab)
# For Mg column
stats.probplot(glass.Mg, dist='norm', plot = pylab)
stats.probplot(np.log(glass.Mg), dist='norm', plot = pylab)
stats.probplot(np.exp(glass.Mg), dist='norm', plot = pylab)
glass['Mg'] = np.exp(glass.Mg)
# For Al column
stats.probplot(glass.Al, dist='norm', plot = pylab)
# For si column
stats.probplot(glass.Si, dist='norm', plot = pylab)
# For Ba column
stats.probplot(glass.Ba, dist='norm', plot = pylab)
stats.probplot(np.log(glass.Ba), dist='norm', plot = pylab)
stats.probplot(np.sqrt(glass.Ba), dist='norm', plot = pylab)
stats.probplot(np.exp(glass.Ba), dist='norm', plot = pylab)
# For Fe column
stats.probplot(glass.Fe, dist='norm', plot = pylab)
stats.probplot(np.log(glass.Fe), dist='norm', plot = pylab)
stats.probplot(np.sqrt(glass.Fe), dist='norm', plot = pylab)
stats.probplot(np.exp(glass.Fe), dist='norm', plot = pylab)

# v) Feature Scaling
# Feature Scaling ensures that all features contribute equally to the model 
# and prevents features with larger values from dominating.
y = glass[['Type']]
glass.drop(columns=['Type'], inplace=True)
scaler = MinMaxScaler()
glass = pd.DataFrame(scaler.fit_transform(glass), columns=glass.columns)


# 3. Model Building

x = glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'Ba', 'Fe']]


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state=45)

knn_model = KNeighborsClassifier(n_neighbors=3, metric = 'manhattan')
knn_model.fit(x_train, y_train)

knn_model.score(x_train, y_train)
knn_model.score(x_test, y_test)

# creating confusion matrix for test data predictions
y_pred = knn_model.predict(x_test)
report = classification_report(y_test,y_pred) 
accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred) 
sns.heatmap(cm, cmap = "RdBu", annot=True)

x_test 
y_pred = knn_model.predict([['RI', 'Na', 'Mg', 'Al', 'Si', 'Ba', 'Fe'],[0.2,0.1,0.4,0.333,0.9,0.87,0.1]])


new_data = {
    'RI': [0.9996],
    'Na': [0.54],
    'Mg': [0.23],
    'Al': [0.12],
    'Si': [0.345],
    'Ba': [0.0002],
    'Fe': [0.008]
}

new_X = pd.DataFrame(new_data)
y_pred = knn_model.predict(new_X)

print(y_pred)



























































































