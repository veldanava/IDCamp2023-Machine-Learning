# train regression linear
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

# rooms data
rooms = np.array([1,1,2,2,3,4,4,5,5,5])

# house data ( as dollar )
house_price = np.array([15000, 18000, 27000, 34000, 50000, 68000, 65000, 81000,85000, 90000])

# render matplot
plt.scatter(rooms, house_price)

# train logistic regression
import pandas as pd

# read dataset and change to dataframe
df = pd.read_csv('Social_Network_Ads.csv')

# delete row user_id
data = df.drop(columns=['User ID'])

# run process
data = pd.get_dummies(data)
data
