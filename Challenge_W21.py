#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing

# In[2]:


# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf

#  Import and read the charity_data.csv.
import pandas as pd 
application_df = pd.read_csv("https://static.bc-edx.com/data/dla-1-2/m21/lms/starter/charity_data.csv")
application_df.head()


# In[3]:


# Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
application_df = application_df.drop(["EIN","NAME"],axis=1)


# In[4]:


# Determine the number of unique values in each column.
application_df.nunique()


# In[6]:


# Look at APPLICATION_TYPE value counts for binning
application_df['APPLICATION_TYPE'].value_counts()


# In[15]:


# Choose a cutoff value and create a list of application types to be replaced
# use the variable name `application_types_to_replace`
application_types_to_replace=['T3','T4' , 'T6' ,'T5','T19' ,'T8','T7','T10' ]

# Replace in dataframe
for app in application_types_to_replace:
    application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(app,"Other")

# Check to make sure binning was successful
application_df['APPLICATION_TYPE'].value_counts()


# In[16]:


# Look at CLASSIFICATION value counts for binning
application_df['CLASSIFICATION'].value_counts()


# In[25]:


# You may find it helpful to look at CLASSIFICATION value counts >1
application_df['CLASSIFICATION'].value_counts()[application_df['CLASSIFICATION'].value_counts()>1]


# In[27]:


# Choose a cutoff value and create a list of classifications to be replaced
# use the variable name `classifications_to_replace`
classifications_to_replace =  ['C1000', 'C2000', 'C3000', 'C1200', 'C2700', 'C7000', 'C7200',
       'C1700', 'C4000', 'C7100', 'C2800', 'C6000', 'C2100', 'C1238',
       'C5000', 'C7120', 'C1800', 'C4100', 'C1400', 'C1270', 'C2300',
       'C8200', 'C1500', 'C7210', 'C1300', 'C1230', 'C1280', 'C1240',
       'C2710', 'C2561', 'C1250', 'C8000', 'C1245', 'C1260', 'C1235',
       'C1720', 'C1257', 'C4500', 'C2400', 'C8210', 'C1600', 'C1278',
       'C1237', 'C4120', 'C2170', 'C1728', 'C1732', 'C2380', 'C1283',
       'C1570', 'C2500', 'C1267', 'C3700', 'C1580', 'C2570', 'C1256',
       'C1236', 'C1234', 'C1246', 'C2190', 'C4200', 'C0', 'C3200',
       'C5200', 'C1370', 'C2600', 'C1248', 'C6100', 'C1820', 'C1900',
       'C2150']

# Replace in dataframe
for cls in classifications_to_replace:
    application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(cls,"Other")
    
# Check to make sure binning was successful
application_df['CLASSIFICATION'].value_counts()


# In[28]:


# Convert categorical data to numeric with `pd.get_dummies`
application_df=pd.get_dummies(application_df,dtype=float)


# In[31]:


# Split our preprocessed data into our features and target arrays
#  YOUR CODE GOES HERE

# Split the preprocessed data into a training and testing dataset
#  YOUR CODE GOES HERE


# In[32]:


# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# ## Compile, Train and Evaluate the Model

# In[33]:


# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
#  YOUR CODE GOES HERE

nn = tf.keras.models.Sequential()
# First hidden layer
nn.add(Dense(units=8, activation=))

# Second hidden layer
#  YOUR CODE GOES HERE

# Output layer
#  YOUR CODE GOES HERE

# Check the structure of the model
nn.summary()


# In[34]:


# Compile the model
#  YOUR CODE GOES HERE


# In[36]:


# Train the model
#  YOUR CODE GOES HERE


# In[37]:


# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# In[38]:


# Export our model to HDF5 file
#  YOUR CODE GOES HERE

