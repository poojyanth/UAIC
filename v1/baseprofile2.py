
# -*- coding: utf-8 -*-
"""01_neural_network_regression_with_tensorflow.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18ytOshAD4BR8-4nvqgurMpBijgaTxDIw
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x=np.array([-1,-2,-3,0,1,2,3,4])
y=np.array([9,8,7,10,11,12,13,14])

y==x+10

tf.expand_dims(x,axis=-1)

a=np.array([[1,2],[3,4]])
tf.expand_dims(a,axis=-1)

"""# Steps in modelling with tensorflow
1. Creating the model -define an input and output layers and as well as the hidden layers with in the deep learning model.
2. Compiling a model - define a loss function (a function which tells our model how wrong it is) and optimizer (which tells our funtion hoe to improve and learn) and evaluation metrics.
3. Fitting the model.
"""

# Create the model
model=tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

#compile the model

model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])

#fit th model

model.fit(tf.expand_dims(x,axis=-1),y,epochs=100)

model.predict([-5])

"""# How to improve model
To improve our model, we alter almost every part of the 3 steps we went through before.

1. Creating a model - here you might want to add more layers, increase the number of hidden units (also called neurons) within each layer, change the activation functions of each layer.
2. Compiling a model - you might want to choose optimization function or perhaps change the learning rate of the optimization function.
3. Fitting a model - perhaps you could fit a model for more epochs (leave it training for longer) or on more data (give the model more examples to learn from).
"""

model=tf.keras.Sequential([
    tf.keras.layers.Dense(50,activation=None),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mae',optimizer=tf.keras.optimizers.Adam(lr=0.01),metrics=['mae'])

model.fit(tf.expand_dims(x,axis=-1),y,epochs=100)

model.predict([17])

x=tf.range(-100,100,4)

y=x+10

x_train=x[:40]
x_test=x[40:]
y_train=y[:40]
y_test=y[40:]

len(x_test
  )

# Create the model
model=tf.keras.Sequential([
    tf.keras.layers.Dense(6,name='input_layer'),
    tf.keras.layers.Dense(1,name='output_layer'),

],name='onemodel')

#compile the model

model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])

#fit th model

model.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=100)

model.summary()

"""Trainable parameters are those that the model can update as it trains.
Non trainable parameters are those which can't be updated while training.These may include already trained params in the transfer learning models.
"""

from tensorflow.keras.utils import plot_model

plot_model(model=model,show_shapes=True)

preds=model.predict(x_test)
x_test.shape

def plotter(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,preds=preds):

      plt.figure(figsize=(10,10))
      plt.scatter(x_train,y_train,c='b',label='Training data')
      plt.scatter(x_test,y_test,c='g',label='Testing Data')
      plt.scatter(x_test,preds,c='r',label='Predictions')
      plt.legend()

plotter()

preds=tf.constant(preds)
preds=tf.squeeze(preds)

y_test

preds.shape

mae=tf.metrics.mean_absolute_error(y_test,preds)
mae

def mse(y_test,preds):
  return tf.metrics.mean_squared_error(y_test,preds)
def mae(y_test,preds):
  return tf.metrics.mean_absolute_error(y_test,preds)

# Create the model
tf.random.set_seed(9)
model_1=tf.keras.Sequential([
    tf.keras.layers.Dense(1,name='output_layer'),

],name='onemodel')

#compile the model

model_1.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])

#fit th model

model_1.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=100)

y_preds1=model_1.predict(x_test,)
model_1.evaluate(x_test,y_test)
plotter(preds=y_preds1)



# Create the model
tf.random.set_seed(4)
model_2=tf.keras.Sequential([
        tf.keras.layers.Dense(1,name='input_layer'),

    tf.keras.layers.Dense(1,name='output_layer'),

],name='onemodel')

#compile the model

model_2.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])

#fit th model

model_2.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=100)

y_preds2=model_2.predict(x_test)

plotter(preds=y_preds2)

# Create the model
tf.random.set_seed(4)
model_3=tf.keras.Sequential([
        tf.keras.layers.Dense(1,name='input_layer'),

    tf.keras.layers.Dense(1,name='output_layer'),

],name='onemodel')

#compile the model

model_3.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])

#fit th model

model_3.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=500)

y_preds3=model_3.predict(x_test)

plotter(preds=y_preds3)

[mae(y_test,tf.squeeze(y_preds1)),
mae(y_test,tf.squeeze(y_preds2)),
mae(y_test,tf.squeeze(y_preds3))]

import pandas as pd
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

insurance

insurance.dtypes

y=insurance['charges']
x=insurance.drop('charges',axis=1)
y

from sklearn.model_selection import train_test_split

x=pd.get_dummies(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

y_test

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# cate_feat=['sex','region','smoker']
# one_hot=OneHotEncoder()
# transformer=ColumnTransformer([('one_hot',one_hot,cate_feat)],remainder='passthrough')
# transformer2=transformer.fit_transform(x_train)
# transformer3=transformer.fit_transform(x_test)

# x_train=pd.DataFrame(transformer2)
# x_test=pd.DataFrame(transformer3)

tf.random.set_seed(3)
insurance_model=tf.keras.Sequential([
        tf.keras.layers.Dense(1,name='input_layer'),

    tf.keras.layers.Dense(1,name='output_layer'),

],name='onemodel')

#compile the model

insurance_model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])

#fit th model

insurance_model.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=100)

insurance_model.evaluate(x_test,y_test)

tf.random.set_seed(3)
insurance_model2=tf.keras.Sequential([
        tf.keras.layers.Dense(100,name='input_layer'),
        tf.keras.layers.Dense(10,name='mid_layer'),

    tf.keras.layers.Dense(1,name='output_layer'),

],name='onemodel')

#compile the model

# insurance_model2.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])
insurance_model2.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(),metrics=['mae'])


#fit th model

insurance_model2.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=100)

insurance_model2.evaluate(x_test,y_test)

tf.random.set_seed(3)
insurance_model3=tf.keras.Sequential([
        tf.keras.layers.Dense(150,name='input_layer'),
        tf.keras.layers.Dense(15,name='mid_layer'),

    tf.keras.layers.Dense(1,name='output_layer'),

],name='onemodel')

#compile the model

# insurance_model2.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])
insurance_model3.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(),metrics=['mae'])


#fit the model

insurance_model3.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=200)

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Create column transformer (this will help us normalize/preprocess our data)
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]), # get all values between 0 and 1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

# Create X & y
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Build our train and test sets (use random state to ensure same split as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit column transformer on the training data only (doing so on test data would result in data leakage)
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScalar) and one hot encoding (OneHotEncoder)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

X_train_normal[0]

X_train.loc[0]

# Set random seed
tf.random.set_seed(42)

# Build the model (3 layers, 100, 10, 1 units)
insurance_model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])

# Compile the model
insurance_model_3.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['mae'])

# Fit the model for 200 epochs (same as insurance_model_2)
insurance_model_3.fit(X_train_normal, y_train, epochs=200, verbose=0)

insurance_model_3_loss, insurance_model_3_mae = insurance_model_3.evaluate(X_test_normal, y_test)

