!pip install tensorflow

"""  **ANN**"""

import numpy as np
import pandas as pd

df = pd.read_csv("/content/Attrition.csv")

df.head(5)

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

df.columns

df = df.dropna()

x = df.drop(columns = ['Attrition'], axis=1)
y = df['Attrition']

x.info()

categorical_features =['BusinessTravel',
                      'Department',
                      'EducationField',
                      'Gender',
                      'JobRole',
                      'MaritalStatus',
                      'Over18',
                      'OverTime']

x_encoded = pd.get_dummies(x, columns = categorical_features ,
                            drop_first =True)

x_encoded

lb = LabelEncoder()
y_encoded = lb.fit_transform(y)

y_encoded

x_train, x_test, y_train, y_test = train_test_split(x_encoded,
                                                    y_encoded,
                                                    test_size=0.2,
                                                    random_state=42)

x_train.head()

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

"""Now we will create architecture of our model"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units = 128, activation = 'relu',
                          input_dim = x_train_scaled.shape[1]), #input Layer
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units = 64, activation = 'relu'),   #hidden Layer
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')   #output Layer
])

loss_fn = tf.keras.losses.BinaryCrossentropy()
metrics = ['accuracy']

learning_rate = 0.001
momentum = 0.9
optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate,
                                    momentum = momentum , nesterov = True)

model.compile(optimizer = optimizer, loss = loss_fn, metrics = metrics)

#train the model

model.fit(x_train_scaled, y_train, epochs = 50, batch_size = 8, validation_split= 0.1)

loss, accuracy =model.evaluate(x_test_scaled, y_test)

print(f'Test Loss = {loss:.4f}, Test Accuracy ={accuracy:.4f}')

