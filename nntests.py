#!/usr/bin/env python
# coding: utf-8

# In[94]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) # First hidden layer; flattened version of entire input matrix
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)) # Hidden layer 2, 128 neurons
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)) # Hidden layer 3, 128 neurons
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # Output layer, 10 neurons

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)


# In[104]:


import numpy as np

new_model = model # Duplicate model for manipulation
predictions = new_model.predict([x_test]) # Make a prediction
print(np.argmax(predictions[2001])) # Prints prediction


# In[103]:


import matplotlib.pyplot as plt

plt.imshow(x_test[2001], plt.cm.binary) # Shows graph representation of hand-writter digit


# In[ ]:




