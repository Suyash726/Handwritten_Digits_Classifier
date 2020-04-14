#!/usr/bin/env python
# coding: utf-8

# In[16]:


"""
Description :- MNIST handwritten digit classification
"""


# In[12]:


import numpy as np
import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


# In[31]:


# Load the dataset
train_images = mnist.train_images()# This one is the images
train_labels = mnist.train_labels()# This is the label
test_images = mnist.test_images() # Testing data images
test_labels = mnist.test_labels() #Testing data  labels


# In[32]:


# Normalize the data
train_images = (train_images/255)-0.5
test_images = (test_images/255)-0.5
# flatten the images 
train_images = train_images.reshape((-1,784))

test_images = test_images.reshape((-1,784))


# In[33]:


print(train_images.shape)
print(test_images.shape)


# In[34]:


# Model Building
model = Sequential()
model.add(Dense(64,activation='relu',input_dim = 784))
model.add(Dense(34,activation='relu'))
model.add(Dense(10,activation ='softmax'))


# In[35]:


# Compilation
# The loss function and optimizer
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[36]:


# Train the model
model.fit(train_images,to_categorical(train_labels),epochs = 5, batch_size=32)


# In[37]:


# Evaluate the model
model.evaluate(test_images,to_categorical(test_labels))


# In[39]:


model.save_weights('model.h5')


# In[45]:


predictions = model.predict(test_images[:10])
print(np.argmax(predictions,axis=1))
print(test_labels[:10])


# In[49]:


for i in range(0,10):
    first_image = test_images[i]
    first_image = np.array(first_image,dtype='float')
    pixels = first_image.reshape(28,28)
    plt.imshow(pixels,cmap='gray')
    plt.show()


# In[ ]:




