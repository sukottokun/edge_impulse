#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('python -m pip install tensorflow==2.15.0 edgeimpulse')


# In[4]:


from tensorflow import keras
import edgeimpulse as ei


# In[5]:


# Settings
ei.API_KEY = "ei_2f55a1f166c6ea4954db69e9f4ce3c146bb8b65d55354cf19835e94c1a68ff45"
labels = ["0","1","2","3","4","5","6","7","8","9",]
num_classes = len(labels)
deploy_filename = "my_model_cpp.zip"


# In[6]:


# Load Mnist data
(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape = x_train[0].shape


# In[7]:


# Build the model
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu', input_shape=input_shape),
    keras.layers.Dense(num_classes, activation='softmax')
])
    
# Compile the model
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


# In[8]:


# Train the model
model.fit(x_train,
         y_train,
         epochs=5)


# In[10]:


# Evaluate model on test set
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")


# In[11]:


# List the available profile devices
ei.model.list_profile_devices()


# In[12]:


# Estimate the RAM, ROM, and inference time for out model on the target hardware
try:
    profile = ei.model.profile(model=model, device='cortex-m4f-80mhz')
    print(profile.summary())
except Exception as e:
    print(f"Could not profile: {e}")


# In[13]:


ei.model.list_deployment_targets()


# In[15]:


# Set model info
model_output_type = ei.model.output_type.Classification(labels=labels)

# Create C++ libs with trained model
deploy_bytes = None
try:
    deploy_bytes = ei.model.deploy(model=model, model_output_type=model_output_type, deploy_target='zip')
    print(profile.summary())
except Exception as e:
    print(f"Could not deploy: {e}")
    
# Write raw bytes to file
if deploy_bytes:
    with open(deploy_filename, 'wb') as f:
        f.write(deploy_bytes.getvalue())


# In[ ]:




