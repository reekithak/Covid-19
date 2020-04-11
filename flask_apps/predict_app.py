#!/usr/bin/env python
# coding: utf-8

# In[5]:


import base64
import numpy as np
from PIL import Image
import keras
from keras.models import Sequential , load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify , Flask
from keras.models import model_from_json


# In[4]:


app = Flask(__name__)


# In[7]:


def get_model():
    global model
    with open('model.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('model.h5')
    print("model Loaded")


# In[8]:


def preprocess_image(image,target_size):
    if image.mode!="RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    return image
    


# In[ ]:


print("keras model is loaded")
get_model()


# In[16]:


@app.route("/predict",methods=["POST"])
def predict():
    message= request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image,target_size=(512,512))
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #img = cv2.resize(img,(64,64))
    img = np.reshape(img,[1,512,512,3])
    prediction = model.predict_classes(processed_image).to_list()
    
    #prediction = model.predict(processed_image).to_list()
    
    response = {'prediction': {'Normal':prediction[0][0], 'Weird':prediction[0][1] }}
                    
    return jsonify(response)
if __name__ == '__main__':
    app.run(port = 5000, debug=True)

# In[ ]:




