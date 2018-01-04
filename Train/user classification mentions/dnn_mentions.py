
# coding: utf-8

# In[1]:


from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
import os


# In[16]:


# #Determine no of entities
entities_dir=glob('../dataset/*')
#Determine the name of entities
entities=[i.split(os.sep)[-1] for i in entities_dir ]

#Creating hashmap ...we can avoid this in future as string classes are now supported
mapping={}
for i in range(1,len(entities)+1):
    mapping[entities[i-1]]=i


# In[13]:


X_train=[]
y_train=[]


for entity in entities:
    #Read all json files
    num=0
    accounts=glob('../dataset/'+entity+'/*.json')
    for account in accounts:
        f=open(account,encoding='utf-8')
        user=account.split(os.sep)[-1]
        mentions=''
        for tweet in f:
            x=json.loads(tweet)
            for user_mentions in x['entities']['user_mentions']:
                mentions=mentions+' , '+user_mentions['id_str']
        f.close()
        num+=1
        X_train.append(mentions)
        y_train.append(mapping[entity])
    print ("NO of samples for ",entity,' : ',num)
print ("Total Samples:",len(y_train))


# In[4]:


tfidf_vectorizer =TfidfVectorizer(max_df=0.999, max_features=10000, lowercase=True)
X_train =tfidf_vectorizer.fit_transform(X_train)
joblib.dump(tfidf_vectorizer,'tfidf_mentions.joblib.pkl')
#For debugging purposes
print (X_train.shape)
y_train=np.array(y_train)


# In[5]:


#Initializing the label binarizer that will transform the labels into one hot encoding
lb = preprocessing.LabelBinarizer()
lb.fit(y_train)
#Saving the transformer
joblib.dump(lb,'lb_mentions.joblib.pkl')
#Transforming labels
y_train=lb.transform(y_train)


# In[6]:


#Data normalization
#We can experiment with this
#from sklearn.preprocessing import normalize
#X_train=normalize(X_train)


# In[5]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

#Initiallizing model
model = Sequential()
model.add(Dense(units=256, activation='elu',input_dim=X_train.shape[1],kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(units=256, activation='elu',kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(units=256, activation='elu',kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(units=4, activation='softmax'))

callbacks = [
    #This will stop training if the model is not improving for 100 epochs
    keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0,patience=250),
    #This will save the model weights with the best performance
    keras.callbacks.ModelCheckpoint('Mentions_weights.h5', monitor='val_loss', save_best_only=True, verbose=1),
]
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.adamax(),
              metrics=['accuracy'])
model.summary()


# In[8]:


#You can change these parameters and the time it takes depends on the dataset
model.fit(X_train,y_train, epochs=1000, batch_size=64,validation_split=0.15,callbacks=callbacks,shuffle=True)


# In[6]:


#Save model's structure
json_string = model.to_json()
f=open('Mentionsmodel.txt')
f.write(json_string)
f.close()

