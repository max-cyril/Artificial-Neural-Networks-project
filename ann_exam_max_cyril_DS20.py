#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import os
import cv2
from shutil import copyfile,move
import os
from sklearn.manifold import TSNE

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from   sklearn.decomposition import PCA
from   sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from   sklearn.preprocessing import StandardScaler


# In[2]:


os.chdir('E:/poubelle/data/Ann/train_data.csv/')


# In[25]:


get_ipython().run_cell_magic('time', '', "df_tr=pd.read_csv('E:/poubelle/data/Ann/train_data.csv/train_data.csv',sep=',',header=None)\ndf_ts=pd.read_csv('E:/poubelle/data/Ann/test_data.csv/test_data.csv',sep=',',header=None)")


# In[6]:


print(df_tr.shape)
df_tr.head()


# In[26]:


print(df_ts.shape)
df_ts.head()


# In[6]:


x=df_tr.iloc[:,0:784]
y=df_tr.iloc[:,784:]
y.value_counts().plot.bar()


# #!pip install pivottablejs
# from pivottablejs import pivot_ui
# pivot_ui(y)

# In[ ]:





# In[ ]:





# In[ ]:





# # fully connected model

# In[115]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense ,Flatten ,Dropout ,Conv2D,MaxPooling2D
from tensorflow.keras.losses import BinaryCrossentropy,sparse_categorical_crossentropy
from sklearn.model_selection import train_test_split
xtr,xtst,ytr,ytst= train_test_split(x,y,test_size=.2,random_state=10)


# In[143]:


scaler=StandardScaler()
df_xtr=scaler.fit_transform(xtr)
df_xtst=scaler.transform(xtst)
test_data=scaler.transform(df_ts)


# In[119]:


model= Sequential()
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
#model.summary()


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[120]:


hist2=model.fit(df_xtr,ytr,validation_data=(df_xtst,ytst),batch_size=40,epochs=100,verbose=0)


# In[121]:


fc_loss,fc_accuracy=model.evaluate(df_xtst,ytst)
print(fc_loss,fc_accuracy)


# In[122]:


plt.plot(hist2.history['val_accuracy'])
plt.plot(hist2.history['accuracy'])
plt.plot(hist2.history['loss'])
plt.plot(hist2.history['val_loss'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # cnn

# In[ ]:





# In[144]:


df_xtr=df_xtr.reshape(df_xtr.shape[0], 28,28, 1)
df_xtst=df_xtst.reshape(df_xtst.shape[0], 28,28, 1)
test_data=test_data.reshape(test_data.shape[0], 28,28, 1)
df_xtr.shape ,df_xtst.shape,test_data.shape


# In[65]:


mod = Sequential()

# add first convolutional layer
mod.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28, 1)))

# add second convolutional layer
mod.add(Conv2D(64, (3, 3), activation='relu'))

# add one max pooling layer 
mod.add(MaxPooling2D(pool_size=(2, 2)))

# add one dropout layer
mod.add(Dropout(0.125))

# add flatten layer
mod.add(Flatten())

# add dense layer
mod.add(Dense(128, activation='relu'))

# add another dropout layer
mod.add(Dropout(0.25))

# add dense layer
mod.add(Dense(10, activation='softmax'))

# complile the model and view its architecur
mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

mod.summary()


# In[74]:


hist=mod.fit(df_xtr,ytr,validation_data=(df_xtst,ytst),epochs=20,verbose=1)


# In[111]:


cnn_loss,cnn_accuracy=mod.evaluate(df_xtst,ytst)
print(loss,accuracy)


# In[81]:


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])


# In[ ]:





# In[78]:


from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


# In[79]:


earl_stp=EarlyStopping(monitor='val_loss',patience=5)
cb=[earl_stp,ModelCheckpoint(filepath='maxcyril_ANN_exam_model.h5',monitor='val_loss',save_best_only=True)]
hist=mod.fit(df_xtr,ytr,validation_data=(df_xtst,ytst),epochs=20,verbose=1,callbacks=[cb])


# In[84]:


from tensorflow.math import confusion_matrix


# In[104]:


sns.heatmap(confusion_matrix(ytst,np.argmax(mod.predict(df_xtst), axis=-1)),annot=True,fmt="d")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # svm

# In[29]:


from sklearn import svm
msv=svm.SVC()
msv.fit(df_xtr,ytr)


# In[37]:


print(ytr)
ytr2=np.array(ytr).ravel()
ytst2=np.array(ytst).ravel()


# In[33]:


msv.score(df_xtst,ytst)


# In[56]:


get_ipython().run_cell_magic('time', '', 'from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix\nimport seaborn as sns\nfrom sklearn import svm\nmsv2=svm.SVC()\nmsv2.fit(df_xtr,ytr2)\nmsv2.predict(df_xtst)\nprint(msv2.score(df_xtst,ytst2))\nsns.heatmap(confusion_matrix(ytst2,msv2.predict(df_xtst)),annot=True,fmt="d")')


# In[58]:


get_ipython().run_cell_magic('time', '', 'from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix\nimport seaborn as sns\nfrom sklearn import svm\nmsv3=svm.SVC(kernel=\'linear\')\nmsv3.fit(df_xtr,ytr2)\nmsv3.predict(df_xtst)\nprint(msv3.score(df_xtst,ytst2))\nsns.heatmap(confusion_matrix(ytst2,msv3.predict(df_xtst)),annot=True,fmt="d")')


# In[ ]:





# In[ ]:





# In[ ]:





# # knn

# In[55]:


get_ipython().run_cell_magic('time', '', 'from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix\nimport seaborn as sns\nfrom sklearn.neighbors import KNeighborsClassifier\nknn = KNeighborsClassifier()\nknn.fit(df_xtr,ytr2)\nknn.predict(df_xtst)\nprint(knn.score(df_xtst,ytst2))\nsns.heatmap(confusion_matrix(ytst2,knn.predict(df_xtst)),annot=True,fmt="d")')


# In[ ]:





# In[ ]:





# In[ ]:





# In[112]:


#base on accuracy the best model selected for this classification task is 
print(cnn_accuracy)


# In[ ]:





# In[ ]:





# In[145]:


#prediction
pred =np.argmax(mod.predict(test_data), axis=-1)
pred


# In[146]:


#create id column
id=np.array(range(len(pred)))
#create pandas DataFrame
df=pd.DataFrame({'id':id, 'category':list(pred)},columns=['id','category'])


# In[147]:


df


# In[148]:


df.to_csv("ann_max_cyril_DS20.csv",encoding='UTF-8',index=False)

