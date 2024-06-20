#!/usr/bin/env python
# coding: utf-8

# 
# <p style="background-color:purple; font-family:newtimeroman; color:#FFF9ED; font-size:200%; text-align:center; border-radius:20px; padding:20px;"><strong>Medical Text Dataset -Cancer Doc Sentiment Analysis and classification.</strong></p>
# 

# In[23]:


from IPython.display import display, Image

# Display the image
display(Image(filename=r'C:\Users\user\Downloads\can_img.jpg', width=950, height=750))


# ### The data is a collection of 7570 Rows and 3 column variables. Each row includes a written comment as well as additional customer information. Also each row corresponds to a customer review, and includes the variables:
# 

# In[ ]:





# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover"><span style="font-size: 20px;">AIM </span>
# 
# ### The aim of the project is to use NLP and Deep learning techniques to automatically classify medical
# ### documents related to cancer, leveraging a dataset of medical texts.

# In[ ]:





# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover"><span style="font-size: 20px;">Feature Information: </span>
# 
# - **1** - 0 :colon cancer=2579, lung cancer=2180, thyroid cancer=2810
# 
# - **2** - a:Text Data regarding Canser Health Canser Text Classification categoroial

# <a id="2"></a>
# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;"> LIBRARIES NEEDED IN THE STUDY<p>
# 

# In[1]:


#LIBRARIES NEEDED IN THE STUDY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
import keras.models
#from keras.models import Sequential
from keras.layers import Conv2D , MaxPool2D ,Flatten , Dense, Dropout
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from PIL import Image
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D , BatchNormalization;
#from models import custom_convnet
from keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report
from keras.layers import Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.optimizers import Adam




# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
# <span style="font-size: 20px;">NLP</span></a>
# 

# In[3]:


#In a project context, you could succinctly define NLP as "the branch of artificial intelligence (AI)
#focused on enabling computers to understand,interpret, and generate human language."


# <a id="3"></a>
# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;"> Loading and Reading Data <p>

# In[4]:


pd.read_csv(r"D:\Medical.csv",encoding='latin-1')


# In[5]:


med=pd.read_csv(r"D:\Medical.csv",encoding='latin-1')


# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;"> 

# In[6]:


# Plot the histogram using matplotlib
plt.hist(med['0'], bins='auto', color='skyblue', edgecolor='black')

plt.xlabel('0')
plt.ylabel('count')
plt.title('types of canser')

plt.show()


# In[ ]:





# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
# <span style="font-size: 20px;">Data Analysis</span></a>
# 

# In[7]:


med.isnull().sum()>0


# In[8]:


med=med.loc[:,['0','a']]


# In[9]:


med


# In[10]:


med=med.rename(columns={'0':'y','a':'x'})


# In[11]:


med


# In[12]:


med.y.value_counts()


# In[13]:


med.head()


# In[14]:


med=med.replace({'Thyroid_Cancer':0,'Colon_Cancer':1,'Lung_Cancer':2})


# In[15]:


med


# In[16]:


med.x=med.x.str.lower()
# before doing any analysis convert data into lower or upper


# In[17]:


med.head()


# 
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
# <span style="font-size: 20px;">Stop words</span></a>
# 

# In[18]:


from nltk.corpus import stopwords
l1=stopwords.words("english")
# we remove them


# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
# <span style="font-size: 20px;">punctuation</span></a>
# 

# In[19]:


import string

string.punctuation


# In[ ]:





# In[20]:


def text_process(mess):
    """
    1. remove the punctuation
    2. remove the stopwords
    3. return the list of clean textwords
    
    """
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc="".join(nopunc)
    
    return [word for word in nopunc.split() if word not in l1]


# In[21]:


med['x'].apply(text_process)


# 
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
# <span style="font-size: 20px;">Create TDM</span></a>
# 

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer  # this is used to count each and every unique word...


# In[23]:


import timeit
start=timeit.default_timer()

bow_transformer=CountVectorizer(analyzer=text_process).fit(med['x'])

stop=timeit.default_timer()
execution_time=stop-start
print("Program executed in ",execution_time)


# In[24]:


bow_transformer=CountVectorizer(analyzer=text_process).fit(med['x'])


# In[25]:


bow_transformer.vocabulary_


# In[ ]:





# In[26]:


len(bow_transformer.vocabulary_)
# there are 208269 unique words after removing punctuation and stopwords
# that when we create our TDM our tdm will have 208269 columns


# In[27]:


tdm=bow_transformer.transform(med['x'])


# In[28]:


tdm.shape


# In[29]:


type(tdm)


# 
# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;"> Pre-Processing Data  <p>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Spliting Data into Train and Test</span>
# </a>
# 

# In[30]:


# tdm is like our x variable 
from sklearn.model_selection import train_test_split
tdm_train,tdm_test,train_y,test_y=train_test_split(tdm,med['y'],test_size=.2)


# In[31]:


tdm_train.shape


# In[32]:


train_y.shape


# In[33]:


# tdm is like our x variable 
from sklearn.model_selection import train_test_split
tdm_train,tdm_test,train_y,test_y=train_test_split(tdm,med['y'],test_size=.2)

tdm_train.shape

train_y.shape


# 
# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;"> Naive Bayes  <p>

# In[34]:


# In a project, you could describe Naive Bayes as "a simple probabilistic classifier based on 
# Bayes' theorem with the assumption of independence between features."


# In[35]:


from sklearn.naive_bayes import MultinomialNB

nb=MultinomialNB()
nb.fit(tdm_train,train_y)


# In[36]:


pred_nb=nb.predict(tdm_test)


# In[37]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[38]:


tab_nb=confusion_matrix(test_y,pred_nb)
tab_nb


# In[39]:


accuracy_score(test_y,pred_nb)


# In[ ]:





# 
# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;"> Logistic Regression  <p>
# 
# - Logistic regression models the relationship between a binary outcome variable and independent variables by estimating the probability of occurrence. It's favored for binary classification tasks due to its simplicity and effectiveness. Using a sigmoid function, it transforms the equation, enabling the modeling of probabilities and binary decisions.

# In[40]:


# from sklearn.linear_model import LogisticRegression
# logreg=LogisticRegression()


# In[41]:


# logreg.fit(tdm_train,train_y)


# In[42]:


# pred_log=logreg.predict(tdm_test)


# In[43]:


# tab_log=confusion_matrix(test_y,pred_log)
# tab_log


# In[44]:


# accuracy_score(test_y,pred_log)


# In[ ]:





# 
# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;">Decision Tree   <p> 
# 
# - Decision tree models the relationship between input features and a target variable by partitioning the feature space into segments, making binary decisions at each node. It's favored for its simplicity and interpretability, breaking down complex decision-making processes into a series of straightforward if-else conditions. By recursively splitting the data based on feature thresholds, it creates a tree-like structure, enabling clear visualization of decision paths and feature importance.

# In[45]:


# from sklearn.tree import DecisionTreeClassifier
# dec=DecisionTreeClassifier()


# In[46]:


# dec.fit(tdm_train,train_y)


# In[47]:


# pred_dec=dec.predict(tdm_test)


# In[48]:


# tab_dec=confusion_matrix(test_y,pred_dec)
# tab_dec


# In[49]:


# accuracy_score(test_y,pred_dec)


# In[ ]:





# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;">Random Forest   <p> 
# - Random Forest, an ensemble learning method, builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. By aggregating the results of individual trees, it enhances robustness and provides superior performance for classification and regression tasks.

# In[50]:


# from sklearn.ensemble import RandomForestClassifier
# rfc=RandomForestClassifier()


# In[51]:


# rfc.fit(tdm_train,train_y)


# In[52]:


# pred_rfc=rfc.predict(tdm_test)


# In[53]:


# tab_rfc=confusion_matrix(test_y,pred_rfc)
# tab_rfc


# In[54]:


# accuracy_score(test_y,pred_rfc)


# In[ ]:





# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;"> Most Frequent Words from all data</span>

# In[55]:


import matplotlib.pyplot as plt


# In[56]:


from wordcloud import WordCloud
cloud=WordCloud(stopwords=stopwords.words('english'),max_words=30).generate(str(med['x']))
plt.figure(figsize=(10,10))
plt.imshow(cloud)


# In[ ]:





# 
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Most Frequent Words from class 0 -->Thyroid_Cancer</span>

# In[57]:


med_spam_df0=med[med.y==0]


# In[58]:


from wordcloud import WordCloud
cloud=WordCloud(stopwords=stopwords.words('english'),max_words=40).generate(str(med_spam_df0['x']))
plt.figure(figsize=(10,10))
plt.imshow(cloud)


# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Most Frequent Words from class 1 -->Colon_Cancer</span>

# In[59]:


med_spam_df1=med[med.y==1]


# In[60]:


cloud=WordCloud(stopwords=stopwords.words('english'),max_words=40).generate(str(med_spam_df1['x']))
plt.figure(figsize=(10,10))
plt.imshow(cloud)


# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Most Frequent Words from class 0 --> Lung_Cancer</span>

# In[61]:


med_spam_df2=med[med.y==2]


# In[62]:


cloud=WordCloud(stopwords=stopwords.words('english'),max_words=40).generate(str(med_spam_df2['x']))
plt.figure(figsize=(10,10))
plt.imshow(cloud)


# In[ ]:





# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;">RNN  <p> 
# -  "RNNs are a type of neural network architecture designed to effectively process sequential data by retaining information about previous inputs. They achieve this by incorporating loops within their structure, allowing them to maintain a memory of past information while processing new input. This makes RNNs particularly suitablefor tasks such as time series prediction, natural language processing, and sequence generation.

# In[64]:


pd.read_csv(r"D:\Medical.csv",encoding='latin-1')


# In[65]:


med=pd.read_csv(r"D:\Medical.csv",encoding='latin-1')


# In[66]:


df=pd.DataFrame(med['0'])


# In[67]:


df['x']=med['a']


# In[68]:


df.rename(columns={'0':'y'},inplace=True)


# In[69]:


df.y=df.y.replace({'Thyroid_Cancer':0,'Colon_Cancer':1,'Lung_Cancer':2})


# In[70]:


df.head()


# In[71]:


df.y.nunique()


# In[72]:


df.isnull().sum()[df.isnull().sum()>0]


# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Spliting Data into Train and Test</span>
# </a>

# In[74]:


df_x=df.iloc[:,1]
df_y=df.iloc[:,0]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=.2)


# In[75]:


from tensorflow.keras.utils import to_categorical

# Assuming y_train and y_test are your class labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[76]:


max_num_words=8000 #from entire corpus sleect 10000 words
seq_len=100 # how many words out of 10000 you wish to take from each document
embedding_size=100 #vector length of each word


# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Sampling</span>
# </a>

# In[77]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[78]:


tokenizer=Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(df.x)
x_train=tokenizer.texts_to_sequences(x_train)
x_test=tokenizer.texts_to_sequences(x_test)


# In[79]:


x_train=pad_sequences(x_train,maxlen=seq_len)
x_test=pad_sequences(x_test,maxlen=seq_len)

model=Sequential()
model.add(Embedding(input_dim=max_num_words,
                   input_length=seq_len,
                   output_dim=embedding_size))


# In[80]:


model.add(SimpleRNN(128))
#model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))

from tensorflow.keras.optimizers import Adam
adam=Adam(learning_rate=.001)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Fit The Model</span>
# </a>

# In[81]:


model.fit(x_train,y_train,epochs=8,batch_size=32,validation_split=.2)


# In[82]:


pred_prob=model.predict(x_test)


# In[83]:


pred=pred_prob.argmax(axis=1)


# In[84]:


y_test=y_test.argmax(axis=1)


# In[85]:


confusion_matrix(y_test,pred)


# In[86]:


print(classification_report(y_test,pred))


# In[87]:


accuracy_score(y_test,pred)


# In[ ]:





# In[ ]:





# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;">LSTM  <p> 

# In[88]:


# "LSTM networks are a type of recurrent neural network (RNN) architecture designed to address the vanishing 
# gradient problem and capture long-term dependencies in sequential data. They achieve this by introducing 
# specialized memory cells with gating mechanisms, allowing them to selectively retain and update information 
# over multiple time steps. This enables LSTMs to effectively model complex sequential patterns, making them
# well-suited for tasks such as speech recognition, language translation, and time series forecasting."


# In[89]:


pd.read_csv(r"D:\Medical.csv",encoding='latin-1')


# In[90]:


med=pd.read_csv(r"D:\Medical.csv",encoding='latin-1')


# In[91]:


df=pd.DataFrame(med['0'])


# In[92]:


df['x']=med['a']


# In[93]:


df.rename(columns={'0':'y'},inplace=True)


# In[94]:


df.y=df.y.replace({'Thyroid_Cancer':0,'Colon_Cancer':1,'Lung_Cancer':2})


# In[95]:


df.head()


# In[96]:


df.y.nunique()


# In[97]:


df.isnull().sum()[df.isnull().sum()>0]


# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Spliting Data into Train and Test</span>
# </a>

# In[98]:


df_x=df.iloc[:,1]
df_y=df.iloc[:,0]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=.2)


# In[99]:


from tensorflow.keras.utils import to_categorical

# Assuming y_train and y_test are your class labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[100]:


max_num_words=8000 #from entire corpus sleect 10000 words
seq_len=100 # how many words out of 10000 you wish to take from each document
embedding_size=100 #vector length of each word


# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Sampling</span>
# </a>

# In[101]:


tokenizer=Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(df.x)
x_train=tokenizer.texts_to_sequences(x_train)
x_test=tokenizer.texts_to_sequences(x_test)


# In[102]:


x_train=pad_sequences(x_train,maxlen=seq_len)
x_test=pad_sequences(x_test,maxlen=seq_len)

model=Sequential()
model.add(Embedding(input_dim=max_num_words,
                   input_length=seq_len,
                   output_dim=embedding_size))


# In[103]:


model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))

from tensorflow.keras.optimizers import Adam
adam=Adam(learning_rate=.001)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Fit The Model</span>
# </a>

# In[104]:


model.fit(x_train,y_train,epochs=8,batch_size=32,validation_split=.2)


# In[105]:


pred_prob=model.predict(x_test)


# In[106]:


pred=pred_prob.argmax(axis=1)


# In[107]:


y_test=y_test.argmax(axis=1)


# In[108]:


confusion_matrix(y_test,pred)


# In[109]:


print(classification_report(y_test,pred))


# In[110]:


accuracy_score(y_test,pred)


# In[ ]:





# 
# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;">Bidirection LSTM  <p> 
# - BiLSTM networks extend the capabilities of traditional LSTMs by processing input sequences in both forward and backward directions. This allows them to capture context from past and future inputs simultaneously, enhancing their ability to understand and model complex dependencies in sequential data. BiLSTMs areparticularly effective for tasks such as sequence labeling, sentiment analysis, and machine translation, where bidirectional context is crucial for accurate prediction."

# In[112]:


pd.read_csv(r"D:\Medical.csv",encoding='latin-1')


# In[116]:


med=med.replace({'Thyroid_Cancer':0,'Colon_Cancer':1,'Lung_Cancer':2})
med.head()


# In[113]:


med=pd.read_csv(r"D:\Medical.csv",encoding='latin-1')


# In[114]:


med=med.loc[:,['0','a']]


# In[115]:


med=med.rename(columns={'0':'y','a':'x'})


# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Spliting Data into Train and Test</span>
# </a>

# In[117]:


med_x =  med.iloc[:,1]
med_y = med.iloc[:,0]
from sklearn.model_selection import train_test_split


# In[118]:


x_train, x_test, y_train, y_test = train_test_split(med_x, med_y, test_size=.2)


# In[119]:


y_train = to_categorical(y_train) # one hot endcoding


# In[120]:


max_num_words = 8000      # from the entire corpus select 10000 words
seq_len = 100               # how many words out of 10000 you wish to take from each document
embeddings_size = 100      # vector length of each word


# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Sampling</span>
# </a>

# In[121]:


tokenizer = Tokenizer(num_words = max_num_words)
tokenizer.fit_on_texts(med.x)
x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=seq_len)


x_test = pad_sequences(x_test, maxlen=seq_len)


# In[122]:


model = Sequential() # initialize the network
model.add(Embedding(input_dim= max_num_words,
                   input_length= seq_len,
                   output_dim = embeddings_size))


# In[123]:


model.add(Bidirectional(LSTM(65)))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))# Dense 2--> spam and ham


# In[124]:


from tensorflow.keras.optimizers import Adam
adam = Adam(learning_rate = .001)
model.compile(optimizer='adam', loss ='categorical_crossentropy', metrics =['accuracy'])


# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Fit The Model</span>
# </a>

# In[125]:


model.fit(x_train, y_train, epochs = 5, batch_size = 32, validation_split=.2)


# In[126]:


pred=model.predict(x_test)
pred


# In[127]:


pred_classes=pred.argmax(axis=1)


# In[128]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[129]:


tab=confusion_matrix(y_test,pred_classes)
tab


# In[130]:


accuracy_score(y_test,pred_classes)


# In[ ]:





# In[ ]:


#Compression of Models--------------->
l1=('Naive Bayes ','RNN','LSTM','Bidirection LSTM')
l2=(94.25,97.98.4197.95)
l3=(40,50,65,54,49)
l4=(84,90,71,27,84)
l5=(54,64,67,36,62)
importances=pd.DataFrame()
importances['Models']=l1
importances['Accuracy']=l2
importances['precision_score']=l3
importances['recall_score']=l4
importances['f1_score']=l5
importances


# In[ ]:





# <p style="background-color:purple; font-family:newtimeroman; color:#FFF9ED; font-size:200%; text-align:center; border-radius:20px; padding:20px;"><strong>Thank You :)</strong></p>
# 

# In[ ]:





# In[ ]:





# In[ ]:




