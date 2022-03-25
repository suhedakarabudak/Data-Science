
import numpy as np
import pandas as pd
import os


yorumlar=pd.read_csv('C:/Users/Şuheda/Desktop/mlogr/restaurant_reviews.csv',sep=';')
print(yorumlar)
import re
import nltk
from nltk.stem.porter import PorterStemmer 
ps=PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

#preprocessing (önişleme)
derlem=[]
for i in range(1000):
    yorum=re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum=yorum.lower()
    yorum.split()
    yorum=[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum=''.join(yorum)
    derlem.append(yorum) 

#Feature Extraction (öznitelik çıkarımı)
#Bag of Word (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
X=cv.fit_transform(derlem).toarray() #bağımsız değişken
y=yorumlar.iloc[:,1].values #bağımlı değişken

#makine öğrenmesi
from sklearn.model_selection import train_test_split
X_train , X_test, y_train ,y_test=train_test_split(X,y, test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB 
gnb=GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)

from sklearn.metrics import confusion_matris
cm=confusion_matris(y_test,y_pred)
print(cm)   