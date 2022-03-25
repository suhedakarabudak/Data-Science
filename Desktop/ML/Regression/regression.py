import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




#veri önişleme
eksik_veri=pd.read_csv('C:/Users/Şuheda/Desktop/mlogr/eksikveri.csv',sep=";")
print(eksik_veri)

#sci -kit learn
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

Yas=eksik_veri.iloc[:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)

#encoder kategorik->numeric
ulke=eksik_veri.iloc[:,-1:].values
print(ulke)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()
c[:,0]=le.fit_transform(eksik_veri.iloc[:,0])
print(ulke)
#Kolon başlıklarını etiketleri taşıma
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke) 
#nump dizilerin dataframe dönüşümü
sonuc=pd.DataFrame(data=ulke,index=range(22), columns=['fr','tr','us'])
print(ulke)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=eksik_veri.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)
#datafeamleri birleştirme
s=pd.concat([sonuc,sonuc2],axis=1) #dataframleri birleştirmek için concat komutu kulanılır 
print(s)
s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#verilerimi bölme işleme

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#öznitelik ölçekleme
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
print(X_test)

#çoklu regresyon

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('C:/Users/Şuheda/Desktop/data.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)


ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#encoder: Kategorik -> Numeric
c = veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)


ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)



#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)



#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)



