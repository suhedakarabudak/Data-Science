import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("weight-height.csv")
data.head()

data.isnull().sum()

data.describe()

#Aykırı değerlerin tespiti için box-plot grafiği çizdirelim.
plt.figure(figsize = (4,8))
sns.boxplot(y = data.Height)


def out_iqr(df , column):
    global lower,upper
    q25, q75 = np.quantile(df[column], 0.25), np.quantile(df[column], 0.75)
    #IQR
    iqr = q75 - q25
    # outlier cutoff
    cut_off = iqr * 1.5
    #Üst ve alt değer belirleme
    lower, upper = q25 - cut_off, q75 + cut_off
    print('Çeyrekler Arası Aralık (IQR) ',iqr)
    print('Altsınır değerleri', lower)
    print('Üstsınır değeri', upper)
    #alt sınır değerin altındaki ve üstündeki değerlerin sayısını hesaplayın
    df1 = df[df[column] > upper]
    df2 = df[df[column] < lower]
    return print('Aykırı değerlerin toplam sayısı:', df1.shape[0]+ df2.shape[0])

out_iqr(data,"Height")

#Z-skor Metodu
plt.figure(figsize = (10,5))
sns.distplot(data['Height'])

def out_zscore(data):
    global outliers,zscore
    outliers = []
    zscore = []
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    for i in data:
        z_score= (i - mean)/std
        zscore.append(z_score)
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return print("Aykırı değerlerin toplam sayısı:",len(outliers))

#Kırmızı bölge aykırı değerlerin bulunduğu kısımdır
plt.figure(figsize = (10,5))
sns.distplot(zscore)
plt.axvspan(xmin = 3 ,xmax= max(zscore),alpha=0.2, color='red')
  
