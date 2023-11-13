import pandas as pd
import datetime
from mlxtend.frequent_patterns import apriori, association_rules

df=pd.read_csv("/home/suhedata/Desktop/armut_data.csv")
df.head()

#ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluştur
df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]


Year= 2017
Month =8
df["New_Date"] = pd.to_datetime(df.assign(Day=1, Year=Year, Month=Month)[["Year", "Month", "Day"]])

df["SepetID"]= df["UserId"].astype(str) + '_' + df["New_Date"].astype(str)

# Pivot table oluştur
pivot_table = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

pivot_table.head()

frequent_itemsets = apriori(pivot_table, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

def arl_recommender(rules_df,product_id,rec_count =1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    #Kuralları lifte göre büyükten küçüğe sıralar.(en uyumlu ilk ürünü yakalayabilmek için)
    recommendation_list = [] # tavsiye edilecek ürünler için bos bir liste olusturuyoruz.
    # antecedents: X
    #items denildigi için frozenset olarak getirir. index ve hizmeti birleştirir.
    # i: index
    # product: X yani öneri isteyen hizmet
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product): # hizmetlerde(product) gez:
            if j == product_id:# eger tavsiye istenen ürün yakalanırsa:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
                # index bilgisini i ile tutuyordun bu index bilgisindeki consequents(Y) değerini recommendation_list'e ekle.
    # tavsiye listesinde tekrarlamayı önlemek için:
    # mesela 2'li 3'lü kombinasyonlarda aynı ürün tekrar düşmüş olabilir listeye gibi;
    # sözlük yapısının unique özelliginden yararlanıyoruz.
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count] # :rec_count istenen sayıya kadar tavsiye ürün getir.
#örnek
arl_recommender(rules,'2_0',rec_count =1)

