import pandas as pd
from sklearn.naive_bayes import GaussianNB
df=pd.read_csv('/Users/continuumlabs/Downloads/FullData.csv.zip')
label=df["Club_Position"]
cols=["Rating","Penalties","Volleys","GK_Positioning"]
train=df[cols]

x=df[cols].values
y=label.values
x=x[:100]
y=y[:100]
gnb=GaussianNB().fit(x,y)
test=[92,80,71,21]
gnb_predict=gnb.predict(test)
print "Predicted Position:"
print gnb_predict
