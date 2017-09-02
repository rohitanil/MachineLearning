
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection

df= pd.read_csv('/Users/continuumlabs/Downloads/FullData.csv.zip')

cols=['GK_Positioning','GK_Diving', 'GK_Kicking', 'GK_Handling', 'GK_Reflexes']

df['GK_Positioning']=df['GK_Positioning'].fillna(df['GK_Positioning'].mean())
df['GK_Diving']=df['GK_Diving'].fillna(df['GK_Diving'].mean())
df['GK_Kicking']=df['GK_Kicking'].fillna(df['GK_Kicking'].mean())
df['GK_Handling']=df['GK_Handling'].fillna(df['GK_Handling'].mean())
df['GK_Reflexes']=df['GK_Reflexes'].fillna(df['GK_Reflexes'].mean())

train=df[cols]
pos=['GK' if value=='GK' else 'not GK' for value in df.Club_Position]
df['pos']=pos
label=df['pos']

Y=label[:1000].values
X=train[:1000].values

X_train, Y_train, X_test, Y_test=model_selection.train_test_split(X,Y,test_size=0.20, random_state=0)

clf1=GaussianNB().fit(X,Y)

X_valid=train[1001:1500].values
Y_valid=label[1001:1500].values

score=clf1.score(X_valid, Y_valid)
test=[75,78,77,34,88]
pred=clf1.predict(test)



print "Prediction",pred
print "Score",score
