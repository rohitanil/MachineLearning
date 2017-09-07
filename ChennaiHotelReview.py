import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

df=pd.read_csv('/Users/continuumlabs/Downloads/chennai_reviews.csv.zip')
cols=["Review_Title","Sentiment"]
df=df[cols]
df=df.dropna(axis=0, how='any')

train=df["Review_Title"]
label=df["Sentiment"]

X_train=train[:3500]
Y_train=label[:3500]

X_test=train[3501:4300]
Y_test=label[3501:4300]


vectorizer= CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

train_data_features=vectorizer.fit_transform(X_train)
train_data_features=train_data_features.toarray()

test_data_features=vectorizer.transform(X_test)
test_data_features=test_data_features.toarray()

print "Training Model"
forest=RandomForestClassifier(n_estimators=100)
forest=forest.fit(train_data_features,Y_train)
print "Testing Model"
prediction=forest.predict(test_data_features)
accuracy=np.mean(prediction==Y_test)
print "Accuracy of Random Forest Model:"
print accuracy

X_validation=train[4001:4002]
validation_data_features=vectorizer.transform(X_validation)
validation_data_features=validation_data_features.toarray()

predict1=forest.predict(validation_data_features)
print "Data: "+ X_validation
print "Sentiment Prediction:"+ predict1
