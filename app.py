import pandas as pd 
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from  sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv(r'D:\Fake news detector\fake_or_real_news.csv')

df['News'] = df['title'] + ' ' + df['text']
df['News'] = df['News'].str.lower()
df['label'] = df['label'].str.lower()
df['News']=df['News'].apply(lambda x : re.sub(r'[^\w\s]','',x))
df['News'] = df['News'].str.strip() 

X = df.News
y = df.label

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
vectorizer = CountVectorizer(stop_words='english', min_df=5)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vectors,y_train)
prediction = model.predict(X_test_vectors)
# print("Predicted:",prediction)
# print("Actual:",y_test)
print("Accuracy:",accuracy_score(y_test, prediction))
print("Confusion Matrix:",confusion_matrix(y_test, prediction))

