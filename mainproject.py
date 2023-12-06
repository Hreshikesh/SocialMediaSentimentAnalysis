import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

# Downloading the stopwords
nltk.download('stopwords')

# Tokenization
tokenizer=RegexpTokenizer(r"\w+")
en_stopwords=set(stopwords.words('english'))
ps=PorterStemmer()

def getCleanedText(text):
    text=text.lower()
    tokens=tokenizer.tokenize(text)
    new_tokens=[token for token in tokens if token not in en_stopwords]
    stemmed_tokens=[ps.stem(token) for token in new_tokens]
    clean_text=" ".join(stemmed_tokens)
    return clean_text

# Training data
X_train=["This was really awesome an awesome movie",
           "Great movie! I likes it a lot",
           "Happy Ending! Awesome Acting by hero",
           "loved it!",
           "Bad not upto the mark",
           "Could have been better",
           "really Disappointed by the movie"]
y_train=["positive", "positive", "positive", "positive", "negative", "negative", "negative"]

# Convert y_train to a Pandas Series
y_train_series=pd.Series(y_train)

# Test data
X_test=["it was bad"]

# Text cleaning
X_clean=[getCleanedText(text) for text in X_train]
Xt_clean=[getCleanedText(text) for text in X_test]

# Vectorization
tfidf=TfidfVectorizer(ngram_range=(1, 2))
X_vec_tfidf=tfidf.fit_transform(X_clean)
Xt_vec_tfidf=tfidf.transform(Xt_clean)

# Support Vector Machine classifier
svm=SVC(kernel='linear', C=1)
svm.fit(X_vec_tfidf, y_train)

# Predictions
y_pred_train=svm.predict(X_vec_tfidf)

# Evaluating the model on the training set
print("\nTraining Accuracy :", accuracy_score(y_train, y_pred_train))

# Classification report
print("\nClassification Report:")
print(classification_report(y_train, y_pred_train))

# Confusion Matrix
cm =confusion_matrix(y_train, y_pred_train)

# Bar Graph for Confusion Matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x=['Negative','Positive'], y=cm[0], label='True Negative', color='red', ax=ax1)
sns.barplot(x=['Negative', 'Positive'], y=cm[1], label='True Positive', color='blue', ax=ax1)
ax1.set_title('Confusion Matrix Bar Graph')
ax1.set_xlabel('Actual Sentiment')
ax1.set_ylabel('Count')
ax1.legend()

# Pie Chart 
sentiment_distribution=y_train_series.value_counts()  
ax2.pie(sentiment_distribution, labels=sentiment_distribution.index, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen'])
ax2.set_title('Distribution of Sentiments in Training Set')
plt.show()

# Final prediction 
fp= svm.predict(Xt_vec_tfidf)
print("\nFinal Predicted Sentiment:", fp)
