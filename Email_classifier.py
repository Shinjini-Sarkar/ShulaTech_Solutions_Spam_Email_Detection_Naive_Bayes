import os
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('punkt_tab')
spam_folder_path="C:\\Users\\shinj\\Spam_mail_classifier\\spam_zipped"
ham_folder_path="C:\\Users\\shinj\\Spam_mail_classifier\\ham_zipped"

#combine the 2 folders obtained from the zipfile of the email dataset
def load_data_from_folder(folder,label):
    emails=[]
    for filename in os.listdir(folder):
        filepath=os.path.join(folder,filename)
        file=open(filepath,"r",encoding="latin-1")
        emails.append({"message":file.read(),"label":label})
    return emails
    
#program for preprocessing data
def preprocess_text(text):
    text=text.lower()    
     #Remove URLs
    text=re.sub(r'http\S+|www\S+','',text)
     #Remove HTML tags
    text=re.sub(r'\d+','',text)
    text=text.translate(str.maketrans('','',string.punctuation))
    #Tokenization
    words=nltk.word_tokenize(text)
    #Remove stopwords
    words=[word for word in words if word not in stop_words]
    #return a list of words
    return ' '.join(words)

#testing model with new emails
def predict_spam(email_text):
    email_text=preprocess_text(email_text)
    email_vector=vectorizer.transform([email_text])
    #model.predict(data)always returns an array even if there is only one value
    prediction=model.predict(email_vector)
    if prediction[0]==1:
       return "Spam"
    else:
       return "Not Spam"
    

spam_emails=load_data_from_folder(spam_folder_path,0)
ham_emails=load_data_from_folder(ham_folder_path,1)
df = pd.DataFrame(spam_emails + ham_emails)
df.sample(frac=1,random_state=42).reset_index(drop=True)
#extracting all stopwords and store them as a set
stop_words=set(stopwords.words('english'))
# Apply preprocessing function to the 'message' column of the df dataframe already created
df['cleaned_message']=df['message'].apply(preprocess_text)
print(df.head())
df.info()

#feature extraction using TF-IDF Vectorization, assigns values in terms of importance of words

vectorizer=TfidfVectorizer(max_features=7000)

#Transform into a form where numerical values are assigned to words as per their importance in a message
X=vectorizer.fit_transform(df['cleaned_message'])
y=df['label']

#split 70% train and 30%test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

#create a Naive Bayes model using the Multinomial Naive Bayes (MultinomialNB) algorithm.
model=MultinomialNB()
model.fit(X_train,y_train)
#make prediction using test data
y_pred=model.predict(X_test)
#Evaluate the model after training
accuracy=accuracy_score(y_test,y_pred)
print(f"Model Accuracy:{accuracy:.2f}")
#print the classification report
print("\nClassification Report:\n",classification_report(y_test,y_pred))
print("\nConfusion matrix\n",confusion_matrix(y_test,y_pred))

#passing mails to the predict_spam method
email=input("Enter your mail to check if it is spam or not:\n")
print(f"Email : {predict_spam(email)}")




    
    


