import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk 
import re 
import pickle

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
#----------------------------------------------------
ps = PorterStemmer()
def clean_data(review):
    l=[]
    for i in review:
        rev = i.lower()
        rev = re.sub("./|!':;$%^&*@_-*"," ",rev)
        rev = word_tokenize(rev)
        rev = [ps.stem(r) for r in rev]
        rev = [r for r in rev if not r in stopwords.words("english")]
        rev = " ".join(rev)
        l.append(rev)
    return l
def pick(rate):
    dic = open("review.pickle","wb")
    pickle.dump(rate,dic)
    dic.close()
def load_data():
    pick_in =open("review.pickle","rb")
    rev = pickle.load(pick_in)
    return rev

def testing_algorithm(x_train,x_test,y_train,y_test):
    svc = SVC(kernel='rbf',degree=2)
    svc.fit(x_train,y_train)
    algo1 = str(svc.score(x_train,y_train))

    knn = KNeighborsClassifier(n_neighbors=50,leaf_size=50)
    knn.fit(x_train,y_train)
    algo2 = str(knn.score(x_train,y_train))

    dtc = DecisionTreeClassifier(max_depth=50,min_samples_split=5)
    dtc.fit(x_train,y_train)
    algo3 = str(dtc.score(x_train,y_train))
 
    rfc = RandomForestClassifier(max_depth=50,min_samples_split=5)
    rfc.fit(x_train,y_train)
    algo4 = str(rfc.score(x_train,y_train))
  
    adb = AdaBoostClassifier(n_estimators=80,learning_rate=1)
    adb.fit(x_train,y_train)
    algo5 = str(adb.score(x_train,y_train))
    
    res=str("SVC: ")+algo1+str(" KNeighborsClassifier: ")+ algo2+str(" DecisionTreeClassifier: ")+algo3+str(" RandomForestClassifier: ")+algo4 +str(" AdaBoostClassifier: ")+algo5
    return res
#---------------------------------------------------
if __name__ == '__main__':
    data = pd.read_csv("clean_review.csv")
    review = data['review'].values
    rating = data['rating'].values.reshape(-1,1)
    #-----------------------------------------
#    rate= clean_data(review)
#    pick(rate)
    rev = load_data()
    x_train,x_test,y_train,y_test = train_test_split(rev,rating,test_size=0.2,random_state=1)
    tfv = TfidfVectorizer()
    x_train = tfv.fit_transform(x_train)
    x_test = tfv.transform(x_test)
    a= testing_algorithm(x_train,x_test,y_train,y_test)
    
    knn = KNeighborsClassifier(n_neighbors=5,leaf_size=50)
    knn.fit(x_train,y_train)
    algo2 = knn.score(x_train,y_train)
    query = str(input("Enter input: "))
    loc = np.array([query])
    msg = tfv.transform(loc.ravel())
    pred = knn.predict(msg)
    
    if(pred[0]==1):
        print("It is worst product")
    elif(pred[0]==2):
        print("It can be impoved")
    elif(pred[0]==3):
        print("It is Improving")
    elif(pred[0]==4):
        print("It is good product")
    elif(pred[0]==5):
        print("It is best product")
    else:
        print("System cannot understand your query")