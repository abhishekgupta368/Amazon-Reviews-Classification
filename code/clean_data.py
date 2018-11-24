import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer

Im= Imputer()
if __name__ == '__main__':
    
    data = pd.read_csv("review.csv")
    title = pd.DataFrame(data['reviews.title'].values)
    review = pd.DataFrame(data['reviews.text'].values)
    temp = data['reviews.rating'].values
    
    temp = temp.reshape(-1,1)
    temp=Im.fit_transform(temp)
    
    temp = list(map(int,temp))
    temp1=np.array([])
    for i in range(len(temp)):
        temp1 = np.append(temp1,temp[i])
    rating = pd.DataFrame(temp1)
        
    n_data  = pd.concat([title,review,rating],axis=1)
    n_data.columns = ["title","review","rating"]
    n_data.to_csv("clean_review.csv")