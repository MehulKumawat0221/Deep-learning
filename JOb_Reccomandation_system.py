"""Job_Recommendation_System"""

import numpy as np
import pandas as pd

from google.colab import files
path = list(files.upload().keys())[0]

df = pd.read_csv(path)

df.head()

df = df.sample(5000, ignore_index=True)

import string

panch = string.punctuation
panch

def clean_text(text):
  mylist = []
  for i in text.split():
    if i not in panch:
      mylist.append(i)
  return " ".join(mylist)

df['Job Title'] = df['Job Title'].apply(clean_text)

df['Location'] = df['Location'].apply(clean_text)

df['Company Name'] = df['Company Name'].astype(str)

df['Company Name'] = df['Company Name'].apply(clean_text)

df['Content'] = df.apply(lambda x:" ".join(x.dropna().astype(str)) , axis=1)

df

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)

metrics = cv.fit_transform(df['Content']).toarray()

metrics.shape

similarity = cosine_similarity(metrics)

def recommend(Job_Title):
  index = df[df['Job Title'] == Job_Title].index[0]
  distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x:x[1])
  new_index = [i[0] for i in distances[1:11]]
  return df.iloc[new_index , :-1]

df.sample(5)

recommend('Academic Tutor')

recommend('Career Leader')