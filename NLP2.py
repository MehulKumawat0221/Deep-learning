
"""Stemming"""

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def stem_words(text):
  return " ".join([ps.stem(word) for word in text.split()])

sample = "walk walks walking walked"
stem_words(sample)

# stemming mapping a group pf word to the same stem even if the stem itself
# is not valid word in Language. So, this is the big issue of stemming.
text = 'propbably my alltime fav movie a story of selflessness sacrifice and dedicaton'
print(text)

stem_words(text)

#So, we have to use Lemmatization insted of stemming.
# note - stemming is fate and Lemmatiztion is slow.



from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('wordnet')

wordnet_lemmatizer =  WordNetLemmatizer()

sentence = 'He was running and eating at the same time . He has bad habbit of swimming after playing long hours in the sun'
punctuation = '?:!.,;'
sentence_words = nltk.word_tokenize(sentence)
for word in sentence_words:
    sentence_words.remove(word)

sentence_words
print("{0:20}{1:20}".format("Word", "Lemm "))
for word in sentence_words:
    print("{0:20}{1:20}".format(word, wordnet_lemmatizer.lemmatize(word)))

"""# Text Representation"""

import numpy as np
import pandas as pd

df = pd.DataFrame({'text':['people watch cricket' , 'player was player' ,
                           'people write comment',
                           "cricket play coment" ] , 'output':[1,1,0,0]})

df

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

bow = cv.fit_transform(df['text'])

#vocab
print(cv.vocabulary_)

print(bow[0].toarray())
print(bow[1].toarray())

cv.transform(["cricket watch and write comment of crficket comment"]).toarray()

# 2.N-grams

# cv = CountVectorizer(ngram_range(2,2))

v = CountVectorizer(ngram_range=(1,2))
# v = CountVectorizer(ngram_range(2,2))

bow = v.fit_transform(df['text'])

df['text']

#vocab
print(v.vocabulary_)

# TF(Term Frequencey)
# 2. IDF(Inverse Documnet Frequency)

#TF(t,d) = (Number of Occurences of term T in document d)/(total number of term in document d)
# example =>
# d1 = people watch cicket.
#tf(people,d1) = 1/3
#d2 = match was match.
#tf(match, d2) = 2/3

#IDF(t) = Loge(Total number of documnets in the corpus )/(number of documnets with term t in them)
# here corpus means number of sentences

#example =>
#d1 = people watch cricket.
#d2 = cricket was cricket.
#d3 = people write comment.
#d4 = cricket write comment.

#idf(cricket) = Log(4/4)
#idf(people) = Log(4/2)

# we can use this by apply feature_extraction method in sklearn

