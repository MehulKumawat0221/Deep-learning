
import numpy as np
import pandas as pd



df.head()

df['review'][3]

"""##step1 Lowercase"""

df['review'][3].lower()

df['review'] = df['review'].str.lower()

df

import re
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'',text)

text = "<html><body><p> Movies </p><p>Click here to <a herf = 'http://google.com'>download</a></p>"

remove_html_tags(text)

df['review'] = df['review'].apply(remove_html_tags)

df

def remove_url(text):
  pattern = re.compile(r'ttps?://\S+|www.\.\S+')
  return pattern.sub(r'', text)

text4 = 'For notebook click https://wwww.deeture.cpm/delta/note1234acc'

remove_url(text4)

import string
string.punctuation

exclude = string.punctuation

def remove_punc(text):
    for char in exclude:
      text = text.replace(char, '')
    return text

text1 = 'String. with. punctuation?'

remove_punc(text1)

df['review'] = df['review'].apply(remove_punc)

print(remove_punc(text))

chat_words = {'u2':'you too', 'BBL':'Be Back Later', 'GAL':'Get A Life'}

chat_words

def chat_conversation(text):
  new_text = []
  for w in text.split():
    if w.upper() in chat_words:
      new_text.append(chat_words[w.upper()])
    else:
      new_text.append(w)

    return " ".join(new_text)

chat_conversation('BBL')

"""#step-6 Spelling Correction"""

!pip install TextBlob

from textblob import TextBlob

incorrect_text = 'ceertain conditionas duriing seveal ggenerations aree moodified in the saame maner.'
textBlb = TextBlob(incorrect_text)
textBlb.correct().string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

from types import new_class
def remove_stopwords(text):
  new_text = []
  for word in text.split():
    if word in set(stopwords.words('english')):
      new_text.append('')
    else:
      new_text.append(word)
  x = new_text[:]
  new_text.clear()
  return "".join(x)

remove_stopwords('probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it\'s not preachy or boring. it just never gets old, despite my having seen it some 15 or more times')

import re
def remove_emoji(text):
  emoj_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"/U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "] +" , flags = re.UNICODE)

!pip install emoji

import emoji
print(emoji.demojize('Python is 👍'))

"""#tokenization"""

# 1. prefix - character(s) at the begning = $(".)
# 2. suffix - - character(s) at the end = km).!"
# 3. Infix - character(s) in between = --/...
# 4. Exception - Special-case rule to splite a strig into several tokens or
# prevent a token from being split when puncctuation rule are applied = Let's

#Approch the tokenizetion using the split function
# 1. word token.
sent1 = "I am goning to delhi"
sent1.split()

# sentence token.
sent2 = "I am going to delhi . I will stay there for 3 days. Let\'s hope the trip to be great"
sent2.split('.')

# problem with split function
sent3 = " I am going to delhi!"
sent3.split()

# in the sent3 problem, there is seeing that ! is included with delhi. so, we have to remove it by regular expression.

import re
sent3 = "I am going to delhi"
tokens = re.findall("[\w']+", sent3)
sent3

"""1. NLTK"""

from nltk.tokenize import word_tokenize , sent_tokenize

import nltk
nltk.download('punkt')

import nltk
nltk.download('punkt_tab')

sent1 = "I am going to visite delhi"
word_tokenize(sent1)

sent2 = "I have a Ph.D in A.I"
sent3 = "We're here to help! mail us at nks@gmail.com"
sent4 = "A 5km ride cost $10.50"

word_tokenize(sent2)

word_tokenize(sent3)

word_tokenize(sent4)
