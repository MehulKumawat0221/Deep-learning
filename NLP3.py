df

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
tfidf.fit_transform(df['text']).toarray()

print(tfidf.idf_)
print(tfidf.get_feature_names_out())

!pip install spacy
!python -m spacy download en_core_web_sm

import spacy

# Load English NER model
nlp = spacy.load("en_core_web_sm")

# Example text
text = "Apple Inc. was founded by Steve Jobs in Cupertino in 1976. Sundar Pichai is the CEO of Google."

# Process text
doc = nlp(text)

# Print entities
for ent in doc.ents:
    print(f"{ent.text:<20} | {ent.label_}")

from spacy import displacy

# Render visualization inside Jupyter Notebook
displacy.render(doc, style="ent", jupyter=True)

sentences = [
    "Elon Musk founded SpaceX in 2002 and Tesla in Palo Alto.",
    "Apple Inc. was founded by Steve Jobs in Cupertino in 1976.",
    "Sundar Pichai is the CEO of Google."
]

for sentence in sentences:
    doc = nlp(sentence)
    print(f"\nProcessing sentence: '{sentence}'")
    for ent in doc.ents:
        print(f"{ent.text:<20} | {ent.label_}")

"""# google trans

"""

!pip install googletrans==4.0.0-rc1

from googletrans import Translator
translator = Translator()

# Example text
text = "Hello, how are you?"

# Translate to Hindi
result = translator.translate(text, src="en", dest="hi")

print("Original:", result.origin)
print("Translated:", result.text)
print("Source Language:", result.src)
print("Destination Language:", result.dest)

sentences = [
    "Good morning!",
    "I love programming.",
    "Data Science is the future."
]

for s in sentences:
    translated = translator.translate(s, src="en", dest="hi")  # English → hindi
    print(f"{s}  -->  {translated.text}")

text = "mera naam Yash hai, main jaipur mein rehta hoon , how are you"


result_hi = translator.translate(text, dest="hi")
print("Original:", result_hi.origin)
print("Detected Language:", result_hi.src)
print("Translated to Hindi:", result_hi.text)


result_en = translator.translate(text, dest="en")
print("Translated to English:", result_en.text)

texts = [
    "hi aap kaise ho , its been so long , lets meet , jb free ho?",
    "mera naam Yash hai aur main Delhi mein rehta hoon",
    "Sundar Pichai is the CEO of Google",
    "Steve Jobs founded Apple in 1976",
    "Elon Musk founded SpaceX and Tesla"
]

for i, text in enumerate(texts, 1):
    print(f"\n--- Sentence {i} ---")
    print("Original:", text)

    # 1️⃣ Translate to Hindi
    translated_hi = translator.translate(text, dest="hi")
    print("Translated to Hindi:", translated_hi.text)

    # 2️⃣ Translate to English
    translated_en = translator.translate(text, dest="en")
    print("Translated to English:", translated_en.text)

    # 3️⃣ NER using spaCy (on original text)
    doc = nlp(text)
    print("Named Entities:")
    if doc.ents:
        for ent in doc.ents:
            print(f"  {ent.text:<20} | {ent.label_}")
    else:
        print("  No entities detected")

    # Optional: visualize entities in notebook
    # displacy.render(doc, style="ent", jupyter=True)

