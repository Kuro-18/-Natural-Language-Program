import nltk
from textblob import TextBlob
import spacy

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load("en_core_web_sm")

text = """
OpenAI has developed GPT-5, an advanced AI language model. 
It can perform tasks such as writing, summarization, and question answering.
"""

from nltk.tokenize import word_tokenize, sent_tokenize

sentences = sent_tokenize(text)
for i, sentence in enumerate(sentences, 1):
    print(f"{i}: {sentence}")

words = word_tokenize(text)
print(words)

pos_tags = nltk.pos_tag(words)
print(pos_tags)

blob = TextBlob(text)
print(f"Polarity: {blob.sentiment.polarity}, Subjectivity: {blob.sentiment.subjectivity}")

doc = nlp(text)
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
