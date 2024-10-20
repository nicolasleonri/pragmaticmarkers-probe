import re

def normalize(sentence):
    sentence = sentence.lower()
    # sentence = sentence1.replace(" .", "")
    sentence= re.sub(r"[^a-zA-Z0-9 ]+", '', sentence) # Remove special characters
    return sentence