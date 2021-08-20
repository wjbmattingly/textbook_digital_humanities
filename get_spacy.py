import spacy
nlp = spacy.load("en_core_web_sm")
nlp.to_disk("en_core_web_sm")
