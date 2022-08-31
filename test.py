import spacy
import json

data_list = []
with open("file_model", "r") as f:
    data_list = json.load(f)[:1]
    data = data_list[0]["raw_text"]

str_data = ""

check = False

for appnd in data:
    str_data = str_data + " " + appnd
print(str_data)
nlp = spacy.load("en_core_web_lg")
doc = nlp(str_data)
for ent in doc.ents:
    if ent.label_ == "DATE":
        print(ent.text, ent.label_)
