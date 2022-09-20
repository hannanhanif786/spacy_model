import spacy
import json
import random
from spacy.training import Example


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def train_spacy(data, iteration):
    Train_Data = data
    nlp = spacy.blank("en")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe("ner")
    for _, annotations in Train_Data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iteration):
            random.shuffle(Train_Data)
            losses = {}
            for text, annotations in Train_Data:
                # create Example
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update the model
                nlp.update([example], losses=losses, drop=0.225)
    return nlp


train_data = load_data("data/hp_training_data.json")
nlp = train_spacy(train_data, 35)

nlp.to_disk("hp_ner_model")

test = "Account Code : 123456789 OrderDate : 11/12/2022 Email : check@gmail.com Phone no : +92300321456 Order # : 0000200 Invoice Date : 11-07-2022 DueDate : 30-07-2022  "

nlp = spacy.load("hp_ner_model")
doc = nlp(test)
for ent in doc.ents:
    print(ent.text, ent.label_)
