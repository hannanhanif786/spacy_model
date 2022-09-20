import spacy
import json
import random


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def test_model(model, text):
    doc = nlp(text)
    results = []
    entities = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))
    if len(entities) > 0:
        results = [text, {"entities": entities}]
        return results


nlp = spacy.load("hp_ner")
Train_data = []

with open("data/random_data.txt", "r") as f:
    text = f.read()
    receipts = text.split("Receipts")
    for receipt in receipts:
        receipt_num = receipt.split("\n")[0]
        receipt_num = receipt_num.strip()
        segments = receipt.split("\n")[2:]
        for segment in segments:
            segment = segment.strip()
            segment = segment.strip("\n")
            results = test_model(nlp, segment)
            if results != None:
                Train_data.append(results)

save_data("data/hp_training_data.json", Train_data)
