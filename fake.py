from faker import Faker
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example


fake = Faker()
fake_data = []
for i in range(5):
    fake_data.append(
        f"orderdate for this product is orderdate : {fake.date()} we delivered this product at deliverydate : {fake.date()} the due date for shipping is duedate : {fake.date()} "
    )
TRAIN_DATA = [
    (
        "orderdate for this product is orderdate : 2016_04_09 we delivered this product at deliverydate : 2015_04_10 ",
        {"entities": [[29, 50, "ORDERDATE"], [78, 103, "DELIVERDATE"]]},
    ),
    #   ("the due date for shipping is duedate : 1974-05-19", {"entities": [(19, 28, "GPE")]}),
    #   ("I recently ordered a book from Amazon", {"entities": [(24,32, "ORG")]}),
]

nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")

for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]


# TRAINING THE MODEL
with nlp.disable_pipes(*unaffected_pipes):

    # Training for 30 iterations
    for iteration in range(10):

        # shuufling examples  before every iteration
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        # batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))

        for batch in spacy.util.minibatch(TRAIN_DATA, size=2):
            for text, annotations in batch:
                # create Example
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update the model
                nlp.update([example], losses=losses, drop=0.3)
                print("Losses", losses)

        # print(batches)
        # for batch in batches:
        #     print(batch)
        #     texts, annotations = zip(*batch)
        #     text = list(texts)
        #     text = text[0]
        #     doc = nlp.make_doc(text)
        #     print()
        #     example = Example.from_dict(doc, annotations)
        #     print(type(example))

        #     nlp.update(
        #         example,
        #         drop=0.5,  # dropout - make it harder to memorise data
        #         losses=losses,
        #     )
        #     print("Losses", losses)

doc = nlp(fake_data[0])
print(
    "orderdate for this product is orderdate : 1986-01-17 we delivered this product at deliverydate : 1984-08-30 the due date for shipping is duedate : 2002-06-17 "
)
for ent in doc.ents:
    # if ent.label_ == "DATE":
    print(ent.text, ent.label_)
