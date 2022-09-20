from tabnanny import check
import spacy
from spacy.lang.en import English
from spacy.pipeline import entityruler
import json
from train_data import create_training_data

data_list = []


def generate_rules(patterns):
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler")
    for i in patterns:
        for x in i:
            ruler.add_patterns(x)
    nlp.to_disk("hp_ner")


pattern = []
pattern.append(create_training_data("data/fake.json", "ORDERDATE"))
pattern.append(create_training_data("data/fake1.json", "DUEDATE"))


generate_rules(pattern)


# nlp = spacy.load("hp_ner")
#     "OrderDate : Jun 11, 2022 was the order data and shipping date is  Jun 11 2022. delieverdate 4/15/2022 was the 44th president of the United States Jun 11 2022"
# )
# doc = nlp(
#     "orderdate for this is orderdate : 11/10/2018 WinsPort LOGISTICS INVOICE WinsPort Logistics Load #: 28 6700 Alexander Bell Drive, 200 Customer Load #: 098431 Columbia, Maryland 21046 Customer ID #: winsportlogistics.com 10 PO / Order #: 2864894 Phone: (301)798-6130,  OrderDate : 12-11-2022  Email: cmartin@winsportlogistics.com duedate : 22-32-2011 - CUSTOMER INFORMATION DMT Trucking Phone: +1 (773) 309-2111 Mount Rainier 3408 Rhode Island Ave, Mt Rainier, MD 20712, USA, Maryland 20712 Contact: Kim Bay PAY ITEMS Description Linked Invoice Notes Quantity Rate Amount Line Haul Charge 1 $3,000.00 $3,000.00 TOTAL $3,000.00 PICKUP #1 Test 2 153-01 Jamaica Ave, Jamaica, NY 11432, USA Date/Time Contact Name Contact Phone References Weight Packaging Jun 16, 2022 90 67,000 lbs Floor Loaded 13:35 Commodity Pickup Notes High Value cans DROPOFF #1 Test 1 135 Madison Ave, New York, NY 10016, USA Date/Time Contact Name Contact Phone References Weight Packaging Jun 17. 2022 67,000 lbs Floor Loaded 13:35 Commodity Dropoff Notes High Value cans Payment Instructions Remit Checks To: WinsPort Logistics 6700 Alexander Bell Drive, Columbia, Maryland, 21046 Services will be invoiced in accordance with the Service Description. You must pay all undisputed invoices in full within 12 days of the invoice date, unless otherwise specified under the Specified Terms and Conditions. All payments must reference the invoice number. Unless otherwise specified, all invoices shall be paid in the currency of the invoice"
# )
doc = nlp(
    " Account Code : 123456789 OrderDate : 11-07-2022 Email : check@gmail.com Phone no : +92300321456 Order # : 0000200 Invoice Date : 11-07-2022 DueDate : 30-07-2022"
)

for ent in doc.ents:
    print(ent.text, " ", ent.label_, ent.start_char)
