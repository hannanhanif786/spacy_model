import json


def load_data(file):
    with open(file, "r") as f:
        data = json.load(f)
        return data


def create_training_data(file, type):
    data = load_data(file)
    patterns = []
    lab_list = []
    for item in data:
        lab = [
            {"LOWER": item},
            {"IS_PUNCT": True, "OP": "?"},
            {"TEXT": {"REGEX": r"^\d{1,2}/\d{1,2}/\d{2}(?:\d{2})?$"}},
        ]
        pattern = {"label": type, "pattern": lab}
        patterns.append([pattern])  # MM-DD-YYYY and YYYY-MM-DD
        lab = [
            {"LOWER": item},
            {"IS_PUNCT": True, "OP": "?"},
            {"IS_DIGIT": True},
            {"ORTH": "-"},
            {"IS_DIGIT": True},
            {"ORTH": "-"},
            {"IS_DIGIT": True},
        ]
        pattern = {"label": type, "pattern": lab}
        patterns.append([pattern])

        # dates of the form 10-Aug-2018
        lab = [
            {"LOWER": item},
            {"IS_PUNCT": True, "OP": "?"},
            {"IS_DIGIT": True},
            {"ORTH": "-"},
            {"IS_ALPHA": True},
            {"ORTH": "-"},
            {"IS_DIGIT": True},
        ]
        pattern = {"label": type, "pattern": lab}
        patterns.append([pattern])

        # dates of the form Aug-10-2018
        lab = [
            {"LOWER": item},
            {"IS_PUNCT": True, "OP": "?"},
            {"IS_ALPHA": True},
            {"IS_PUNCT": True, "OP": "?"},
            {"IS_DIGIT": True},
            {"IS_PUNCT": True, "OP": "?"},
            {"IS_DIGIT": True},
        ]
        pattern = {"label": type, "pattern": lab}
        patterns.append([pattern])

        # for i in item:
        items = item.split(" ")
        if len(items) > 1:
            lab = [
                {"LOWER": items[0]},
                {"IS_SPACE": True, "OP": "?"},
                {"LOWER": items[1]},
                {"IS_PUNCT": True, "OP": "?"},
                {"TEXT": {"REGEX": r"^\d{1,2}/\d{1,2}/\d{2}(?:\d{2})?$"}},
            ]
            pattern = {"label": type, "pattern": lab}
            patterns.append([pattern])

        items = item.split(" ")
        if len(items) > 1:
            lab = [
                {"LOWER": items[0]},
                {"IS_SPACE": True, "OP": "?"},
                {"LOWER": items[1]},
                {"IS_PUNCT": True, "OP": "?"},
                {"IS_DIGIT": True},
                {"ORTH": "-"},
                {"IS_DIGIT": True},
                {"ORTH": "-"},
                {"IS_DIGIT": True},
            ]
            pattern = {"label": type, "pattern": lab}
            patterns.append([pattern])

        # print(patterns)
    return patterns
