from pathlib import Path


def get_allowed_labels():
    labels_list_path = Path("./data_preparation/allowedLabels.txt")
    return set(labels_list_path.read_text().lower().split("\n"))


def get_labels(table_path, sep="|"):
    with open(table_path) as file:
        return file.readline().replace("\n", "").lower().split(sep)
