from pathlib import Path
import re

import pandas as pd


def main():
    with open("data_preparation/allowedLabels.txt") as file:
        allowed_labels = set(file.read().lower().split("\n"))
        allowed_labels.remove("")

    print(allowed_labels)

    paths = list(Path("data/filteredData").glob("./*/*.csv"))
    # print("Num of files to filter:", len(paths))

    def remove_links(data: str) -> str:
        # remove wikipedia links aka [123]

        return re.sub("\[[^\]]*\]", "", data)

    def get_labels(path):
        with open(path) as file:
            return remove_links(file.readline().replace("\n", "")).lower().split("|")

    new_dataset = {"label": [], "column": []}

    for _, path in zip(range(100000000), paths):
        labels = get_labels(path)
        try:
            df = pd.read_csv(path, sep="|")

            assert len(labels) == len(df.columns)

            for label, column_idx in zip(labels, df.columns):
                if label in allowed_labels:
                    new_dataset["label"].append(label)
                    new_dataset["column"].append(
                        df[column_idx].astype(str).str.cat(sep=" "))
        except Exception as e:
            print(e)
            # print("Could not open: ", path)

    pd.DataFrame(new_dataset).to_csv("data/columns.csv", index=False, sep="<")


if __name__ == "__main__":
    main()
