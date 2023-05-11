# Filters out tables that have labels that are not allowed in the list

from pathlib import Path, PurePath
import re


def main():
    with open("data_preparation/allowedLabels.txt") as file:
        allowed_labels = set(file.read().lower().split("\n"))

    print(allowed_labels)

    paths = list(Path("data/filteredData").glob("./*/*.csv"))
    print("Num of files to filter:", len(paths))

    def remove_links(data: str) -> str:
        # remove wikipedia links aka [123]

        return re.sub("\[[^\]]*\]", "", data)

    def get_labels(path):
        with open(path) as file:
            return remove_links(file.readline().replace("\n", "")).lower().split("|")

    for path in paths:
        new_path = Path("data/filteredData2") / PurePath(*path.parts[2:])

        labels = set(get_labels(path))

        if len(labels | allowed_labels) > len(allowed_labels):
            new_path = Path("data/excluded_by_labels") / PurePath(*path.parts[2:])
            data = path.read_text()
            new_path.parent.mkdir(exist_ok=True, parents=True)
            new_path.touch(exist_ok=True)
            new_path.write_text(data)
            continue

        data = path.read_text()

        new_path.parent.mkdir(exist_ok=True, parents=True)
        new_path.touch(exist_ok=True)
        new_path.write_text(data)

    filtered_paths = list(Path("data/filteredData2").glob("./*/*.csv"))
    print("Num of result files:", len(filtered_paths))


if __name__ == "__main__":
    main()
