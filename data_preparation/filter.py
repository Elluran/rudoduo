# Dirty way to filter some* data for learning

from pathlib import Path, PurePath
import re


def main():
    paths = list(Path("data/raw").glob("./*/*.csv"))
    print("Num of files to filter:", len(paths))

    def remove_links(data: str) -> str:
        # remove wikipedia links aka [123]

        return re.sub("\[[^\]]*\]", "", data)

    def get_labels(path):
        with open(path) as file:
            return remove_links(file.readline().replace("\n", "")).split("|")

    def any_bad_labels(labels):
        MONTHS = [
            "Июнь",
            "Май",
            "Июль",
            "Март",
            "Янв.",
            "Фев.",
            "Апр.",
            "Авг.",
            "Сен.",
            "Окт.",
            "Нояб.",
            "Дек.",
        ]

        for label in labels:
            is_unnamed = re.search("Unnamed", label)
            is_number = bool(re.fullmatch(r"[0-9]*", label))
            is_too_small = len(label) <= 2
            is_note = label == "Примечания" or label == "Примечание"
            is_of_compound_column = re.search(r"\.[1-9]*", label)

            if (
                is_number
                or is_unnamed
                or is_too_small
                or is_note
                or label in MONTHS
                or is_of_compound_column
            ):
                return True

        return False

    for path in paths:
        new_path = Path("data/filteredData") / PurePath(*path.parts[2:])

        labels = get_labels(path)
        if any_bad_labels(labels):
            continue

        # If header consists only of repeating labels e.g. author,author,author
        # It's an arifact of the crawler
        if len(labels) > 3 and len(set(labels)) == 1:
            continue

        regular_expressions = [
            r"\.ts-comment",
            r"Unnamed:",
            r"Основная статья:",
            r"Final Fantasy",
            r"яп. ",
        ]

        compiled_regexp = re.compile("|".join(regular_expressions))
        data = path.read_text()

        if compiled_regexp.search(data):
            continue

        filtered_data = remove_links(data)

        new_path.parent.mkdir(exist_ok=True, parents=True)
        new_path.touch(exist_ok=True)
        new_path.write_text(filtered_data)

    filtered_paths = list(Path("data/filteredData").glob("./*/*.csv"))
    print("Num of result files:", len(filtered_paths))


if __name__ == "__main__":
    main()
