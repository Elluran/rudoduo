from pathlib import Path, PurePath
import pandas as pd


def main():
    paths = list(Path("data/filteredData2").glob("./*/*.csv"))
    print("Num of files to filter:", len(paths))

    replacement = {
        "кол-во": "количество",
        "автор сценария": "сценарист",
        "рост, см": "рост",
        "year": "год",
        "численность населения": "население, человек",
        "население": "население, человек",
    }

    total_replacements = 0

    def get_labels(path):
        with open(path) as file:
            return file.readline().replace("\n", "").lower().split("|")

    def replace_labels(labels):
        nonlocal total_replacements
        for i in range(len(labels)):
            if labels[i] in replacement.keys():
                labels[i] = replacement[labels[i]]
                total_replacements += 1
        return labels

    final_labels_list = set()

    for path in paths:
        new_path = Path("data/filteredData3") / PurePath(*path.parts[2:])

        labels = get_labels(path)
        new_labels = "|".join(replace_labels(labels))
        data = path.read_text()
        new_data = "\n".join([new_labels] + data.split("\n")[1:])
        
        final_labels_list.update(replace_labels(labels))

        new_path.parent.mkdir(exist_ok=True, parents=True)
        new_path.touch(exist_ok=True)
        new_path.write_text(new_data)

    filtered_paths = list(Path("data/filteredData3").glob("./*/*.csv"))
    df = pd.DataFrame({"paths": filtered_paths})
    df.to_csv("data/filteredData3/files_list.csv", index=False)
    pd.DataFrame(final_labels_list).to_csv("data/labels.csv", index=False)
    print("Num of result files:", len(filtered_paths))
    print("Total label replacements:", total_replacements)


if __name__ == "__main__":
    main()
