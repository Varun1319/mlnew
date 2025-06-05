import pandas as pd


def find_s_algorithm(dataset):
    attributes = dataset.iloc[:, :-1].values
    hypothesis = ["?" for _ in range(attributes.shape[1])]
    labels = dataset.iloc[:, -1].values

    for i, label in enumerate(labels):
        if label == "Yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] == "?":
                    hypothesis[j] = attributes[i][j]
                elif hypothesis[j] != attributes[i][j]:
                    hypothesis[j] = "?"
    return hypothesis


dataset = pd.read_csv("Dataset.csv")
hypothesis = find_s_algorithm(dataset)

print(hypothesis)
