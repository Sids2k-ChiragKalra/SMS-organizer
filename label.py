import csv


if __name__ == '__main__':
    same = list(csv.reader(open('data/pruned_db/same.csv', encoding='utf8')))
    old_same = list(csv.reader(open('data/pruned_db/same_labeled.csv', encoding='utf8')))
    features = list(csv.reader(open('data/pruned_db/unlabeled.csv', encoding='utf8')))

    labels = list(csv.reader(open('C:/Users/bruhascended/PycharmProjects/SMSDiscordBot/labels.csv', encoding='utf8')))
    old_labels = list(csv.reader(open('data_collection/pruned_db/old_labels.csv', encoding='utf8')))

    same = old_same.__add__(same)
    labels = old_labels.__add__(labels)

    for i, indexes in enumerate(same):
        for index in indexes:
            features[int(index)].append(int(labels[i][1]))

    csv.writer(open("data/train_db/dataset.csv", "w+", encoding='utf8', newline=''), delimiter=',').writerows(features)
