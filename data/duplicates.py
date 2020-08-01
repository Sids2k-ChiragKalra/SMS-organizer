import csv
import time

import data.features as ft


def group_duplicates(data):
    start_time = time.time()

    done = [False] * len(data)
    same = list(csv.reader(open('data/pruned_db/old_same.csv', encoding='utf8')))
    _ini = len(same)

    for i, a in enumerate(data):
        if not done[i]:
            done[i] = True
            similar = [i]
            for j, b in enumerate(data[i+1:], start=i+1):
                if not done[j] and ft.similar_features(a, b):
                    done[j] = True
                    similar.append(j)
            same.append(similar)
        if i % 500 == 0:
            print(100*i/len(data), time.time()-start_time, sep=', ')

    print(100, time.time()-start_time, sep=', ')

    csv.writer(open("data/pruned_db/same_labeled.csv", "w+", encoding='utf8', newline=''), delimiter=',').writerows(same[:_ini])
    csv.writer(open("data/pruned_db/same.csv", "w+", encoding='utf8', newline=''), delimiter=',').writerows(same[_ini:])
