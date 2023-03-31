import csv
import numpy as np

def count_original():
    with open('data/train.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    array = np.array(data)
    print(array[1:,6])

    def count_classes(array):
        """
        renvoie un dictionnaire avec les classes et leur nombre d'occurence
        """
        print('Evaluating classes...')
        classes = {}
        for couple in array:
            label = int(couple[6])
            if label in classes:
                classes[label] += 1
            else:
                classes[label] = 1
        print('Done!')
        return classes

    labels = count_classes(array[1:])
    print(labels)