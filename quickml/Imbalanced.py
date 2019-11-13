
# from Utility import Utility as utility
import pandas as pd

class Imbalanced:

    def __init__(self, data):
        self.data = data

    def DivideAndConquer(self, target, minority_class, majority_class, n=5000):
        total = len(self.data)
        minority = self.data[self.data[target] == minority_class]
        majority = self.data[self.data[target] == majority_class]
        majoritysampling = majority.sample(n=len(majority), random_state=10)
        new_majority_len = ((len(majority) - len(minority)) / total) * n
        iterations = len(majority) / new_majority_len
        pickrange = list(range(int(new_majority_len), len(majority), int(new_majority_len)))
        print(len(pickrange))
        iterable = [0]
        datatable = []

        for IndexValue in range(int(iterations)):
            newMajority = majoritysampling[iterable[-1]:pickrange[IndexValue]]
            new_data = pd.concat([newMajority, minority], ignore_index=True)
            datatable.append(new_data)
            iterable.append(pickrange[IndexValue])
            # utility.update_progress(1,IndexValue, int(iterations) - 1)
        return datatable
