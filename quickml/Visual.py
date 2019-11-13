import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
class Visual:

    def __init__(self):
        self.data = pd.DataFrame()
        self.column = ['dummy']
        self.target = None
        self.size = [20, 20]

    def adjust(self):
        self.column.insert(0, 'dummy')

    def setsize(self):
        sns.set(rc={'figure.figsize': (self.size[0], self.size[1])}, font_scale=1.5, style='darkgrid')

    def count(self):
        for index in range(len(self.column)):
            if index == 0:
                index += 1
            plt.subplot(round(len(self.column) / 2), 2, index)
            sns.countplot(self.data[self.column[index]], hue=self.data[self.target])

