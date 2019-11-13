from IPython.display import clear_output
from IPython.display import display

class Utility:
    def __init__(self):
        self.nothing = None

    def update_progress(self, index, range_of_index):
        progress = index / range_of_index
        bar_length = 40
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
        if progress < 0:
            progress = 0
        if progress >= 1:
            progress = 1
        block = int(round(bar_length * progress))
        clear_output(wait=True)
        text = "Progress: [{0}] {1:.1f}%".format("#" * block + '>>' + "_" * (bar_length - block), progress * 100)
        one_of = "Processing {} of {}".format(index + 1, range_of_index + 1)
        print(text)
        print(one_of)
        if index == range_of_index:
            print('Processing completed')

    def generate_submission(self, submission_file, target, predicted_values):
        submission_file[target] = predicted_values
        submission_file.to_csv(input('Enter filename: ') + '.csv', index=False)
        print('Submission File has been created, Goodluck!')