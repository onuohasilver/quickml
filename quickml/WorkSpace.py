class WorkSpace:
    """
    Create and manage a basic Worksspace.
    Creates a folder tree of data, notebooks,submission files.
    """

    def __init__(self, files):
        self.trainWS = files[0]
        self.testWS = files[1]
        self.submissionWS = files[2]

    def create(self, move=False):
        try:
            os.mkdir('data')
            os.mkdir('notebooks')
            os.mkdir('submissions')
        except:
            print('Error Making Directories')
            pass
        if move:
            try:
                shutil.move(self.trainWS, 'data')
                shutil.move(self.testWS, 'data')
                shutil.move(self.submissionWS, 'data')
            except:
                print('Error moving files..')

