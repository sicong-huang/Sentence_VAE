class Config():
    def __init__(self, load=True):
        if load:
            self.load()
    def load(self):
        pass


    # embedding
    dim_word = 300

    # glove files
    filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)

    # trimmed glove file
    filename_trimmed_glove = 'data/trimmed_glove.npz'

    # dataset
    filename_train = 'data/train'
    filename_valid = 'data/valid'
    filename_test = 'data/test'

    # trimmed data
    filename_trimmed_train = 'data/train.npz'
    filename_trimmed_valid = 'data/valid.npz'
    filename_trimmed_test = 'data/test.npz'

    max_length = 32

    # sign
    flag_text = True
    start = True
    pad = True

    # vocab
    filename_words = 'data/word2index.pkl'
    filename_index = 'data/index2word.pkl'

    # training

    #model hyperparameters

