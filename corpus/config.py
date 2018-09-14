class Config():
    def __init__(self):
        # word embedding dimension
        self.dim_word = 300

        # glove files
        self.filename_glove = "embedding/glove.6B/glove.6B.{}d.txt".format(self.dim_word)

        # trimmed glove file
        self.filename_trimmed_glove = 'data/trimmed_glove.npz'

        # dataset
        self.filename_train = 'raw_data/ptb/train'
        self.filename_valid = 'raw_data/ptb/valid'
        self.filename_test = 'raw_data/ptb/test'

        # trimmed data
        self.filename_trimmed_train = 'data/train.npz'
        self.filename_trimmed_valid = 'data/valid.npz'
        self.filename_trimmed_test = 'data/test.npz'

        # sequence length
        self.max_length = 32

        # vocab
        self.filename_words = 'data/word2index.pkl'
        self.filename_index = 'data/index2word.pkl'
