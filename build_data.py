import argparse
from config import Config
from data_utils import file_check, dim_check, get_datasets_1, get_datasets, seq_len, get_train_vocab, get_glove_vocab, word2index, index2word, write_vocab,glove_embedding, get_trimmed_datasets


def main():
    # get config
    config = Config(load=False)

    # input
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', '-p', action='store_true', help='files are processed')
    parser.add_argument('--filename', '-f', metavar='filename', type=str, nargs='+',
                        help='filenames must be given in order [filename_train, filename_valid, filename_test, filename_glove]\
                             (default:*), if only a few files are to be given, default ones must be replaced by "*"')
    parser.add_argument('--max_length', '-m', metavar='NUM', type=int, help='display max_length')
    parser.add_argument('--dim_word', '-d', metavar='NUM', type=int, help='display dim_word')
    args = parser.parse_args()
    if args.processed:
        config.flag_text = args.processed
    if args.filename:
        name = ['config.filename_train', 'config.filename_valid', 'config.filename_test', 'config.filename_glove']
        for i in range(len(args.filename)):
            if args.filename[i] == '*':
                continue
            else:
                a = name[i] + '=' + '\'' + args.filename[i] + '\''
                exec(a)
    print(config.filename_train)
    print(config.filename_valid)
    print(config.filename_test)
    print(config.filename_glove)
    file_check([config.filename_train, config.filename_valid, config.filename_test, config.filename_glove])

    # Generators
    if config.flag_text:
        train = get_datasets(config.filename_train)
        valid = get_datasets(config.filename_valid)
        test = get_datasets(config.filename_test)
    else:
        train = get_datasets_1(config.filename_train, lowercase=True)
        valid = get_datasets_1(config.filename_valid, lowercase=True)
        test = get_datasets_1(config.filename_test, lowercase=True)

    config.max_length = seq_len(train)

    # input
    if args.max_length:
        config.max_length = args.max_length
    if args.dim_word:
        config.dim_word = args.dim_word

    dim_check(config.filename_glove, config.dim_word)

    # add <start> to glove
    # add_glove(config.filename_glove, config.dim_word)

    # Build word vocab
    train_words = get_train_vocab(train)
    glove_vocab = get_glove_vocab(config.filename_glove)

    # train & glove(word to index)
    vocab = word2index(train_words, glove_vocab)
    # save vocab
    write_vocab(config.filename_words, vocab)

    # index to word
    index = index2word(vocab)
    write_vocab(config.filename_index, index)

    # embedding
    glove_embedding(config.filename_glove, config.filename_trimmed_glove, config.dim_word, vocab, config.start, config.pad)

    # trim datasets
    get_trimmed_datasets(config.filename_trimmed_train, train, vocab, config.max_length)
    get_trimmed_datasets(config.filename_trimmed_valid, valid, vocab, config.max_length)
    get_trimmed_datasets(config.filename_trimmed_test, test, vocab, config.max_length)


if __name__ == '__main__':
    main()