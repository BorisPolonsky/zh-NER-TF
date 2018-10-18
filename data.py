import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


def get_tag2label(entity_tokens):
    tag2label = {"O": 0}
    index = 1
    for token in entity_tokens:
        tag2label["B-%s" % token] = index
        index += 1
        tag2label["I-%s" % token] = index
        index += 1
    return tag2label


def read_corpus(corpus_path, sep='\t'):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :param sep:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        line = line.rstrip()
        if len(line) > 0:
            char, label = line.rstrip().split(sep)
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    if len(sent_) > 0:
        data.append((sent_, tag_))
    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, (word_id, word_freq) in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id,
                digit_token_override='<NUM>',
                latin_char_token_override='<ENG>',
                unknown_word_token='<UNK>'):
    """

    :param sent: Sentence.
    :param word2id: A dictionary s.t. word2id[word] indicates the index of word.
    :param digit_token_override. If not None, override digits with this token (str)
    :param latin_char_token_override. If not None, override latin characters with this token (str).
    :param unknown_word_token. Token (str) for unknown words.
    :return:
    """
    sentence_id = []
    for word in sent:
        if (digit_token_override is not None) and word.isdigit():
            word = digit_token_override
        elif (latin_char_token_override is not None) and \
                ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = latin_char_token_override
        if word not in word2id:
            word = unknown_word_token
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False,
                digit_token='<NUM>', latin_char_token='<ENG>', unknown_word_token='<UNK>'):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :param digit_token. If not None, override digits with this token (str)
    :param latin_char_token. If not None, override latin characters with this token (str).
    :param unknown_word_token. Token (str) for unknown words.
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab,
                            digit_token_override=digit_token,
                            latin_char_token_override=latin_char_token,
                            unknown_word_token=unknown_word_token)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

