import csv
import os

from keras.utils import to_categorical
from numpy import array

# import train

ITALIAN = False
create_file = False

def preprocess(dir, dir2, COLUMN_CATEGORY = 4, COLUMN_TWEET = 1):
    all_files = os.listdir(dir)
    csv_files = [i for i in all_files if i.endswith('.csv') and not 'preprocessed' in i]

    for i in csv_files:
        with open(dir + '/' + i, 'r', encoding='utf8') as csvfile:
            csv_write = open(dir2 + '/' + i[:-4] + "_preprocessed.csv", "w", encoding='utf8', newline='')
            lines = csv.reader(csvfile, delimiter=',')
            writer = csv.writer(csv_write, delimiter=',', quotechar='|')
            for row in lines:
                if row[COLUMN_TWEET][:2] != "RT":
                    #   and row[COLUMN_CATEGORY] != "not_related_or_irrelevant" \
                    #   and row[COLUMN_CATEGORY] != "other_useful_information": \
                    writer.writerow([row[COLUMN_TWEET].replace(',',''), row[COLUMN_CATEGORY]])

# preprocess('./CrisisLexT26', './CrisisLexT26_preprocessed')



import numpy as np
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(dir):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    all_files = os.listdir(dir)
    all_csv = [i for i in all_files if i.endswith('.csv')]

    corpus = []
    labels = []
    l = {}
    for i in all_csv:
        with open(dir + '/' + i, 'r', encoding='utf8') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            index = 0
            for line in lines:
                if index == 0:
                    index += 1
                    continue
                if len(line) == 2:
                    # print(line, (ITALIAN and line[1] != 'no damage'), (not ITALIAN and (line[1] !='Not related' and line[1] !='Not applicable')))
                    if (ITALIAN and line[1] != 'not relevant') or (not ITALIAN and (line[1] !='Not related' and line[1] !='Not applicable')):
                        corpus.append(clean_str(line[0]))
                        # l.append(line[1])
                        if line[1] in l:
                            labels.append(l[line[1]])
                        else:
                            l[line[1]] = len(l)
                            labels.append(l[line[1]])
    print(l)
    print([labels.count(i) for i in range(len(l))])
    #exit()
    data = array(labels)
    encoded = to_categorical(data)

    if create_file:
        file = open("CrisisLexT26_output_related.txt", "w", encoding="utf8")
        for i in range(len(corpus)):
            file.write(corpus[i] + "\n")
            file.flush()
        exit()
    return corpus, encoded


def load_data_and_bin_labels(dir):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    all_files = os.listdir(dir)
    all_csv = [i for i in all_files if i.endswith('.csv')]

    corpus = []
    labels = []
    l = {}
    for i in all_csv:
        with open(dir + '/' + i, 'r', encoding='utf8') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            index = 0
            for line in lines:
                if index == 0:
                    index += 1
                    continue
                if len(line) == 2:
                    if (ITALIAN and line[1] != 'not relevant') or (not ITALIAN and (line[1] !='Not related' and line[1] !='Not applicable') ):
                        corpus.append(clean_str(line[0]))
                        # l.append(line[1])
                        if (ITALIAN and line[1] == "damage") or (not ITALIAN and line[1] == "Related and informative"):
                            labels.append(1)
                        else:
                            labels.append(0)

    data = array(labels)
    encoded = to_categorical(data)
    print(len(corpus))
    return corpus, encoded


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_from_file(name):
    file = open(name, "r", encoding="utf-8")
    x = [line for line in file]
    # print(x)
    return x