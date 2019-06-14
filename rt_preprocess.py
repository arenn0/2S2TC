import os
import csv

import preprocess
from numpy import array
import tensorflow as tf
import nltk
import numpy

tf.flags.DEFINE_string("input_dir", "italian_preprocessed/", "input dir for preprocessing purposes")
tf.flags.DEFINE_string("output_dir", "rt_like_data/italian_expanded/", "output dir for preprocessing purposes")
tf.flags.DEFINE_string("unlabeled_dir", "rt_like_data/italian_unlabeled/", "unlabeled dir for preprocessing purposes")
tf.flags.DEFINE_bool("italian", True, "English or Italian corpus")
tf.flags.DEFINE_bool("use_unlabeled", True, "Use unlabeled data")
tf.flags.DEFINE_float("percentage", 0.1, "Percentage of labeled data to use")

def format_file(text):
    file = ""
    tokenizer = nltk.data.load('tokenizers/punkt/italian.pickle')
    return '\n'.join(tokenizer.tokenize(text))

FLAGS = tf.flags.FLAGS

# percentage = 0.5

if FLAGS.use_unlabeled:
    if FLAGS.italian:
        dev_labeled_index = 1000
    else:
        dev_labeled_index = 4000
else:
    dev_labeled_index = 1000000

all_files = os.listdir(FLAGS.input_dir)
all_csv = [i for i in all_files if i.endswith('.csv')]
k = 0
j = 0
file_unlabeled = open(FLAGS.unlabeled_dir + "file.txt", "w", encoding="utf-8")
for i in all_csv:
    with open(FLAGS.input_dir + '/' + i, 'r', encoding='utf8') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        index = 0

        for line in lines:
            index += 1
            if index == 1:
                continue

            if len(line) == 2:
                if (FLAGS.italian and line[1] != 'not relevant') or (
                        not FLAGS.italian and (line[1] != 'Not related' and line[1] != 'Not applicable')):
                    # l.append(line[1])
                    text = preprocess.clean_str(line[0])
                    if k + j > dev_labeled_index:
                        file_unlabeled.write("review/text " + text + '\n')

                    elif (FLAGS.italian and line[1] == "damage") or (not FLAGS.italian and line[1] == "Related and informative"):
                        n = numpy.random.uniform(0,1)
                        if n < FLAGS.percentage:
                            file = open(FLAGS.output_dir+ str(int(FLAGS.percentage *100)) + "/" + str(k) + ".pos", "w", encoding="utf-8")
                            text = format_file(text)
                            file.write(text)
                        k += 1

                    else:
                        n = numpy.random.uniform(0, 1)
                        if n < FLAGS.percentage:
                            file = open(FLAGS.output_dir + str(int(FLAGS.percentage *100)) + "/" + str(j) + ".neg", "w", encoding="utf-8")
                            text = format_file(text)
                            file.write(text)
                        j += 1
