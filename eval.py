#! /usr/bin/env python

import csv
import os

import numpy as np
import sklearn
import tensorflow as tf
from tensorflow.contrib import learn

import main_pre_trained_embeddings
import preprocess

PRETRAINEDEMBEDDING = True
ITALIAN = False
Embedding = main_pre_trained_embeddings.Embedding


def initialize(x_text):
    if Embedding == "GloVe":
        d = 300
        path = "../Data/embeddings/glove.840B.300d.txt"
        from gensim.models import KeyedVectors

        # glove_file = datapath(path)
        # tmp_file = get_tmpfile("test_word2vec.txt")
        # _ = glove2word2vec(glove_file, tmp_file)
        model = KeyedVectors.load_word2vec_format(path)
        print("GloVe model loaded")
    elif Embedding == "fastText":
        d = 300
        if ITALIAN:
            path = "../Data/italian_embeddings/cc.it.300.vec"
        else:
            path = "../Data/embeddings/crawl-300d-2M-subword.vec"
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format(path)
        print("fastText model loaded")


    word2idx = {}
    weights = []
    if Embedding == "fastText" or Embedding == "GloVe":
        weights = []
        word_set = set()
        index = 0
        for i in x_text:
            for j in i.split():
                if j not in word_set:
                    if j in model.vocab:
                        word_set.add(j)
                        weights.append(model.vectors[model.vocab[j].index])
                        # print(j, model.vocab[j].index)  # Print the correspondences between indices and embedding table
                        word2idx[j] = index
                        index += 1
        weights.append(np.random.randn(d))
        word2idx['UNK'] = index
    return word2idx, weights

# Parameters
# ==================================================

# Data Parameters
# tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1556639491/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_dir", "eval_dir/", "Evaluate directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
x_raw, y_test = preprocess.load_data_and_bin_labels(FLAGS.eval_dir)
x_text, y_text = preprocess.load_data_and_bin_labels("./CrisisLexT26_preprocessed/")
y_test = np.argmax(y_test, axis=1)

word2idx, weights = initialize(x_text)

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        n = graph.get_operation_by_name("n").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "fastText"):
            max_document_length = max([len(x.split(" ")) for x in x_text])
            x = []
            for counter, j in enumerate(x_raw):
                x.append([])
                x[counter] = []
                for i in j.split():
                    index = word2idx.get(i, len(word2idx) - 1)
                    # print(i,index)
                    x[counter].append(index)
                while len(x[counter]) < max_document_length:
                    x[counter].append(len(word2idx) - 1)

            x_test = np.array(x)
        else:
            # Map data into vocabulary
            vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
            vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
            x_test = np.array(list(vocab_processor.transform(x_raw)))

        print("\nEvaluating...\n")
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch

        batches = preprocess.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, n: len(x_test_batch), dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    x = sklearn.metrics.confusion_matrix(y_test, all_predictions)
    accuracy = sklearn.metrics.accuracy_score(y_test, all_predictions)
    precision = sklearn.metrics.precision_score(y_test, all_predictions)
    recall = sklearn.metrics.recall_score(y_test, all_predictions)
    f1 = sklearn.metrics.f1_score(y_test, all_predictions)
    print(x)
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(accuracy))
    print("Precision: {:g}".format(precision))
    print("Recall: {:g}".format(recall))
    print("F1: {:g}".format(f1))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
