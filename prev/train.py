#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import preprocess

BINARY = True
PRETRAINEDEMBEDDING = True
if PRETRAINEDEMBEDDING:
    import main_pre_trained_embeddings

    from main_pre_trained_embeddings import TextCNN
else:
    import main_with_embeddings
    from main_with_embeddings import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

weights = []

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
if PRETRAINEDEMBEDDING and main_pre_trained_embeddings.Embedding!="ELMo":
    tf.flags.DEFINE_integer("embedding_dim", main_pre_trained_embeddings.d, "Dimensionality of character embedding (default: 128)")
elif not PRETRAINEDEMBEDDING:
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
else:
    tf.flags.DEFINE_integer("embedding_dim", 1024, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 250, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 250, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess_():
    # Data Preparation
    # ==================================================
    # Load data
    print("Loading data...")
    if BINARY:
        x_text, y = preprocess.load_data_and_bin_labels("./CrisisLexT26_preprocessed/")
    else:
        x_text, y = preprocess.load_data_and_labels("./CrisisLexT26_preprocessed/")
    if PRETRAINEDEMBEDDING:
        main_pre_trained_embeddings.x_text = x_text
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    print("Max Document length:", max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "fastText" or main_pre_trained_embeddings.Embedding == "Bert"):
        x = []
        for counter, j in enumerate(x_text):
            x.append([])
            x[counter] = [main_pre_trained_embeddings.word2idx.get((i,counter), main_pre_trained_embeddings.word2idx[('UNK',0)]) for i in j.split()]
            while len(x[counter]) < max_document_length:
                x[counter].append(0)

        x = np.array(x)

    else:
        x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    # np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                binary=BINARY)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "fastText" or main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "Bert"):
                embedding_init = cnn.W.assign(cnn.embedding_placeholder)
                # x_init = cnn.x_.assign(cnn.x)




            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            precision_summary = tf.summary.scalar("precision", cnn.precision)
            recall_summary = tf.summary.scalar("recall", cnn.recall)

            conf_summary = tf.summary.tensor_summary("confusion", cnn.confusion)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged, conf_summary])# , conf_summary
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "fastText" or main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "Bert"):
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), embedding_init], feed_dict={cnn.embedding_placeholder: cnn.weights}) #, x_init], feed_dict={cnn.embedding_placeholder: cnn.weights, cnn.x_placeholder: cnn.x})
            else: #if not PRETRAINEDEMBEDDING:
                sess.run(tf.global_variables_initializer())
            #else:
            #sess.run([tf.global_variables_initializer(), main_pre_trained_embeddings.embeddings], feed_dict={cnn.train_x: cnn.train_x})


            def train_step(x_batch, y_batch):
                """
                A single training step
                """

                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                }
                _, step, summaries, loss, accuracy, precision, recall, confusion = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.confusion],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, accuracy {:g}, precision {}, recall {}".format(time_str, step, loss, accuracy, precision, recall))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }

                step, summaries, loss, accuracy, precision, recall, confusion = sess.run( #, confusion
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.confusion],#, cnn.confusion],
                    feed_dict)
                # step, loss, accuracy, confusion = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.confusion], feed_dict=feed_dict)
                print(confusion)
                # print(sess.run(cnn.predictions, feed_dict={cnn.predictions: cnn.predictions}))
                # sess.run(cnn.predictions)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, accuracy {:g}, precision {}, recall {}".format(time_str, step, loss, accuracy, precision, recall))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = preprocess.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)

                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step == FLAGS.num_epochs:
                    break

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess_()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()
