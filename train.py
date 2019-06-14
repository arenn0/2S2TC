#! /usr/bin/env python

import datetime
import os
import shutil
import time

import numpy as np
import numpy.random as random
import tensorflow as tf

import preprocess

ITALIAN = True
BINARY = True
PRETRAINEDEMBEDDING = True
TRAIN = True

CNN = True
SVM= False
NB = False
# Parameters
# ==================================================

max_document_length = 0

weights = []
confidence = 0.999

percentage_of_labeled_data = 1
if ITALIAN:
    dev_labeled_index = 1500
else:
    dev_labeled_index = 4000

import main_pre_trained_embeddings
if PRETRAINEDEMBEDDING and CNN:
    ITALIAN = main_pre_trained_embeddings.ITALIAN
    from main_pre_trained_embeddings import TextCNN
    print("Pretrained Embedding: {}".format(main_pre_trained_embeddings.Embedding))
    print("Italian: {}".format(ITALIAN))

elif CNN:
    import main_with_embeddings
    from main_with_embeddings import TextCNN
    ITALIAN = main_with_embeddings.ITALIAN
    print("No Pretrained Embedding")
    print("Italian: {}".format(ITALIAN))
elif SVM:
    print("SVM")
else:
    print("NaiveBayes")
from tensorflow.contrib import learn
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
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("cotrain_every", 50, "Cotrain Every (default: 50)")
if ITALIAN:
    tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
else:
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 25, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
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
    tf.reset_default_graph()

    if BINARY:
        if ITALIAN:
            x_text, y = preprocess.load_data_and_bin_labels("./italian_preprocessed/")
        else:
            x_text, y = preprocess.load_data_and_bin_labels("./CrisisLexT26_preprocessed/")
    else:
        if ITALIAN:
            x_text, y = preprocess.load_data_and_labels("./italian_preprocessed/")
        else:
            x_text, y = preprocess.load_data_and_labels("./CrisisLexT26_preprocessed/")

    global max_document_length
    max_document_length = max([len(x.split(" ")) for x in x_text])
    print("Max Document length:", max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    if CNN:

        if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "ELMo" or main_pre_trained_embeddings.Embedding=="Bert" or main_pre_trained_embeddings.Embedding == 'GloVe' or main_pre_trained_embeddings.Embedding == "fastText"):
            x_shuffled = x_text
            y_shuffled = y
        else:
            x_shuffled = np.array(list(vocab_processor.fit_transform(x_text)))
            # shuffle_indices = np.random.permutation(np.arange(len(y)))
            # x_shuffled = x[shuffle_indices]
            # y_shuffled = y[shuffle_indices]
            y_shuffled = y

    if SVM or NB:
        x_shuffled = x_text
        y_shuffled = y
    # Split train/test set

    print("Confidence Threshold: ", confidence)

    if PRETRAINEDEMBEDDING and (
            main_pre_trained_embeddings.Embedding == "ELMo" or main_pre_trained_embeddings.Embedding == "Bert" or main_pre_trained_embeddings.Embedding == 'GloVe' or main_pre_trained_embeddings.Embedding == "fastText"):
        x, x_unlabeled = x_shuffled[:dev_labeled_index], x_text[dev_labeled_index:]
    else:
        x, x_unlabeled = x_shuffled[:dev_labeled_index], x_shuffled[dev_labeled_index:]

    y, y_unlabeled = y_shuffled[:dev_labeled_index], y_shuffled[dev_labeled_index:]

    x_dev = []
    y_dev = []
    x_eval = []
    y_eval = []
    x_train = []
    y_train = []
    x_t = []

    for i in range(len(x)):
        random_int = random.randint(1, 10)
        is_validation = random_int == 1
        if ITALIAN:
            is_test = (random_int == 2 or random_int == 3 or random_int == 4)
        else:
            is_test = random_int == 2

        if is_validation:
            x_dev.append(x[i])
            y_dev.append(y[i])
        elif is_test:
            x_eval.append(x_text[i])
            y_eval.append(y[i])
        else:
            x_train.append(x[i])
            y_train.append(y[i])
            x_t.append(x_text[i])

    # CREATE the eval directory
    try:
        os.mkdir('eval_dir')
    except:
        pass
    shutil.rmtree('eval_dir')
    os.mkdir('eval_dir')
    file_eval = open("eval_dir/file.csv", "w", encoding='utf-8')
    for i in range(len(x_eval)):
        if ITALIAN:
            c = "damage" if np.argmax(y_eval[i]) == 1 else "no damage"
        else:
            c = "Related and informative" if np.argmax(y_eval[i]) == 1 else "Related - but not informative"
        file_eval.write(str(x_eval[i]) + str(",") + c + '\n')
    del x, y, x_shuffled, y_shuffled

    x_train, y_train, x_t = x_train[:int(percentage_of_labeled_data * len(x_train))], y_train[:int(percentage_of_labeled_data * len(x_train))], x_t[:int(percentage_of_labeled_data * len(x_train))]

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    # print("Dimension of the training set: ", len(x_shuffled[:int(dev_labeled_index * percentage_of_labeled_data)]), percentage_of_labeled_data*100, "%")
    print("Train/Unlabeled/Dev split: {:d}/{:d}/{:d}".format(len(y_train),len(y_unlabeled), len(y_dev)))
    return x_text, x_t, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev


def train(x_text, x_t, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev):
    # Training
    # ==================================================

    if SVM or NB:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import SGDClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from NLTKPreprocessor import NLTKPreprocessor
        from NLTKPreprocessor import identity

        if SVM:
            text_clf = Pipeline([('vect', NLTKPreprocessor('English')), ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), ('clf-svm', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, random_state=42)), ])
        elif NB:
            text_clf = Pipeline([('vect', NLTKPreprocessor('English')), ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), ('clf-svm', MultinomialNB()), ])
        labels = np.argmax(y_train, axis=1) + 1
        next = []
        step = 0
        while len(next) != 0 or step == 0:
            step += 1
            _ = text_clf.fit(x_train,(labels))
            predicted_svm = text_clf.predict(x_dev)
            # SELF-TRAINING
            predicted_svm_unlabeled = text_clf.predict_proba(x_unlabeled)
            maxs = np.max(predicted_svm_unlabeled, axis=1)

            next = []
            next_y = []
            next_y_italian = []
            next_unlabeled = []

            for i in range(len(maxs)):
                if maxs[i] > confidence:
                    next.append(x_unlabeled[i])
                    next_y.append(np.argmax(predicted_svm_unlabeled[i]) + 1)
                else:
                    next_unlabeled.append(x_unlabeled[i])
                    # CO-TRAINING

            x_train += next
            labels = np.append(labels, next_y)
            x_unlabeled = next_unlabeled
            # END OF SELF-TRAINING
            labels_dev = np.argmax(y_dev, axis=1) + 1

            matrix = confusion_matrix(labels_dev, predicted_svm)
            # predicted_svm == np.ravel(labels_dev))
            print("English evaluation: ")
            print(matrix)
            print("accuracy {}, precision {}, recall {}, f1_score {}".format(accuracy_score(labels_dev, predicted_svm),
                                                                             precision_score(labels_dev, predicted_svm),
                                                                             recall_score(labels_dev, predicted_svm),
                                                                             f1_score(labels_dev, predicted_svm)))

    if CNN:
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == 'ELMo'):
                    pass

                cnn = TextCNN(
                    sequence_length=max_document_length,
                    num_classes=len(y_train[0]),
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    text=x_text,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    binary=BINARY)

                if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "fastText"):
                    # max_document_length = max([len(x.split(" ")) for x in x_text])
                    x = []
                    for counter, j in enumerate(x_t):
                        x.append([])
                        x[counter] = []
                        for i in j.split():
                            index = cnn.word2idx.get(i, len(cnn.word2idx) - 1)
                            # print(i,index)
                            x[counter].append(index)
                        while len(x[counter]) < max_document_length:
                            x[counter].append(len(cnn.word2idx) - 1)

                    x_u = []
                    for counter, j in enumerate(x_unlabeled):
                        x_u.append([])
                        x_u[counter] = []
                        for i in j.split():
                            index = cnn.word2idx.get(i, len(cnn.word2idx) - 1)
                            # print(i,index)
                            x_u[counter].append(index)
                        while len(x_u[counter]) < max_document_length:
                            x_u[counter].append(len(cnn.word2idx) - 1)
                    x_d = []
                    for counter, j in enumerate(x_dev):
                        x_d.append([])
                        x_d[counter] = []
                        for i in j.split():
                            index = cnn.word2idx.get(i, len(cnn.word2idx) - 1)
                            # print(i,index)
                            x_d[counter].append(index)
                        while len(x_d[counter]) < max_document_length:
                            x_d[counter].append(len(cnn.word2idx) - 1)

                    x_train = np.array(x)
                    x_unlabeled = np.array(x_u)
                    x_dev = np.array(x_d)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "fastText" or main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "Bert"):
                    embedding_init = cnn.W.assign(cnn.embedding_placeholder)

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

                def train_step(x_batch, y_batch, x_unlabeled, y_unlabeled):
                    """
                    A single training step
                    """
                    # x_batch = tf.cast(x_batch, tf.float32)
                    # a = result + x_batch

                    if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "ELMo" or main_pre_trained_embeddings.Embedding=="Bert"):
                        feed_dict = {
                            cnn.unlabeled_training: x_unlabeled,
                            cnn.y_unlabeled: y_unlabeled,
                            cnn.n: len(x_batch),
                            cnn.input_x: x_batch,
                            cnn.x_text: x_batch,
                            cnn.input_y: y_batch,
                            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        }
                    else:

                        feed_dict = {
                            cnn.unlabeled_training: x_unlabeled,
                            cnn.y_unlabeled: y_unlabeled,
                            cnn.n: len(x_batch),
                            cnn.input_x: x_batch,
                            cnn.input_y: y_batch,
                            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        }

                    if TRAIN and COTRAIN:
                        _, step, summaries, loss, accuracy, precision, recall, confusion, next, next_y, result, result_y, accuracy_unlabeled, confusion_unlabeled = sess.run(
                            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.confusion, cnn.next, cnn.next_y, cnn.results, cnn.results_y, cnn.accuracy_unlabeled, cnn.confusion_unlabeled],
                            feed_dict)

                        print("Number of tweets above confidence threshold: ", len(result))
                        print(confusion_unlabeled)
                    else:
                        result_y = []
                        result = []
                        next = x_unlabeled
                        next_y = y_unlabeled
                        _, step, summaries, loss, accuracy, precision, recall, confusion = sess.run(
                            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall,
                             cnn.confusion],
                            feed_dict)
                        #                 print(confusion)

                    time_str = datetime.datetime.now().isoformat()
                    # print("{}: step {}, loss {:g}, accuracy {:g}, precision {}, recall {}".format(time_str, step, loss, accuracy, precision, recall))
                    train_summary_writer.add_summary(summaries, step)

                    b = []
                    for i in result_y:
                        b.append(i)
                    a = []
                    for i in result:
                        a.append(i)

                    return next, next_y, a, b

                def eval_step():
                    x_raw, y_test = preprocess.load_data_and_bin_labels("eval_dir/")

                    if PRETRAINEDEMBEDDING and (
                            main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "fastText"):
                        x = []
                        for counter, j in enumerate(x_raw):
                            x.append([])
                            x[counter] = []
                            for i in j.split():
                                index = cnn.word2idx.get(i, len(cnn.word2idx) - 1)
                                # print(i,index)
                                x[counter].append(index)
                            while len(x[counter]) < max_document_length:
                                x[counter].append(len(cnn.word2idx) - 1)

                        x_test = np.array(x)
                    else:
                        # Map data into vocabulary
                        x_test = np.array(list(vocab_processor.transform(x_raw)))
                    print("EVALUATION SET:")
                    dev_step(x_test, y_test)

                def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "ELMo" or main_pre_trained_embeddings.Embedding=="Bert"):
                        feed_dict = {
                            cnn.unlabeled_training: x_unlabeled,
                            cnn.n: len(x_batch),
                            cnn.x_text: x_batch,
                            cnn.input_y: y_batch,
                            cnn.dropout_keep_prob: 1.0,
                        }
                    else:
                        feed_dict = {
                            cnn.unlabeled_training: x_unlabeled,
                            cnn.n: len(x_batch),
                            cnn.input_x: x_batch,
                            cnn.input_y: y_batch,
                            cnn.dropout_keep_prob: 1.0,
                        }
                    sess.run(tf.local_variables_initializer())
                    step, summaries, loss, accuracy, precision, recall, confusion, f1 = sess.run( #, confusion
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.precision[-1], cnn.recall[-1], cnn.confusion, cnn.f1[-1]],#, cnn.confusion],
                        feed_dict)
                    print(confusion)
                    print("step {}, loss {:g}, accuracy {:g}, precision {}, recall {}, f1 {}".format(step, loss, accuracy, precision, recall, f1))
                    if writer:
                        writer.add_summary(summaries, step)

                # Training loop. For each batch...
                if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "ELMo"):
                    x_ = [[np.array(i)] for i in range(len(x_train))]
                    batches = preprocess.batch_iter(list(zip(x_, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                else:
                    batches = preprocess.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

                result = []
                result2 = []
                y_prediction = []
                y_prediction2 = []
                current_step = 0
                n = n_prev = 0
                l = 0
                unlabeled_training = x_unlabeled
                while current_step == 0 or n != 0:
                    result += result2
                    y_prediction += y_prediction2
                    if current_step != 0:
                        if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "ELMo"):
                            x_ = [[np.array(i)] for i in range(len(x_train + result))]
                            batches = preprocess.batch_iter(list(zip(x_, y_train + y_prediction)), FLAGS.batch_size, FLAGS.num_epochs)
                        else:
                            x = list(x_train) + result
                            y = list(y_train) + y_prediction

                            batches = preprocess.batch_iter(list(zip(x, y)), FLAGS.batch_size , FLAGS.num_epochs)

                    print("New CNN run:", l)
                    l += 1
                    # Initialize all variables
                    if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "fastText" or main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "Bert"):
                        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), embedding_init], feed_dict={cnn.embedding_placeholder: cnn.weights})
                    else:
                        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

                    for batch in batches:
                        x_batch, y_batch = zip(*batch)
                        current_step = tf.train.global_step(sess, global_step) + 1

                        COTRAIN = False

                        if PRETRAINEDEMBEDDING and main_pre_trained_embeddings.Embedding == "ELMo":
                            x_t = []
                            for i in x_batch:
                                x_t.append(x_train[i[0]])
                            for i in range(len(x_t)):
                                while len(x_t[i].split()) < max_document_length:
                                    x_t[i] += ' unk '
                            train_step(x_t, y_batch, unlabeled_training, y_unlabeled)
                        else:
                            train_step(x_batch, y_batch, unlabeled_training, y_unlabeled)


                        if current_step % FLAGS.evaluate_every == 0:
                            print("Evaluation:")
                            if PRETRAINEDEMBEDDING and main_pre_trained_embeddings.Embedding == "ELMo":
                                for i in range(len(x_dev)):
                                    while len(x_dev[i].split()) < seq:
                                        x_dev[i] += ' unk '
                            dev_step(x_dev, y_dev, writer=dev_summary_writer)
                            print("")
                        if current_step % FLAGS.checkpoint_every == 0:
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))
                        if current_step == FLAGS.num_epochs:
                            COTRAIN = True
                            unlabeled_training, y_unlabeled, result2, y_prediction2 = train_step(x_batch, y_batch, unlabeled_training, y_unlabeled)
                            n_prev = len(result)
                            n = len(result2)
                            print("Added " + str(n) + " to Training Set. " + str(n_prev+n) + "/" + str(len(x_unlabeled)))
                            COTRAIN = False
                            eval_step()
                            break





def main(argv=None):
    x_text, x_t, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev = preprocess_()
    train(x_text, x_t, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev)

if __name__ == '__main__':
    tf.app.run()
