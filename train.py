#! /usr/bin/env python

import datetime
import os
import time

import numpy as np
import tensorflow as tf

import preprocess
ITALIAN = False
BINARY = True
PRETRAINEDEMBEDDING = True
TRAIN = True

CNN = True
SVM= False
NB = False
# Parameters
# ==================================================

weights = []
confidence = 0.95

percentage_of_labeled_data = 0.5
dev_sample_index = -750
dev_labeled_index = 3000
dev_unlabeled_index = 7500

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
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 25, "Evaluate model on dev set after this many steps (default: 100)")
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


    max_document_length = max([len(x.split(" ")) for x in x_text])
    print("Max Document length:", max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "ELMo" or main_pre_trained_embeddings.Embedding=="Bert"):
        x = x_text
    else:
        x = np.array(list(vocab_processor.fit_transform(x_text)))

    shuffle_indices = np.random.permutation(np.arange(len(y)))

    if CNN:
        if PRETRAINEDEMBEDDING == True and (main_pre_trained_embeddings.Embedding == "ELMo" or main_pre_trained_embeddings.Embedding == "Bert" or main_pre_trained_embeddings.Embedding == 'GloVe' or main_pre_trained_embeddings.Embedding =="fastText"):
            x_shuffled = x
            y_shuffled = y
        else:
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]

    if SVM or NB:
        x_shuffled = x_text
        y_shuffled = y
    # Split train/test set

    if PRETRAINEDEMBEDDING and main_pre_trained_embeddings.Embedding == "Bert":
        import bert_tokenization
        if ITALIAN:
            tokenizer = bert_tokenization.FullTokenizer(
                vocab_file="../Data/bert_checkpoint_multilingual/vocab.txt",
                do_lower_case=True)
        else:
            tokenizer = bert_tokenization.FullTokenizer(vocab_file="../Data/bert_checkpoint/vocab.txt",
                                                        do_lower_case=True)

        tokenized = [tokenizer.tokenize(j) for i, j in enumerate(x_shuffled)]
        x_t = []
        index = 0
        for i in tokenized:
            x_t.append([])
            for j in tokenized:
                x_t[-1].append(index)
                index += 1
        x_shuffled = x_t

    # x_ = [[np.array(i)] for i in range(len(x_train))]

    # SPLIT THE DATASET in 1) labeled training 2) unlabeled 3) validation 4) test


    print("Confidence Threshold: ", confidence)
    print("Dimension of the training set: ", len(x_shuffled[:int(dev_labeled_index * percentage_of_labeled_data)]), percentage_of_labeled_data, "%")
    x_train, x_unlabeled, x_dev = x_shuffled[:int(dev_labeled_index * percentage_of_labeled_data)], x_shuffled[dev_labeled_index:dev_unlabeled_index + dev_labeled_index], x_shuffled[dev_sample_index:]
    y_train, y_unlabeled, y_dev = y_shuffled[:int(dev_labeled_index * percentage_of_labeled_data)], y_shuffled[dev_labeled_index:dev_unlabeled_index + dev_labeled_index], y_shuffled[dev_sample_index:]

    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Unlabeled/Dev split: {:d}/{:d}/{:d}".format(len(y_train),len(y_unlabeled), len(y_dev)))
    return x_text, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev


def train(x_text, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev):
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
            _ = text_clf.fit(x_train, np.ravel(labels))
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

        if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == 'ELMo'):
            import bert_tokenization
            seq = max([len(x.split(" ")) for x in x_train])
        elif PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == 'Bert'):
            seq = max([len(x) for x in x_train])
        else:
            seq = x_train.shape[1]

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == 'ELMo'):
                    import bert_tokenization
                    seq = max([len(x.split(" ")) for x in x_train])
                elif PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == 'Bert'):
                    seq = max([len(x) for x in x_train])
                else:
                    seq = x_train.shape[1]

                cnn = TextCNN(
                    sequence_length=seq,
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    text=x_text,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    binary=BINARY)

                if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "fastText"):
                    max_document_length = max([len(x.split(" ")) for x in x_text])
                    x = []
                    for counter, j in enumerate(x_text):
                        x.append([])
                        x[counter] = []
                        for i in j.split():
                            index = cnn.word2idx.get(i, len(cnn.word2idx) - 1)
                            # print(i,index)
                            x[counter].append(index)
                        while len(x[counter]) < max_document_length:
                            x[counter].append(len(cnn.word2idx) - 1)

                    x = np.array(x)
                    x_train, x_unlabeled, x_dev = x[:int(dev_labeled_index * percentage_of_labeled_data)], x[dev_labeled_index:dev_unlabeled_index + dev_labeled_index], x[dev_sample_index:]

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





                def train_step(x_batch, y_batch, x_unlabeled, y_unlabeled, x, y):
                    """
                    A single training step
                    """
                    # x_batch = tf.cast(x_batch, tf.float32)
                    # a = result + x_batch
                    a = []
                    for i in x:
                        a.append(i)
                    for i in x_batch:
                        a.append(i)
                    b = []
                    for i in y:
                        b.append(i)
                    for i in y_batch:
                        b.append(i)

                    if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "ELMo" or main_pre_trained_embeddings.Embedding=="Bert"):
                        feed_dict = {
                            cnn.unlabeled_training: x_unlabeled,
                            cnn.y_unlabeled: y_unlabeled,
                            cnn.n: len(x_batch),
                            cnn.input_x: a,
                            cnn.x_text: x_batch,
                            cnn.input_y: b,
                            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        }
                    else:

                        feed_dict = {
                            cnn.unlabeled_training: x_unlabeled,
                            cnn.y_unlabeled: y_unlabeled,
                            cnn.n: len(x_batch),
                            cnn.input_x: a,
                            cnn.input_y: b,
                            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        }

                    if TRAIN and COTRAIN:
                        _, step, summaries, loss, accuracy, precision, recall, confusion, cross, next, next_y, maxs, scores, labels, y_predictions, result, accuracy_unlabeled, confusion_unlabeled, scores_unlabeled = sess.run(
                            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.confusion, cnn.cross, cnn.next, cnn.next_y, cnn.maxs, cnn.scores_unlabeled, cnn.predictions, cnn.predict, cnn.results, cnn.accuracy_unlabeled, cnn.confusion_unlabeled, cnn.scores_unlabeled],
                            feed_dict)
                        # print(sess.run(cnn.results, {cnn.x_unlabeled: x_unlabeled}))
                        # print(cnn.scores_unlabeled)
                        # print(scores)
                        print("Number of tweets above confidence threshold: ", len(cross))
                        print(confusion_unlabeled)
                        result = result[1:]
                    else:
                        y_predictions = []
                        result = []
                        next = x_unlabeled
                        next_y = y_unlabeled
                        _, step, summaries, loss, accuracy, precision, recall, confusion = sess.run(
                            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall,
                             cnn.confusion],
                            feed_dict)
                        #                 print(confusion)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, accuracy {:g}, precision {}, recall {}".format(time_str, step, loss, accuracy, precision, recall))
                    train_summary_writer.add_summary(summaries, step)
                    b=[]
                    for i in y:
                        b.append(i)
                    for i in y_predictions:
                        b.append(i)
                    a=[]
                    for i in x:
                        a.append(i)
                    for i in result:
                        a.append(i)

                    return next, next_y, a, b

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
                    step, summaries, loss, accuracy, precision, recall, confusion = sess.run( #, confusion
                        [global_step, dev_summary_op, cnn.loss_dev, cnn.accuracy_dev, cnn.precision_dev, cnn.recall_dev, cnn.confusion_dev],#, cnn.confusion],
                        feed_dict)
                    print(confusion)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, accuracy {:g}, precision {}, recall {}".format(time_str, step, loss, accuracy, precision, recall))
                    if writer:
                        writer.add_summary(summaries, step)

                # Training loop. For each batch...
                if PRETRAINEDEMBEDDING and (main_pre_trained_embeddings.Embedding == "ELMo"):
                    x_ = [[np.array(i)] for i in range(len(x_train))]
                    batches = preprocess.batch_iter(list(zip(x_, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                else:
                    batches = preprocess.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)


                unlabeled_training = []
                for i in x_unlabeled:
                    unlabeled_training.append([])
                    for j in i:
                        unlabeled_training[-1].append(j)
                result = []
                result2 = []
                y_prediction = []
                y_prediction2 = []
                current_step = 0
                n = n_prev = 0
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

                    print("New CNN run:", current_step)

                    # Initialize all variables
                    if PRETRAINEDEMBEDDING and (
                            main_pre_trained_embeddings.Embedding == "fastText" or main_pre_trained_embeddings.Embedding == "GloVe" or main_pre_trained_embeddings.Embedding == "Bert"):
                        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), embedding_init],
                                 feed_dict={
                                     cnn.embedding_placeholder: cnn.weights})
                    else:  # if not PRETRAINEDEMBEDDING:
                        sess.run(tf.global_variables_initializer())

                    for batch in batches:
                        x_batch, y_batch = zip(*batch)
                        current_step = tf.train.global_step(sess, global_step) + 1

                        COTRAIN = False

                        if PRETRAINEDEMBEDDING and main_pre_trained_embeddings.Embedding == "ELMo":
                            x_t = []
                            for i in x_batch:
                                x_t.append(x_train[i[0]])
                            for i in range(len(x_t)):
                                while len(x_t[i].split()) < seq:
                                    x_t[i] += ' unk '
                            _, _, _, _ = train_step(x_t, y_batch, unlabeled_training, y_unlabeled, [], [])
                        else:
                            _, _, _, _ = train_step(x_batch, y_batch, unlabeled_training, y_unlabeled, [], [])
                        if current_step % FLAGS.evaluate_every == 0:
                            print("\nEvaluation:")

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
                            unlabeled_training, y_unlabeled, result2, y_prediction2 = train_step(x_batch, y_batch, unlabeled_training, y_unlabeled, [], [])
                            n_prev = len(result)
                            n = len(result2)
                            print("Added " + str(n) + " to Training Set. " + str(n_prev+n) + "/" + str(len(x_unlabeled)))
                            COTRAIN = False
                            break




def main(argv=None):
    x_text, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev = preprocess_()
    train(x_text, x_train, x_unlabeled, x_dev, vocab_processor, y_train, y_unlabeled, y_dev)

if __name__ == '__main__':
    tf.app.run()
