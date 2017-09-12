# Load the modules
import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from etl import load_data
import itertools

# PART 1: Tensorflor setup

# Step 1: ETL data into numpy arrays
train_features, train_labels, valid_features, valid_labels, test_features, test_labels = load_data()

# Step 2: Set up placeholders for features, labels, weights and biases
features_count = train_features.shape[1]
labels_count = train_labels.shape[1]

features = tf.placeholder(tf.float32, [None, features_count])
labels = tf.placeholder(tf.float32, [None, labels_count])

weights = tf.Variable(tf.truncated_normal([features_count, labels_count]))
biases = tf.Variable(tf.zeros(labels_count))

# Step 3: Set up feed_dict for train, validation, test data
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

# Step 4: Forward propagation
logits = tf.add(tf.matmul(features,weights),biases)
prediction = tf.nn.softmax(logits)

# Step 5: Loss calculation
cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), axis=1)
loss = tf.reduce_mean(cross_entropy)

# Step 6: Accurracy calculation
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

init = tf.global_variables_initializer()

# PART 2: executing model

def run_one_config(epochs, batch_size, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(train_features)/batch_size))

        for epoch_i in range(epochs):
            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i*batch_size
                batch_features = train_features[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]

                # Run optimizer and get loss
                _, l = session.run(
                    [optimizer, loss],
                    feed_dict={features: batch_features, labels: batch_labels})

            # Check accuracy against Validation data
            validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)
    return validation_accuracy

def run_test(epochs, batch_size, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(train_features)/batch_size))
        for epoch_i in range(epochs):
            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i*batch_size
                batch_features = train_features[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]

                # Run optimizer
                _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

            # Check accuracy against Test data
            test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)
    return test_accuracy

config2 = list(itertools.product(*[[1],[100],[.5,.1,.05]]))
config3 = list(itertools.product(*[[1,2,3,4,5],[100],[.2]]))
configs = config2 + config3

# Measurements use for graphing loss and accuracy
best_acc = 0

for epochs, batch_size, learning_rate in configs:
    print("epochs: ", epochs, ". batch_size: ", batch_size, ". learning_rate: ", learning_rate)

    validation_accuracy = run_one_config(epochs, batch_size, learning_rate)
    if validation_accuracy > best_acc:
        best_acc = validation_accuracy
        best_acc_params = (epochs, batch_size, learning_rate)

    print('Validation accuracy at {}'.format(validation_accuracy))
    print("Best accuracy so far: ", best_acc, " with: ", best_acc_params)
    print("=====")

best_epochs, best_batch_size, best_learning_rate = best_acc_params
test_accuracy = run_test(best_epochs, best_batch_size, best_learning_rate)

print('Test Accuracy is {}'.format(test_accuracy))
