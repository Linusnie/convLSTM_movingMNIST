import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import datetime

import visualization
from models import encoder, output_layers

data = np.load("mnist_test_seq.npy")
sequence_length = data.shape[0]
image_height = data.shape[2]
image_width = data.shape[3]

# parameters
input_sequence_length = 10
batch_size = 1
num_train_batches = 100000
print_step = 50
num_prediction_steps = 20


def build_network(input_sequences, initial_state=None, initialize_to_zero=True):
    input_sequences_rs = tf.expand_dims(input_sequences, axis=-1)

    # encoder network
    encoder_channels = [32, 16]
    encoding_channels = encoder_channels[-1]
    with tf.variable_scope("encoder"):
        all_encoder_states, final_encoder_state = encoder(inputs=input_sequences_rs,
                                                          channels=encoder_channels,
                                                          initial_state=initial_state,
                                                          initialize_to_zero=initialize_to_zero)
        encoder_saver = tf.train.Saver()

    # additional output network
    predictions_flat = tf.reshape(all_encoder_states,
                                  shape=(-1, image_height, image_width, encoding_channels))
    predictions_flat = output_layers(predictions_flat)

    predictions = tf.reshape(predictions_flat, tf.shape(input_sequences))

    return predictions_flat, predictions, final_encoder_state, encoder_saver


def run():
    # placeholders
    input_sequences = tf.placeholder(dtype=tf.float32,
                                     shape=(None, None, image_height, image_width))

    output_sequences = tf.placeholder(dtype=tf.float32,
                                      shape=(None, None, image_height, image_width))
    output_sequences_rs = tf.reshape(output_sequences,
                                     shape=(-1, image_height, image_width, 1))

    # build network
    with tf.variable_scope("convlstm") as scope:
        predictions_flat, predictions, final_encoder_state, encoder_saver = build_network(input_sequences)

        # loss and training
        with tf.variable_scope("trainer"):
            loss = tf.reduce_mean((predictions_flat - output_sequences_rs) ** 2)
            trainer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        scope.reuse_variables()

        # repeatedly apply lstm to get long term prediction (doesn't work very well)
        def condition(step, _, __, ___):
            return tf.less(step, num_prediction_steps)

        def body(step, input, state, prediction_sequence):
            _, new_predictions, new_state, _ = build_network(input, state, False)

            return tf.add(step, 1), \
                   new_predictions[:1, :, :, :], \
                   new_state, \
                   prediction_sequence.write(step, new_predictions[:1, :, :, :])

        init = (tf.constant(0, name="i"),
                predictions[-1:, :, :, :],
                final_encoder_state,
                tf.TensorArray(tf.float32, num_prediction_steps))

        _, _, _, new_predictions_ta = tf.while_loop(condition, body, init)
        new_predictions = new_predictions_ta.concat()

    full_saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # uncomment to load pre-trained variables
    full_saver.restore(sess, "restore/full/it_700/full")

    # construct summaries for tensorboard
    tf.summary.scalar('batch_loss', loss)
    summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('tensorboard/' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                           sess.graph)

    # get batch from data, assumes shape [time, batches, height, width]
    def generate_batch():
        batch_inds = np.random.choice(data.shape[1], batch_size, replace=False)
        return data[:-1, batch_inds, :, :], data[1:, batch_inds, :, :]

    fig, axes = plt.subplots(1, 1)
    for i in range(num_train_batches):
        last_time = time.time()
        input_batch, output_batch = generate_batch()
        batch_feed = {input_sequences: input_batch,
                      output_sequences: output_batch}

        batch_summary, batch_loss, _, batch_predictions, batch_new_predictions = \
            sess.run([summaries, loss, trainer, predictions, new_predictions],
                     feed_dict=batch_feed)

        summary_writer.add_summary(batch_summary, i)
        sys.stdout.write("\rIteration: %i - loss %f - batches/s: %f" % (i, batch_loss, 1. / (time.time() - last_time)))
        sys.stdout.flush()

        if i % print_step == 0:
            full_saver.save(sess, "restore/full/it_{}/full".format(i))
            encoder_saver.save(sess, "restore/encoder/it_{}/encoder".format(i))

            # plot a visualization of the performance, red is the current frame (seen by the network)
            # and green is the one-step-ahead prediction
            visualization.animate_anaglyph_comparison(input_batch[:, 0, :, :], batch_predictions[:, 0, :, :], axes)


if __name__ == '__main__':
    run()
