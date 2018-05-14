import tensorflow as tf
import numpy as np
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
batch_size = 5
num_train_batches = 100000000   # number of training batches before training terminates
print_step = 10                 # example output is saved every `print_step` batches
length_step = 10000             # max output sequence length increases by 1 every `length_step` batches, starting at 1
save_step = 5000                # model parameters are saved every `save_step` batches


def get_uninitialized_variables(session):
    variables = tf.global_variables()
    init_flag = session.run(
        [tf.is_variable_initialized(v) for v in variables]
    )
    return [v for v, f in zip(variables, init_flag) if not f]


def build_network(input_sequences, output_sequences, initial_state=None, initialize_to_zero=True):
    batch_size = tf.shape(input_sequences)[1]
    input_sequences_rs = tf.expand_dims(input_sequences, axis=-1)
    num_prediction_steps = sequence_length - tf.shape(input_sequences)[0]

    # encoder network
    encoder_channels = [32, 16]
    encoding_channels = encoder_channels[-1]
    with tf.variable_scope("encoder"):
        all_encoder_states, final_encoder_state = encoder(inputs=input_sequences_rs,
                                                          channels=encoder_channels,
                                                          initial_state=initial_state,
                                                          initialize_to_zero=initialize_to_zero)
        encoder_saver = tf.train.Saver()

    # decoder network
    # uses a tf.while_loop to store the predictions in a tf.TensorArray
    # The decoder state is initialized to the final vailes of the encoder state,
    # and the final output of the encoder is used as input to the decoder
    decoder_channels = encoder_channels[::-1]
    with tf.variable_scope("decoder") as scope:
        decoder_lstm_cells = [tf.contrib.rnn.Conv2DLSTMCell(input_shape=[image_height, image_width, encoding_channels],
                                                            kernel_shape=[3, 3],
                                                            output_channels=num_channels)
                              for num_channels in decoder_channels]

        decoder_lstm = tf.contrib.rnn.MultiRNNCell(decoder_lstm_cells)

        # array to store outputs
        init_prediction_sequence = tf.TensorArray(tf.float32, size=tf.shape(output_sequences)[0])

        # use the last encoder state to initialize the first decoder state
        init_decoder_state = final_encoder_state[::-1]

        def condition(step, _, __, ___):
            return tf.less(step, num_prediction_steps)

        def body(step, state, input, prediction_sequence):
            decoder_state, new_state = decoder_lstm(input, state)
            new_prediction = output_layers(decoder_state)

            # ensure variables in output layers are reused
            scope.reuse_variables()

            return tf.add(step, 1), new_state, new_prediction, prediction_sequence.write(step, new_prediction)

        init = (tf.constant(0, name="i"),
                init_decoder_state,
                input_sequences_rs[-1, :, :, :, :],
                init_prediction_sequence)

        i, final_decoder_state, _, predictions_ta = tf.while_loop(condition, body, init)
        predictions_flat = predictions_ta.concat()

        # predictions: [time, batch, height, width, 1]
        # contains the sequence predicted to follow the encoder input
        predictions = tf.reshape(predictions_flat, (-1, batch_size, image_height, image_width))

    return predictions, predictions_flat, final_encoder_state, encoder_saver


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
        predictions, predictions_ta, final_encoder_state, encoder_saver = build_network(input_sequences,
                                                                                        output_sequences)

        # loss and training
        with tf.variable_scope("trainer"):
            loss = tf.reduce_mean((predictions - output_sequences)**2)
            trainer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

        saver = tf.train.Saver()

    sess = tf.InteractiveSession()

    # uncomment to restore encoder variables (remember to set restore path)
    # print("restoring encoder variables")
    # encoder_saver.restore(sess, "restore/encoder/it_700/encoder")

    # uncomment to restore all variables
    print("restoring all variables...")
    saver.restore(sess, "restore/predictor/it_475000/predictor")

    # initialize remaining variables
    uninitialized_variables = get_uninitialized_variables(sess)
    if len(uninitialized_variables) > 0:
        print("Variables not found in resore path: {}".format(uninitialized_variables))
        sess.run(tf.variables_initializer(uninitialized_variables))
    else:
        print("all variables restored")

    # get example batch for visualizing progress
    feed = {input_sequences: data[:10, :1, :, :],
            output_sequences: data[10:, :1, :, :]}

    test_input, test_predictions = sess.run([input_sequences, predictions], feed)

    visualization.save_animation(test_input[:, 0, :, :], test_predictions[:, 0, :, :])

    # construct summaries for tensorboard
    tf.summary.scalar('batch_loss', loss)
    summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('tensorboard/' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sess.graph)

    # get batch from data, assumes shape [time, batches, height, width]
    def generate_batch(min_length, randomize=False):
        if randomize:
            length = np.random.randint(min_length, sequence_length)
        else:
            length = min_length

        batch_inds = np.random.choice(data.shape[1], batch_size, replace=False)
        return data[:length, batch_inds, :, :], data[length:, batch_inds, :, :]

    input_sequence_length = 11
    for i in range(num_train_batches):
        last_time = time.time()

        batch_input, batch_output = generate_batch(input_sequence_length, randomize=False)

        batch_feed = {input_sequences: batch_input,
                      output_sequences: batch_output}

        batch_summary, batch_loss, _, batch_predictions = sess.run([summaries, loss, trainer, predictions],
                                                                   feed_dict=batch_feed)

        summary_writer.add_summary(batch_summary, i)
        sys.stdout.write("\rIteration: %i - loss %f - batches/s: %f" % (i, batch_loss, 1. / (time.time() - last_time)))
        sys.stdout.flush()

        if i % print_step == 0:
            # save example prediction sequence, showing the frames given to the network followed by the predictions
            test_predictions = sess.run(predictions, feed)
            visualization.save_animation(test_input[:, 0, :, :], test_predictions[:, 0, :, :])

        if i > 0 and i % 10000 == 0 and input_sequence_length > 11:
            input_sequence_length -= 1

        if i % save_step == 0:
            saver.save(sess, "restore/predictor/it_{}/predictor".format(i))


if __name__ == '__main__':
    run()
