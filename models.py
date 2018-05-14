import tensorflow as tf


def encoder(inputs, channels, initial_state=None, initialize_to_zero=True):
    batch_size = tf.shape(inputs)[1]
    input_shape = inputs.shape.as_list()[2:]

    encoder_lstm_cells = [tf.contrib.rnn.Conv2DLSTMCell(input_shape=input_shape,
                                                        kernel_shape=[3, 3],
                                                        output_channels=num_channels)
                          for num_channels in channels]
    encoder_lstm = tf.contrib.rnn.MultiRNNCell(encoder_lstm_cells)

    if initialize_to_zero:
        initial_state = encoder_lstm.zero_state(batch_size, tf.float32)

    all_encoder_states, final_encoder_state = tf.nn.dynamic_rnn(cell=encoder_lstm,
                                                                inputs=inputs,
                                                                initial_state=initial_state,
                                                                dtype=tf.float32,
                                                                time_major=True)
    return all_encoder_states, final_encoder_state


def output_layers(decoder_state):
    with tf.variable_scope("output"):
        predictions_flat = tf.contrib.slim.conv2d(decoder_state, 16, 3, padding='SAME', activation_fn=tf.nn.relu)
        predictions_flat = tf.contrib.slim.conv2d(predictions_flat, 8, 3, padding='SAME', activation_fn=tf.nn.relu)
        predictions_flat = tf.contrib.slim.conv2d(predictions_flat, 1, 1, padding='SAME', activation_fn=None)

    return predictions_flat
