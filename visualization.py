from matplotlib import pyplot as plt
import numpy as np
import os

cwd = os.getcwd()


def plot_encoding(enc, axes=None):
    num_rows = int(np.sqrt(enc.shape[-1]))

    if not type(axes) == np.ndarray:
        fig, axes = plt.subplots(num_rows, num_rows)

    i = 0
    for axrow in axes:
        for ax in axrow:
            if len(ax.images) == 0:
                ax.imshow(enc[:, :, i])
            else:
                ax.images[0].set_data(enc[:, :, i])
            i += 1
    plt.draw()
    plt.pause(0.1)


def animate_prediction(input, output, ax, fig):
    num_input_frames = input.shape[0]
    num_output_frames = output.shape[0]

    for i in range(num_input_frames):
        ax.imshow(input[i, :, :])
        fig.canvas.draw()
        fig.canvas.pause(0.1)

    for i in range(num_output_frames):
        ax.imshow(output[i, :, :])
        fig.canvas.draw()
        fig.canvas.pause(0.1)


def animate_anaglyph_comparison(in_data, out_data, ax):
    ax.clear()
    input = in_data.astype(np.float32) / 255.
    output = out_data.astype(np.float32) / 255.

    num_frames = input.shape[0]
    img = np.zeros((input.shape[1], input.shape[2], 3))

    for i in range(num_frames):
        img[:, :, 1] = output[i, :, :]
        img[:, :, 0] = input[i, :, :] * (1 - output[i, :, :])

        img -= img.min()
        img /= img.max()
        img[:, :, 2] = 0

        if len(ax.images) == 0:
            ax.imshow(img)
        else:
            ax.images[0].set_data(img)
        plt.draw()
        plt.pause(0.1)


# sequence: [time, height, width]
def animate_sequence(sequence, last_known=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)

    num_frames = sequence.shape[0]
    for i in range(num_frames):
        # imshow = ax.imshow if len(ax.images) == 0 else ax.images[0].set_data
        if type(last_known) is np.ndarray:
            last_known = last_known.astype(np.float32) / last_known.max()
            img = sequence[i, :, :] * (1 - last_known)
        else:
            img = sequence[i, :, :]

        ax.imshow(img)

        plt.draw()
        plt.pause(0.1)


def save_animation(in_sequence, out_sequence):
    num = 0
    fig, ax = plt.subplots(1, 1)
    for i in range(in_sequence.shape[0]):
        ax.imshow(in_sequence[i, :, :])

        plt.savefig(os.path.join(cwd, "images", "frame{}".format(num)))

        num += 1

    for i in range(out_sequence.shape[0]):
        ax.imshow(out_sequence[i, :, :])

        plt.savefig(os.path.join(cwd, "images", "frame{}".format(num)))

        num += 1





