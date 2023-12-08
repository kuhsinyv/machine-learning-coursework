"""matplotlib utils."""
from matplotlib import pyplot as plt


def show_images(images, title_texts):
    """show gray images."""
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap="gray")
        if (title_text != ''):
            plt.title(title_text, fontsize=15)
        index += 1
