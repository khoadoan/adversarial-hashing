from matplotlib import pyplot as plt
from matplotlib import animation

import torchvision.utils as vutils

import numpy as np

def plot_torch_images(images, n=16, nrow=16, filename=None, figsize=(10,10), normalize=False, title=None):
    grid_images = vutils.make_grid(images[:n], padding=4, pad_value=1, normalize=normalize, nrow=nrow).detach().cpu()
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(np.transpose(grid_images,(1,2,0)))
    if title:
        plt.title(title)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_image(image, title=None, figsize=(8, 8), filename=None):

    plt.figure(figsize=figsize)
    # Plot the real images
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(image, (1, 2, 0)))

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
        
def plot_images(image_sets, titles, figsize=(8, 8), ncols = 1, nrows = 1, filename=None):
    n = len(image_sets)

    assert n <= ncols * nrows

    plt.figure(figsize=figsize)

    for idx, (images, title) in enumerate(zip(image_sets, titles)):
        # Plot the real images
        plt.subplot(nrows, ncols, idx % ncols + 1)
        plt.axis("off")
        plt.title(title)
        plt.imshow(np.transpose(images, (1, 2, 0)))

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def plot_animation(img_list, filename):
    Writer = animation.writers['html']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # Visualization progression
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save(filename, writer=writer)


def plot_losses(losses, title, figsize=(10,5),
                filename=None, colors=['red', 'blue', 'purple'],
                xlabel=None, ylabel=None):
    n = len(losses)

    # Losses during training
    plt.figure(figsize=figsize)
    plt.title(title)
    for idx, (loss_name, loss) in enumerate(losses.iteritems()):
        plt.plot(loss, label=loss_name, color=colors[idx])
        plt.xlabel(xlabel)
        plt.xlabel(ylabel)

    plt.legend()

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()