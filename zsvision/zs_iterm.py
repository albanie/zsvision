import os
import subprocess
import matplotlib.pyplot as plt

IMGCAT = os.path.expanduser('~/local/bin/imgcat')


def zs_dispFig(imgcat_path=IMGCAT):
    """Shows a matplotlib plot inline in iTerm by saving it to a temporary file,
    displaying the file and then cleaning up.

    Note: This function requires the use of an iTerm terminal (available from
    https://www.iterm2.com) and requires that the `imgcat` script is on your $PATH. It
    can be useful when running code on a server when you don't have access to a GUI to
    display figures.

    Copyright (C) 2016 Samuel Albanie
    All rights reserved.
    """

    # save figure as temp image
    im_name = '_tmp.png'
    plt.savefig(im_name)

    # display in iterm
    subprocess.call([imgcat_path, "--depth", "iterm2", im_name])

    # clear up
    os.remove(im_name)
