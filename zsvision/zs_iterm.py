import os
import subprocess
import matplotlib.pyplot as plt

IMGCAT = os.path.expanduser('~/.scripts/imgcat')

def zs_dispFig(imgcat_path=IMGCAT):
    """
    shows a matplotlib plot inline in iTerm by saving 
    it to a temporary file, displaying the file and 
    then cleaning up.

    Note: This function requires the use of an iTerm
    terminal (available from https://www.iterm2.com)
    and requires that the `imgcat` script is on your
    $PATH. It can be useful when running MATLAB on a
    server when you don't have access to a GUI to
    display figures.

    Copyright (C) 2016 Samuel Albanie
    All rights reserved.
    """

    # save figure as temp image
    im_name = '_tmp.jpeg'

    # switch to png if jpeg is unsupported
    try:
        plt.savefig(im_name)
    except ValueError:
        im_name = '_tmp.png'
        plt.savefig(im_name)

    # display in iterm
    subprocess.call([imgcat_path, im_name])
    
    # clear up
    os.remove(im_name)
