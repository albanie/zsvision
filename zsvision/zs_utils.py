import os
import glob
import fnmatch


def get_img_paths_in_dir(image_dir, suffix='jpg'):
    """
    returns list of paths to images in the given
    directory.

    Args:
        image_dir (string): path to the directory
            containing the images
        suffix (string): the suffix used to select
            the images (default is `jpg`)

    Returns:
        [string]: a list of paths

    Note: Hidden system files are ignored
    """
    image_dir = os.path.expanduser(image_dir)
    glob_template = os.path.join(image_dir, '*.{}'.format(suffix))
    paths = glob.glob(glob_template)
    return paths


def get_img_paths_in_subdirs(image_dir, suffix='jpg'):
    """
    returns list of paths to images in the subdirectories
    given of the given directory.

    based on a StackOverflow answer by Johan Dahlin

    Args:
        image_dir (string): path to the directory
            containing the subdirs
        suffix (string): the suffix used to select
            the images (default is `jpg`)

    Returns:
        [string]: a list of paths

    Note: Hidden system files are ignored
    """
    image_dir = os.path.expanduser(image_dir)
    paths = []
    for root, dirnames, fnames in os.walk(image_dir):
        for fname in fnmatch.filter(fnames, '*.{}'.format(suffix)):
            paths.append(os.path.join(root, fname))
    return paths
