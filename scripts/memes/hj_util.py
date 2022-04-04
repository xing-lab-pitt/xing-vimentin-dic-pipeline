import glob
import os
import shutil
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
from itertools import chain


def folder_verify(folder):
    """Verify if the folder string is ended with '/' """
    if folder[-1] != "/":
        folder = folder + "/"
    return folder


def folder_file_num(folder, pattern="*"):
    """How many files in the folder"""
    if folder[-1] != "/":
        folder = folder + "/"
    file_list = sorted(glob.glob(folder + "*" + pattern + "*"))
    print("%s " % folder + "has %s files" % len(file_list))
    return file_list


def create_folder(folder):
    """Create a folder. If the folder exist, erase and re-create."""
    folder = folder_verify(folder)

    if os.path.exists(folder):  # recreate folder every time.
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)
    print("%s folder is freshly created. \n" % folder)


def file_copy(start_folder, pattern, num_of_files, end_folder):
    """
    Copy a list of files to destinated folder.
    """
    file_list = folder_file_num(start_folder, pattern)[:num_of_files]

    if end_folder[-1] != "/":
        end_folder = end_folder + "/"

    if os.path.exists(end_folder):  # recreate folder every time.
        shutil.rmtree(end_folder)
        os.makedirs(end_folder)
    else:
        os.makedirs(end_folder)

    for f in file_list:
        shutil.copy(f, end_folder)

    print("The copy processes have completed.")


# ------------------- Image Show --------------------


def folder_img_show(folder, grey=1):

    """
    A convient function for print pictures under a folder.
    20 pics at a time.
    """
    fig = plt.figure(figsize=(20, 20))
    img_path_list = sorted(glob.glob(folder + "/*"))

    if len(img_path_list) < 20:
        print("Not enough pictures to print")
        return 1

    for i, img_path in enumerate(img_path_list[:20]):
        fig.add_subplot(4, 5, i + 1)
        plt.imshow(imread(img_path) / 256.0, cmap="gray")
    plt.show()


def list_histimg_show(list_2d, bin_num, fig_num, w_num, h_num, w=20, h=20):

    """
    A convient function for printing histogram for a 2d list.
    list_2d - a list of list of number. For each number list, plot a histograph.
    fig_num - int, should be smaller than dimention of list_2d.
    """
    flat_list = np.log(np.array(list(chain.from_iterable(list_2d))) + 1)
    hist, bin_edges = np.histogram(flat_list, bin_num)

    fig = plt.figure(figsize=(w, h))
    if len(list_2d) < fig_num:
        print("Not enough pictures to print")
        return 1

    for i, ls in enumerate(list_2d[:fig_num]):
        fig.add_subplot(w_num, h_num, i + 1)
        plt.hist(np.log(np.array(ls) + 1), bin_edges)
    plt.show()


def list_label_show(label_list, fig_num, w_num, h_num, costom_cmap="coolwarm", w=20, h=20):
    """
    A convient function for printing label images
    """
    fig = plt.figure(figsize=(w, h))
    if len(label_list) < fig_num:
        print("Not enough pictures to print")
        return 1

    for i, label in enumerate(label_list[:fig_num]):
        fig.add_subplot(w_num, h_num, i + 1)
        plt.imshow(label, cmap=costom_cmap)
    plt.show()


# def img_mask_show(img


# -------------------------- General ------------------------------


def list_select_by_index(the_list, index_list):
    """Return a list of selected items. """
    selected_elements = [the_list[index] for index in index_list]
    return selected_elements
