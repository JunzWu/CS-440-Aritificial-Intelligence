# import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def plot_single_line(x_li, y_li, xlabel='', ylabel='', xticks=None, title=''):
    '''plot single line'''
    fig, ax = plt.subplots()

    ax.plot(x_li, y_li, '-')
    ax.set_title(title)
    # xtick range shoud be resonable
    if xticks:
        plt.xticks(xticks)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def display_2d_list(li, ele_length=6):
    '''display 2-dim list'''
    r, c = len(li), len(li[0])
    FORMAT = '{0:^%d}' % (ele_length)

    for i, row in enumerate(li):
        if i == 0:
            print('[', end='')
        else:
            print(' ', end='')

        print('[', end='')
        for col in row:
            print(FORMAT.format(col), end='')

        if i == r - 1:
            print(']]')
        else:
            print(']')
