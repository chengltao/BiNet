import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, dic_bac, title='Fusion_Nomask', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(dic_bac.keys())))
    xloc = []
    for ii, jj in enumerate(list(xlocations)):
        for k, v in dic_bac.items():
            if v == jj:
                xloc.append(k)
                break
    plt.xticks(xlocations, xloc, rotation=90) #label x axis
    plt.yticks(xlocations, xloc) #label y axis
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def draw_result(target, predict_y, dic_bac):
    a = np.array(target)
    b = np.array(predict_y)
    accuracy = sum(a == b) / len(target)
    print("accuracy----------", accuracy)
    # acquiring confusion matrix
    cm = confusion_matrix(target, predict_y, labels=range(0, len(dic_bac.keys())))
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)

    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(dic_bac.keys()))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    # offset the tick
    tick_marks = np.array(range(len(dic_bac.keys()))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    # drawing confusion matrix
    plot_confusion_matrix(cm_normalized, dic_bac, title='Normalized confusion matrix')
    plt.show()