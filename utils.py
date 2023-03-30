import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, filename=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)

    plt.figure(figsize=(6, 6))
    
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    
    if filename:
        plt.savefig(filename)
    
    plt.show()