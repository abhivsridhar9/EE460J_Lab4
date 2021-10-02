import gzip

import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import gzip
def q1():
    dataset = sklearn.datasets.fetch_openml("CIFAR_10_small", data_home="./data", cache=True)
    #dataset = gzip.decompress("./data/openml/openml.org/data/v1/download/1679612.gz")
    data = dataset["data"].iloc[145].tolist()
    classifications = dataset["target"].iloc[145]
    data = np.array(data)
    #print(data)
    data = data.reshape(3, 32, 32)
    im = np.transpose(data, axes=[1, 2, 0])
    im = np.uint8(im)
    print(data)
    #print(img)
    #img = np.array(img)
    plt.imshow(Image.fromarray(im))
    plt.title(classifications)
    plt.show()

    return
if __name__ == '__main__':
    q1()