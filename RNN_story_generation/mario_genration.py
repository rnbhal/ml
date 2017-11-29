import matplotlib.pyplot as plt
import numpy as np

data = open('mario_level.txt', 'r').read()
def addtext(data,props):
    plt.text(50, 50, data, props, rotation=0)
    plt.yticks([0, 100, 1])
    plt.grid(True)

def addtext1(data,props):
    plt.text(50, 50, data, props, rotation=90)
    plt.yticks([0, 100, 1])
    plt.grid(True)
# the text bounding box
bbox = {'fc': '0.8', 'pad': 0}

plt.subplot(211)
addtext(data,{'ha': 'center', 'va': 'center', 'bbox': bbox})
plt.xlim(0, 100)
plt.xticks(np.arange(0, 100, 1), [])
plt.ylabel('center / center')


plt.show()
