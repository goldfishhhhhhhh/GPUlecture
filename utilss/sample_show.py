import PIL
import matplotlib.pyplot as plt

def showimage(inputpath):
    plt.imshow(PIL.Image.open(inputpath))
    plt.axis('off')  # Turn off axis numbers
    plt.show()
