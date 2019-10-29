import cv2
import numpy as np
import matplotlib.pyplot as plt

#convert intensity to binary values
def int2bin(integer):
    bin = np.zeros((8))
    k = integer
    for i in range(8):
        bin[7 - i] = k % 2
        k = k // 2
    return bin

#Plot result
def show_images(images, cols=1, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.savefig('jsdfjsf.png')

img = cv2.imread('100000t.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('mot_tram_nghin.jpg',img)
bimg = np.zeros((img.shape[0], img.shape[1], 8))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        bimg[i, j] = int2bin(img[i, j])

layer0 = bimg[:,:,7]
layer1 = bimg[:,:,6]
layer2 = bimg[:,:,5]
layer3 = bimg[:,:,4]
layer4 = bimg[:,:,3]
layer5 = bimg[:,:,2]
layer6 = bimg[:,:,1]
layer7 = bimg[:,:,0]

#Display Result
images_list = (layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7)
titles_list = ('layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7')

show_images(images_list, cols = 3 ,titles=titles_list)

cv2.waitKey(0)
cv2.destroyAllWindows()