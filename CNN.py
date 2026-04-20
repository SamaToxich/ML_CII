import sys, numpy as np
from keras.datasets import mnist
from numpy.lib._stride_tricks_impl import as_strided

alpha = 2
iterations = 200
pixels_pre_image = 28*28
num_labels = 10
batch_size = 128

rows_input = 28
cols_input = 28

rows_kernel = 3
cols_kernel = 3

num_kernels = 16

hidden_size = (rows_input - rows_kernel + 1) * (cols_input - cols_kernel + 1) * num_kernels

weights_0 = 0.02 * np.random.random((rows_kernel * cols_kernel, num_kernels)) - 0.01
weights_1 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = x_train[0:1000].reshape(1000,pixels_pre_image) / 255
labels = y_train[0:1000]

one_hot_labels = np.zeros((len(labels), num_labels))

test_images = x_test[0:100].reshape(len(x_test[0:100]), pixels_pre_image) / 255
test_labels = np.zeros((len(y_test[0:100]), num_labels))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

for i, l in enumerate(y_test[0:100]):
    test_labels[i][l] = 1

np.random.seed(1)


def tanh(x):
    return np.tanh(x)

def tanhRevers(x):
    return 1 - (x ** 2)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

def get_image_section(layer, row_start, row_end, col_start, col_end):
    section = layer[:, row_start:row_end, col_start:col_end]
    return section.reshape(-1, 1, row_end-row_start, col_end-col_start) # Условно было shape = (1,3,3) стало (1,1,3,3)

def im2col(images, kernel_h, kernel_w):
    """Быстрый im2col для батча"""
    batch, h, w = images.shape
    out_h = h - kernel_h + 1
    out_w = w - kernel_w + 1

    shape = (batch, out_h, out_w, kernel_h, kernel_w)
    strides = (images.strides[0], images.strides[1], images.strides[2],
               images.strides[1], images.strides[2])

    windows = as_strided(images, shape=shape, strides=strides)
    return windows.reshape(batch * out_h * out_w, kernel_h * kernel_w)

def my_im2col(l, r, c):
    sects = list()

    for row_start in range(l.shape[1] - r + 1):
        for col_start in range(l.shape[2] - c + 1):
            sect = get_image_section(l, row_start, row_start + r, col_start, col_start + c)
            sects.append(sect)

    l = np.concatenate(sects, axis=1) # Объединяем условно отдельные 9 строк c shape = (1,1,3,3) в одну с shape = (1,9,3,3)
    l = l.reshape(l.shape[0] * l.shape[1], -1) # Делаем из них строки было shape = (1,9,3,3) стало shape = (9,9)
    return l


for j in range(iterations):
    correct_cnt = 0

    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = i * batch_size, (i+1) * batch_size

        layer_0 = (images[batch_start:batch_end]).reshape(-1, 28, 28)
        layer_0 = im2col(layer_0, rows_kernel, cols_kernel)

        layer_1 = tanh(layer_0 @ weights_0)
        layer_1 = layer_1.reshape(batch_size,-1)

        regular_drop = np.random.randint(2, size=layer_1.shape)
        layer_1 = (layer_1 * regular_drop * 2)

        layer_2 = softmax(layer_1 @ weights_1)

        predictions = np.argmax(layer_2, axis=1)
        true_labels = np.argmax(labels[batch_start:batch_end], axis=1)
        correct_cnt += np.sum(predictions == true_labels)

        layer_2_delta = (layer_2 - labels[batch_start:batch_end]) / (batch_size * layer_2.shape[0])

        layer_1_delta = layer_2_delta @ weights_1.T * tanhRevers(layer_1)
        layer_1_delta *= regular_drop

        weights_1 -= alpha * layer_1.T @ layer_2_delta
        weights_0 -= alpha * layer_0.T @ layer_1_delta.reshape(-1, num_kernels)

    test_correct_cnt = 0

    for i in range(len(test_images)):
        layer_0 = (test_images[i:i+1]).reshape(-1, 28, 28)
        layer_0 = im2col(layer_0, rows_kernel, cols_kernel)

        layer_1 = tanh(layer_0 @ weights_0).reshape(1,-1)
        layer_2 = layer_1 @ weights_1

        predictions = np.argmax(layer_2, axis=1)
        true_labels = np.argmax(test_labels[i:i+1], axis=1)
        test_correct_cnt += np.sum(predictions == true_labels)

    if(j % 10 == 0):
        sys.stdout.write("\n"+ \
                         "I:" + str(j) + \
                         " Test-Acc:"+str(test_correct_cnt/float(len(test_images)))+ \
                         " Train-Acc:" + str(correct_cnt/float(len(images))))
