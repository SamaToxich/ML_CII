import sys, numpy as np
from keras.datasets import mnist

alpha = 0.0005
iterations = 200
pixels_pre_image = 28*28
num_labels = 10
batch_size = 1

rows_input = 28
cols_input = 28

rows_kernel = 3
cols_kernel = 3

num_kernels = 16

hidden_size = (rows_input - rows_kernel) * (cols_input - cols_kernel) * num_kernels

weights_0 = 0.02 * np.random.random((rows_kernel * cols_kernel, num_kernels)) - 0.01
weights_1 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = x_train[0:1000].reshape(1000,pixels_pre_image) / 255
labels = y_train[0:1000]

one_hot_labels = np.zeros((len(labels), num_labels))

test_images = x_test.reshape(len(x_test), pixels_pre_image) / 255
test_labels = np.zeros((len(y_test), num_labels))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

for i, l in enumerate(y_test):
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

for j in range(iterations):
    correct_cnt = 0

    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = i * batch_size, (i+1) * batch_size

        layer_0 = images[batch_start:batch_end]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

        sects = list()
        for row_start in range(layer_0.shape[1] - rows_kernel):
            for col_start in range(layer_0.shape[2] - cols_kernel):
                sect = get_image_section(layer_0, row_start, row_start + rows_kernel, col_start, col_start + cols_kernel)
                sects.append(sect)

        layer_0 = np.concatenate(sects, axis=1) # Объединяем условно отдельные 9 строк c shape = (1,1,3,3) в одну с shape = (1,9,3,3)
        layer_0 = layer_0.reshape(layer_0.shape[0] * layer_0.shape[1], -1) # Делаем из них строки было shape = (1,9,3,3) стало shape = (9,9)

        layer_1 = tanh(layer_0 @ weights_0)
        layer_1_reshape = layer_1.reshape(batch_size,-1)
        regular_drop = np.random.randint(2, size=layer_1_reshape.shape)
        layer_1 = (layer_1_reshape * regular_drop * 2)

        layer_2 = softmax(layer_1 @ weights_1)

        for k in range(batch_size):
            correct_cnt += int(np.argmax(layer_2[k:k+1]) == np.argmax(labels[batch_start+k:batch_start+k+1]))

        layer_2_delta = (layer_2 - labels[batch_start:batch_end]) / (batch_size * layer_2.shape[0])

        layer_1_delta = layer_2_delta @ weights_1.T * tanhRevers(layer_1_reshape)
        layer_1_delta *= regular_drop
        print(layer_0.T.shape, layer_1_delta.reshape(layer_1.shape).shape)
        weights_1 -= alpha * layer_1.T @ layer_2_delta
        weights_0 -= alpha * layer_0.T @ layer_1_delta.reshape(layer_1.shape)

    test_correct_cnt = 0

    for i in range(len(test_images)):
        layer_0 = test_images[i:i+1]
        layer_0.reshape(layer_0.shape[0],28,28)

        sects = list()
        for row_start in range(layer_0.shape[1] - rows_kernel):
            for col_start in range(layer_0[2] - cols_kernel):
                sect = get_image_section(layer_0, row_start, row_start + rows_kernel, col_start, col_start + cols_kernel)
                sects.append(sect)

        layer_0 = np.concatenate(sects, axis=1) # Объединяем условно отдельные 9 строк c shape = (1,1,3,3) в одну с shape = (1,9,3,3)
        layer_0 = layer_0.reshape(layer_0[0] * layer_0[1], -1) # Делаем из них строки было shape = (1,9,3,3) стало shape = (9,9)

        layer_1 = tanh(layer_0 @ weights_0)
        layer_2 = layer_1 @ weights_1

        test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

    if(j % 10 == 0):
        sys.stdout.write("\n"+ "I:" + str(j) + \
                         " Test-Acc:"+str(test_correct_cnt/float(len(test_images)))+ \
                         " Train-Acc:" + str(correct_cnt/float(len(images))))