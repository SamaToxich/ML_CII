import sys, numpy as np
from keras.datasets import mnist

alpha = 0.0005
iterations = 200
hidden_size = 100
pixels_pre_image = 28*28
num_labels = 10
batch_size = 1

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

#relu = lambda x: (x > 0) * x
#relu2 = lambda x: (x >= 0)
def tanh(x):
    return np.tanh(x)

def tanhRevers(x):
    return 1 - (x ** 2)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


weights_0 = 0.02 * np.random.random((pixels_pre_image, hidden_size)) - 0.01
weights_1 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    correct_cnt = 0

    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = i * batch_size, (i+1) * batch_size

        #layer_0 = images[i:i+1]
        layer_0 = images[batch_start:batch_end]

        layer_1 = tanh(layer_0 @ weights_0)
        regular_drop = np.random.randint(2, size=layer_1.shape)
        layer_1 *= regular_drop * 2

        layer_2 = softmax(layer_1 @ weights_1)

        #error += np.sum((layer_2 - labels[i:i+1]) ** 2)
        #correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1])) ВСЕ ЧТО ЗАКОМЕНЧЕНО ДЛЯ НЕ ПАКЕТНОГО СПУСКА
        for k in range(batch_size):
            correct_cnt += int(np.argmax(layer_2[k:k+1]) == np.argmax(labels[batch_start+k:batch_start+k+1]))

        #layer_2_delta = (layer_2 - labels[i:i+1])
        layer_2_delta = (layer_2 - labels[batch_start:batch_end]) / (batch_size * layer_2.shape[0])
        layer_1_delta = layer_2_delta @ weights_1.T * tanhRevers(layer_1)
        layer_1_delta *= regular_drop

        weights_1 -= alpha * layer_1.T @ layer_2_delta
        weights_0 -= alpha * layer_0.T @ layer_1_delta

    test_correct_cnt = 0

    for i in range(len(test_images)):
        layer_0 = test_images[i:i+1]
        layer_l = tanh(np.dot(layer_0,weights_0))
        layer_2 = np.dot(layer_l,weights_1)
        test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

    if(j % 10 == 0):
        sys.stdout.write("\n"+ "I:" + str(j) + \
                         " Test-Acc:"+str(test_correct_cnt/float(len(test_images)))+ \
                         " Train-Acc:" + str(correct_cnt/float(len(images))))