import sys, numpy as np
from keras.datasets import mnist
from MLFrameWork import *

alpha = 2
iterations = 300
hidden_size = 100
pixels_pre_image = 28*28
num_labels = 10
batch_size = 100

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

model = Sequential([Linear(pixels_pre_image,hidden_size), Tanh(), Linear(hidden_size, num_labels), Softmax()])

loss = MSELoss()

optim = SGD(tensors=model.get_parameters(), alpha=alpha)

for j in range(iterations):
    correct_cnt = 0
    total_loss = 0

    for i in range(int(len(images) / batch_size)):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size

        # Прямой проход
        input_batch = Tensor(images[batch_start:batch_end], autograd=True)
        target_batch = Tensor(labels[batch_start:batch_end], autograd=True)

        pred = model.forward(input_batch)

        l = loss.forward(pred, target_batch)
        total_loss += l.data.sum()

        l.backward(Tensor(l.data))

        optim.step()

        # Подсчёт точности
        predictions = np.argmax(pred.data, axis=1)
        true_labels = np.argmax(labels[batch_start:batch_end], axis=1)
        correct_cnt += np.sum(predictions == true_labels)

    # Тестирование
    test_correct_cnt = 0
    for i in range(len(test_images)):
        input_test = Tensor(test_images[i:i+1], autograd=False)
        pred_test = model.forward(input_test)
        predictions = np.argmax(pred_test.data, axis=1)
        true_labels = np.argmax(test_labels[i:i+1], axis=1)
        test_correct_cnt += np.sum(predictions == true_labels)

    if j % 10 == 0:
        sys.stdout.write("\n" + "I:" + str(j) +
                         " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) +
                         " Train-Acc:" + str(correct_cnt / float(len(images))))
        sys.stdout.flush()