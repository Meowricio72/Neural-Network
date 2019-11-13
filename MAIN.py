import numpy as np
111

PicNum = 10000
PixNum = 3072
classes = 9

labels = {}
data = {}
w = {}

for num in range(1, 6):
    data[num] = np.load(f'D:\\Progs\\PythonCharm\\Projects\\CIFAR-10\\Datasets\\Unpacked\\Data\\batch{num}.npy')
    labels[num] = np.load(f'D:\\Progs\\PythonCharm\\Projects\\CIFAR-10\\Datasets\\Unpacked\\Labels\\labels{num}.npy')

batch1 = data[1]
labels1 = labels[1]


def tanh(inp):
    return np.tanh(inp)


def act(inp):  # функция активации через ReLu
    if inp < 0:
        return 0
    else:
        return inp

# def reshape (inp) :
#     for line in range(1000) :
#         for column in range (3024):
#             for x in range(31):
#                 for y in range(31):
#                     out[x,y] = inp[line]


def conv(inp):  # слой свертки.
    global w
    w_dict_len = (dict.__len__(w))
    w[w_dict_len+1] = np.random.random((PixNum, PicNum))
    out = np.zeros((32, 32))
    x = 0
    pic = 0
    while pic < 5:
        while x < 32:
            y = 0
            npix = 0
            while y < 32:
                if npix <= 1024:
                    out[x, y] = w[pic, npix] * tanh((inp[pic, npix]        + inp[pic, npix + 1]      +  # Красный канал
                                                     inp[pic, npix + 33]   + inp[pic, npix + 34]     +
                                                     inp[pic, npix + 1024] + inp[pic, npix + 1025]   +  # Зеленый канал
                                                     inp[pic, npix + 1057] + inp[pic, npix + 1058]   +
                                                     inp[pic, npix + 2048] + inp[pic, npix + 2049]   +  # Синий канал
                                                     inp[pic, npix + 2081] + inp[pic, npix + 2082]))
                    npix += 1
                else:
                    break
                y += 1
            x += 1
        pic += 1
    return out


def conv(inp):  # слой свертки.
    global w
    layer += 1
    out = np.zeros((32, 32))
    x = 0
    pic = 0
    while pic < 5:
        while x < 32:
            y = 0
            npix = 0
            while y < 32:
                if npix <= 1024:
                    out[x, y] = w[pic, npix] * tanh((inp[pic, npix]        + inp[pic, npix + 1]      +  # Красный канал
                                                     inp[pic, npix + 33]   + inp[pic, npix + 34]     +
                                                     inp[pic, npix + 1024] + inp[pic, npix + 1025]   +  # Зеленый канал
                                                     inp[pic, npix + 1057] + inp[pic, npix + 1058]   +
                                                     inp[pic, npix + 2048] + inp[pic, npix + 2049]   +  # Синий канал
                                                     inp[pic, npix + 2081] + inp[pic, npix + 2082]))
                    npix += 1
                else:
                    break
                y += 1
            x += 1
        pic += 1
    return out





def grad(inp):  # функция градиентного спуска
    return inp*(1-inp)


def fullconn(inp):  # "полностью соединенный" слой
    global w
    shape = np.shape(inp)
    lines = shape[0]
    columns = shape[1]
    w_dict_len = (dict.__len__(w))
    w[w_dict_len + 1] = np.random.random((lines, columns))
    return tanh(np.dot(inp, (w[w_dict_len + 1]).T))  # активация умноженной матрицы входов на веса


def softmax(inp):  # функция софтмакс для классификации
    global w
    shape = np.shape(inp)
    lines = shape[0]
    columns = shape[1]
    pred = np.zeros(classes)
    w_dict_len = (dict.__len__(w))
    w[w_dict_len + 1] = np.random.random((lines, columns))
    inp = inp * w
    for i in range(classes):
        for a in range(lines):
            for b in range(columns):
                pred += np.exp(inp) / np.sum(np.exp(inp))
    error = pred - labels
    error_delta = error * grad(inp)
    w += error_delta * inp
    return

# for iter in range(10000):
#     w1 +=
#     convoluted = conv(pixels) * w1
#     for n in range(10):
#         fc = fullconn(convoluted)


print(conv(batch1))
