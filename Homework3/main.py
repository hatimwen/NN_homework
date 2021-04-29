import numpy as np
from skimage.io import imread
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from tqdm import tqdm

Nc = 512
L = 9
epoch_max = 1000
lr = 0.1

def getvector(img):
    # pad
    # img = np.pad(img, pad_width=[[1, 1], [1, 1]], mode='constant', constant_values=0)
    img = np.pad(img, pad_width=[[1, 1], [1, 1]], mode='symmetric')
    # image vector
    X = np.zeros([86**2, L], np.uint8)
    for i in range(86**2):
        y = i // 86
        x = i % 86
        X[i] = img[y*3:y*3+3, x*3:x*3+3].reshape(-1)
    return X

def decode(X):
    X_out = np.zeros([258,258], dtype=np.uint8)
    for i in range(86**2):
        y = i // 86
        x = i % 86
        X_out[y*3:y*3+3, x*3:x*3+3] = X[i].reshape(3, 3)
    return X_out[1:257, 1:257]

# Train:
def train(img):
    X_train = getvector(img)
    # Code Book
    np.random.seed()
    W = np.random.randint(0, 256, size=(Nc, L)).astype(np.float64)
    Last_W = W * 0
    for epoch in tqdm(range(epoch_max)):
        for k in range(X_train.shape[0]):
            D = cdist(np.array([X_train[k]]), W, metric='euclidean')
            q = np.argmin(D)
            # update Wq:
            W[q] += lr * (X_train[k] - W[q])
            dis = np.sum(np.sqrt((W - Last_W)**2))
            Last_W = W.copy()
            if k%1000 == 0:
                print('epoch{0} idx = {1} dis = {2}'.format(epoch, k, dis))
        if dis == 0.:
            break
    return W

# Test
def test(img_in, W):
    X_test = getvector(img_in)
    X_test_out = X_test.copy()
    D = cdist(X_test, W, metric='euclidean')
    for i in range(D.shape[0]):
        q = np.argmin(D[i])
        X_test_out[i] = W[q]
    img_out = decode(X_test_out)
    mse = np.sum(img_in - img_out) / (256.**2)
    psnr = 10 * np.log10(255**2 / mse)
    return img_out, psnr


def vis(img_name, img_in, img_out, psnr):
    print('Image:{0} PSNR: {1:.4f}dB'.format(img_name, psnr))
    plt.subplot(1,2,1)
    plt.imshow(img_in)
    plt.title('Origin image')

    plt.subplot(1,2,2)
    plt.imshow(img_out)
    plt.title('PSNR: {:.4f}dB'.format(psnr))
    plt.show()

if __name__ == '__main__':
    image = 'image/LENA.BMP'
    img = imread(image)
    W = train(img)

    img_name = 'image/LENA.BMP'
    img_in = imread(img_name)
    img_out, psnr = test(img_in, W)
    vis(img_name, img_in, img_out, psnr)

    img_name = 'image/CR.BMP'
    img_in = imread(img_name)
    img_out, psnr = test(img_in, W)
    vis(img_name, img_in, img_out, psnr)

    img_name = 'image/HS4.BMP'
    img_in = imread(img_name)
    img_out, psnr = test(img_in, W)
    vis(img_name, img_in, img_out, psnr)