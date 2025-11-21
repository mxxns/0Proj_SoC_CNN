import numpy as np

def load_cifar10_image(batch_path, index):
    record_size = 1 + 3072
    with open(batch_path, "rb") as f:
        f.seek(index * record_size)
        label = f.read(1)[0]
        pixels = np.frombuffer(f.read(3072), dtype=np.uint8)

    r = pixels[0:1024].reshape(32, 32)
    g = pixels[1024:2048].reshape(32, 32)
    b = pixels[2048:3072].reshape(32, 32)

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img, label

def center_crop_24x24(img32):
    return img32[4:28, 4:28, :]

def normalize_image(img):
    mu = np.mean(img)
    sigma = np.std(img)
    N = img.size
    sigma = max(sigma, 1.0 / np.sqrt(N))
    return (img - mu) / sigma

# Makes any Matix linear
def flatten(x):
    return x.reshape(-1)

def display_img(img, title):
    plt.figure()
    plt.imshow(img)
    plt.title(f"Ceci est un {title} ?")
    plt.axis("off")
    plt.show()
