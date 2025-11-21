from loadCoeff import loadCNNcoeffs
from read_cifar import *
from convolution_relu import conv2d_single, relu
from maxpool import maxpool

def labelCorresp(idx):
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    if idx < 0 or idx >= len(labels):
        return "Unknown"
    return labels[idx]

def CNNmain(batch, index, verbose=False, disp=False):
    #Load CNN coefficients
    w1, b1, w2, b2, w3, b3, wfc, bfc = loadCNNcoeffs("../CIFAR10/CNN_coeff_3x3.txt")

    #Load images
    img, label = load_cifar10_image(batch, index)

    #Preprocess image
    img = center_crop_24x24(img)
    img = normalize_image(img)

    #1st layer : Conv. + ReLU + MaxPool
    conv1 = conv2d_single(img, w1, b1)
    relu1 = relu(conv1)
    pool1 = maxpool(relu1, Ksize=3, stride=2)

    #2nd layer : Conv. + ReLU + MaxPool
    conv2 = conv2d_single(pool1, w2, b2)
    relu2 = relu(conv2)
    pool2 = maxpool(relu2, Ksize=3, stride=2)

    #3rd layer : Conv. + ReLU + MaxPool
    conv3 = conv2d_single(pool2, w3, b3)
    relu3 = relu(conv3)
    pool3 = maxpool(relu3, Ksize=3, stride=2)

    #Reshape to vector
    flat = flatten(pool3)

    #Fully connected layer
    fc_out = np.dot(flat, wfc) + bfc
    predicted_label = np.argmax(fc_out)

    if verbose:
        print("Output scores:", fc_out)
        print("Predicted label:", predicted_label, "and associated class:", labelCorresp(predicted_label))
        print("True label:", label, "and associated class:", labelCorresp(label))

    if disp:
        display_img(img, labelCorresp(label))

    if(predicted_label == label):
        return True
    else:
        return False


print("Running CNN on CIFAR-10 image batch 1 :")

predictionCorrectCount = 0
batchNb = 200

for i in range(200, batchNb+200):
    if CNNmain("../CIFAR10/cifar10_data/cifar-10-batches-bin/test_batch.bin", i, False, False):
        predictionCorrectCount += 1
    print(f"Live reliability percentage : {(predictionCorrectCount/(i-199))*100}%")

print(f"Final reliability count: {(predictionCorrectCount/batchNb)*100}%")