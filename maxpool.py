import numpy as np

#Receives a 2D x1*y1*b1 matrix - numpy array
#and returns a 2D x2*y2*b1 matrix - numpy array

def maxpool(inputMat, Ksize=3, stride=2, verbose = False):
    #We can predict output size from input kernel size and stride:
    inH, inW = inputMat.shape
    outH = inH//2
    outW = inW//2
    
    outputMat = np.zeros((outH, outW))

    #for each pixel of the output
    Kh = Ksize//2
    if(verbose):
        print(f"Checking range is : [{-Kh}, {Kh}]")
    for i in range(0, outH):
        for j in range(0, outW):
            #First, form the kernel in the input matrix
            #Find the coord of the central pixel in input mat.
            centralI = i*stride + Kh
            centralJ = j*stride + Kh
            if(verbose):
                print(f"Processing outputMat[{i}][{j}] with central pixel inputMat[{centralI}][{centralJ}]")
            #Find max in the kernel by pooling around central pixel
            for ki in range(-Kh, Kh+1):
                for kj in range(-Kh, Kh+1):
                    if(centralI + ki > inH-1 or centralJ + kj > inW-1 or centralI + ki < 0 or centralJ + kj < 0):
                        if(verbose):
                            print("Out of bounds, skipping")
                        continue
                    else:
                        if(verbose):
                            print(f"Comparing inputMat[{centralI + ki}][{centralJ + kj}] = {inputMat[centralI + ki][centralJ + kj]} with outputMat[{i}][{j}] = {outputMat[i][j]}")
                        if(inputMat[centralI + ki][centralJ + kj] >= outputMat[i][j]):
                            outputMat[i][j] = inputMat[centralI + ki][centralJ + kj]

    return outputMat


def test_maxpool():
    inputMat = np.array([
        [1, 3, 2, 1],
        [4, 6, 5, 2],
        [7, 9, 8, 3],
        [0, 1, 2, 4]
    ])
    expected = np.array([
        [9, 8],
        [9, 8]
    ])

    output = maxpool(inputMat, 3, 2, True)

    if np.array_equal(output, expected):
        print("TEST PASS")
    else:
        print("TEST FAIL")
        print("Expected:")
        print(expected)
        print("Got:")
        print(output)

#test_maxpool()