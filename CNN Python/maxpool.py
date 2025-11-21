import numpy as np

#Receives a 2D x1*y1*b1 matrix - numpy array
#and returns a 2D x2*y2*b1 matrix - numpy array

def maxpool(inputMat, Ksize=3, stride=2, verbose = False):
    #We can predict output size from input kernel size and stride:
    inH, inW, depth = inputMat.shape
    outH = inH//2
    outW = inW//2
    
    outputMat = np.zeros((outH, outW, depth))

    #for each pixel of the output
    Kh = Ksize//2
    if(verbose):
        print(f"Checking range is : [{-Kh}, {Kh}]")
    for d in range(0, depth):
        if(verbose):
            print(f"Processing depth slice {d}")
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
                            if(inputMat[centralI + ki][centralJ + kj][d] >= outputMat[i][j][d]):
                                outputMat[i][j][d] = inputMat[centralI + ki][centralJ + kj][d]

    return outputMat