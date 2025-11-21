import numpy as np

def loadCNNcoeffs(filename):
    tensors = {}
    current_name = None
    current_lines = []
    #Open and parse
    with open(filename, "r") as f:
        for line in f:
            #Detect new tensor
            if line.startswith("tensor_name:"):
                if current_name is not None:
                    tensors[current_name] = "".join(current_lines)
                current_name = line.split(":", 1)[1].strip()
                current_lines = []
            #Otherwise add data to the current tensor
            else:
                current_lines.append(line)
        if current_name is not None:
            tensors[current_name] = "".join(current_lines)

    #Flatten vector
    def parse_flat(s):
        clean = s.replace("[", " ").replace("]", " ")
        return np.fromstring(clean, sep=" ")

    #1st layer
    b1 = parse_flat(tensors["conv1/biases"])
    w1 = parse_flat(tensors["conv1/weights"]).reshape(3,3,3,64)

    #2nd layer
    b2 = parse_flat(tensors["conv2/biases"])
    w2 = parse_flat(tensors["conv2/weights"]).reshape(3,3,64,32)

    #3rd layer
    b3 = parse_flat(tensors["conv3/biases"])
    w3 = parse_flat(tensors["conv3/weights"]).reshape(3,3,32,20)

    #Fully connected layer weights and biases
    bfc = parse_flat(tensors["local3/biases"])
    wfc = parse_flat(tensors["local3/weights"]).reshape(180,10)

    return w1, b1, w2, b2, w3, b3, wfc, bfc