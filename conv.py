import numpy as np

def load_cnn_coeffs_txt(filename):
    tensors = {}
    current_name = None
    current_lines = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("tensor_name:"):
                if current_name is not None:
                    tensors[current_name] = "".join(current_lines)
                current_name = line.split(":", 1)[1].strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_name is not None:
            tensors[current_name] = "".join(current_lines)

    def parse_flat(s):
        clean = s.replace("[", " ").replace("]", " ")
        return np.fromstring(clean, sep=" ")

    b1 = parse_flat(tensors["conv1/biases"])
    w1 = parse_flat(tensors["conv1/weights"]).reshape(3,3,3,64)

    b2 = parse_flat(tensors["conv2/biases"])
    w2 = parse_flat(tensors["conv2/weights"]).reshape(3,3,64,32)

    b3 = parse_flat(tensors["conv3/biases"])
    w3 = parse_flat(tensors["conv3/weights"]).reshape(3,3,32,20)

    bfc = parse_flat(tensors["local3/biases"])
    Wfc = parse_flat(tensors["local3/weights"]).reshape(180,10)

    print("hello")
    print(w1)

    return w1, b1, w2, b2, w3, b3, Wfc, bfc

load_cnn_coeffs_txt("CIFAR10/CNN_coeff_3x3.txt")
