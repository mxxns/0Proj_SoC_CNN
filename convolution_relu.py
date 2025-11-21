import numpy as np

def conv2d_single(x, w, b, stride=1, padding="SAME"):
    """
    x: (H, W, C_in)             -- image (24x24x3)
    w: (kH, kW, C_in, C_out)    -- weights
    b: (C_out)                  -- bias
    """

    H, W, C_in = x.shape
    kH, kW, C_in_w, C_out = w.shape
    assert C_in == C_in_w

    # Padding : combien de pixels vides on rajoute autour du bord de l'image en entrée
    if padding == "SAME":
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
    elif padding == "VALID":
        pad_h = pad_w = 0
    else:
        raise ValueError("padding must be 'SAME' or 'VALID'")

    x_padded = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")

    H_out = (H + 2 * pad_h - kH) // stride + 1
    W_out = (W + 2 * pad_w - kW) // stride + 1

    out = np.zeros((H_out, W_out, C_out), dtype=np.float32)

    for y in range(H_out):
        for x_ in range(W_out):
            # Local patch: (kH, kW, C_in)
            patch = x_padded[y*stride : y*stride + kH, x_*stride: x_*stride + kW, :]
            # Convolve patch with all filters at once → (C_out,)
            out[y, x_, :] = np.tensordot(patch, w, axes=([0, 1, 2], [0, 1, 2])) + b

    return out

def relu(in):
    H_in, W_in, C_in = in.shape
    out = np.zeros((H_in, W_in, C_in), dtype=np.float32)
    for y in range(H_in)
        for x in range(W_in)
            for c in range(C_in)
                out[y, x, c] = np.max(0.0, in[y, x, c])

    return out
