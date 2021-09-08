"""
Basic function
"""

import torch

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def max_idx(window):
    shape = list(window.size())
    idx = torch.argmax(window).item()
    ind = unravel_index(idx, shape)
    return ind


def maxpool_2d(image, f=2, s=2):
    """
    Downsample 'image' using kernel size 'f' and stride 's'
    """
    N, C, Hin, Win = image.size()
    
    Hout = int((Hin - f)/s + 1)
    Wout = int((Win - f)/s + 1)
    
    out = torch.zeros(N, C, Hout, Wout)
    for b in range(N):
        for c in range(C):
            curr_y = out_y = 0
            while curr_y + f <= Hin:
                curr_x = out_x = 0
                while curr_x + f <= Win:
                    out[b, c, out_y, out_x] = torch.max(image[b, c, curr_y:curr_y+f, curr_x:curr_x+f])
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
    return out

def maxpoolBackward(dpool, orig, f, s):
    """
    Backpropagation through the maxpooling layer
    """
    N, C, Hin, Win = orig.size()
    
    dout = torch.zeros_like(orig)
    for b in range(N):
        for c in range(C):
            curr_y = out_y = 0
            while curr_y + f <= Hin:
                curr_x = out_x = 0
                while curr_x + f <= Win:
                    window = orig[b, c, curr_y:curr_y+f, curr_x:curr_x+f].unsqueeze(0).unsqueeze(0)
                    wn, wc, wh, ww = max_idx(window)
                    dout[b, c, curr_y+wh, curr_x+ww] = dpool[b, c, out_y, out_x]
                    
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
    return dout