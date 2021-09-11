"""
Utilities functions
"""
import numpy as np
from models import HWconv2d, Conv2d, BatchNorm, Linear

def save_params(model):
    conv_idx = 0
    bn_idx = 0
    for n in range(len(model.features)):
        f = model.features[n]
        if isinstance(f, HWconv2d) or isinstance(f, Conv2d):
            x, w, y = f.input.cpu().numpy(), f.weight_old.cpu().numpy(), f.out.cpu().numpy()
            wu = f.weight.cpu().numpy()
            grad = f.w_grad.cpu().numpy()

            # save param
            np.save(f"./params/FF/conv_x{conv_idx}.npy", x)
            np.save(f"./params/FF/conv_w{conv_idx}.npy", w)
            np.save(f"./params/FF/conv_y{conv_idx}.npy", y)

            np.save(f"./params/BP/conv_wgrad{conv_idx}.npy", grad)
            np.save(f"./params/WU/conv_w{conv_idx}_updated.npy", wu)

            conv_idx += 1
        
        if isinstance(f, BatchNorm):
            x, w, b, y = f.input.cpu().numpy(), f.weight_old.cpu().numpy(), f.bias_old.cpu().numpy(), f.output.cpu().numpy()
            mean, std = f.mean.cpu().numpy(), f.std.cpu().numpy()
            wgrad, bgrad = f.w_grad.cpu().numpy(), f.b_grad.cpu().numpy()
            wvel, bvel = f.w_vel.cpu().numpy(), f.b_vel.cpu().numpy()
            wu, bu = f.weight.cpu().numpy(), f.bias.cpu().numpy()

            # save param
            np.save(f"./params/FF/bn_x{bn_idx}.npy", x)
            np.save(f"./params/FF/bn_w{bn_idx}.npy", w)
            np.save(f"./params/FF/bn_b{bn_idx}.npy", b)
            np.save(f"./params/FF/bn_y{bn_idx}.npy", y)
            np.save(f"./params/FF/bn_mu{bn_idx}.npy", mean)
            np.save(f"./params/FF/bn_std{bn_idx}.npy", std)

            np.save(f"./params/BP/bn_wgrad{bn_idx}.npy", wgrad)
            np.save(f"./params/BP/bn_bgrad{bn_idx}.npy", bgrad)
            np.save(f"./params/BP/bn_wvel{bn_idx}.npy", wvel)
            np.save(f"./params/BP/bn_bvel{bn_idx}.npy", bvel)

            np.save(f"./params/WU/bn_w{bn_idx}_updated.npy", wu)
            np.save(f"./params/WU/bn_b{bn_idx}_updated.npy", bu)
            
            bn_idx += 1
    
    # fully connected layer
    x, w, b, y = model.fc.input.cpu().numpy(), model.fc.weight_old.cpu().numpy(), model.fc.bias_old.cpu().numpy(), model.fc.out.cpu().numpy()
    wgrad, bgrad = model.fc.w_vel.cpu().numpy(), model.fc.b_vel.cpu().numpy()
    wu, bu = model.fc.weight.cpu().numpy(), model.fc.bias.cpu().numpy()
    
    # save param
    np.save(f"./params/FF/fc_x{bn_idx}.npy", x)
    np.save(f"./params/FF/fc_w{bn_idx}.npy", w)
    np.save(f"./params/FF/fc_b{bn_idx}.npy", b)
    np.save(f"./params/FF/fc_y{bn_idx}.npy", y)

    np.save(f"./params/BP/fc_wgrad{bn_idx}.npy", wgrad)
    np.save(f"./params/BP/fc_bgrad{bn_idx}.npy", bgrad)

    np.save(f"./params/WU/fc_w{bn_idx}_updated.npy", wu)
    np.save(f"./params/WU/fc_b{bn_idx}_updated.npy", bu)
    