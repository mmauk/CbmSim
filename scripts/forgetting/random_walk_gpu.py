#!/usr/bin/env python3
import time
import torch
import sys
import os
from scipy.stats import halfnorm
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

def load_weights_mask(file_path_weights, file_path_mask):
    return np.fromfile(file_path_weights, dtype=np.single), np.fromfile(file_path_mask, dtype=np.ubyte)

def select_weights(weights, mask):
    weights_to_walk = weights[np.where(mask)]
    print("NUM OF WEIGHTS TO WALK:", len(weights_to_walk))
    return weights_to_walk

def set_weights(walked_weights, weights, mask):
    new_weights = np.asarray([walked_weights.tolist().pop(0) if mask[x] == 1 else y for x, y in enumerate(weights)])
    print(new_weights.shape)
    return new_weights
     
def pytorch_random_walk(
            weight_subset: np.ndarray, num_t_steps: int, d_plus_d_minus_ratio: float,
            d_plus: float, d_minus: float) -> np.ndarray:
    cuda = torch.device('cuda')
    x_1_d = torch.tensor(1 - d_plus).to(device=cuda)
    x_2_d = torch.tensor(d_minus).to(device=cuda)
    d_plus_d = torch.tensor(d_plus).to(device=cuda)
    d_minus_d = torch.tensor(d_minus).to(device=cuda)
    thresh_d = torch.tensor(d_plus_d_minus_ratio).to(device=cuda)
    num_w_subset = weight_subset.size
    weight_subset_d = torch.from_numpy(weight_subset).to(device=cuda)
    for t in np.arange(num_t_steps):
        deltas_d = torch.rand(num_w_subset, device=cuda)
        if t % 1000 == 0: print(f"time step {t}...")
        ltp_elig = torch.where(deltas_d < thresh_d, torch.ones(num_w_subset, device=cuda))[0]
        pluses = weight_subset_d + d_plus_d
        minuses = weight_subset_d + d_minus_d
        weight_subset_d = ltp_elig * torch.where(x_1_d > weight_subset_d, pluses)[0] \
                + (1 - ltp_elig) * torch.where(x_2_d > weight_subset_d, minuses)[0]
        print(weight_subset_d)
    weight_subset_h = weight_subset_d.to('cpu')
    return weight_subset_h

"""
pytorch attempt - did not work :'-()
"""

#ratio = 9.
#d_plus = 0.001
#d_minus = -1. * d_plus * ratio
#
#start = time.time()
#diffused_weights_test = pytorch_random_walk(weights_test, num_t_steps_range_test,
#                                            ratio, d_plus, d_minus)
#end = time.time()
def save_weights_at_timestep(current_timestep, weights_array):
    cwd = os.getcwd()
    filepath = cwd + '/CbmSim/data/outputs/ISI_' + str(ISI) + '/walked_weights/'
    if not os.path.exists(filepath): os.makedirs(filepath)
    os.chdir(filepath)
    filename = 'walked_weights_' + str(ISI) + '_' + str(current_timestep) + '.pfpcw'
    weights_array.tofile(filename)
    os.chdir(cwd)

        
def cupy_random_walk(weight_subset_h: np.ndarray, mask_h: np.ndarray, num_trials: int, t_step_per_trial: int) -> np.ndarray:
    walk_kernel = cp.ElementwiseKernel(
            'int32 deltas_d, float32 weights_d, uint8 mask',
            'float32 result',
            '''
                float ratio = 9.0;
                float d_plus = 0.001;
                float d_minus = -1. * d_plus * ratio;
                float x1 = 1 - d_plus;
                float x2 = -d_minus;
                float thresh = ratio;
                float temp = weights_d;
                temp += (deltas_d < thresh) * d_plus + (deltas_d >= thresh) * d_minus;
                temp = temp * (weights_d < x1) + (temp >= x1);
                temp = temp * (temp > x2) + x2 * (temp <= x2);
                result = mask * temp + (1 - mask) * weights_d; // opposite of cbm_sim mask >:(
            ''',
            'walk_kernel'
            )
    max_delta = 10 # ratio + 1
    num_w_subset = weight_subset_h.size
    with cp.cuda.Device(0):
        weight_subset_d = cp.asarray(weight_subset_h)
        mask_d = cp.asarray(mask_h)
        start = time.time()
        for trial in np.arange(num_trials):
            print(f"running trial {trial+1}...")
            for t in np.arange(t_step_per_trial):
                deltas_d = cp.random.randint(0, high=max_delta, size=num_w_subset, dtype=cp.int32)
                weight_subset_d = walk_kernel(deltas_d, weight_subset_d, mask_d)
            if trial % save_step == 0:
                save_weights_at_timestep(trial, cp.asnumpy(weight_subset_d))
            end = time.time()
            print(f"trial {trial+1} took: {end-start:0.2f}s.")
            start = end
        weight_subset_h = cp.asnumpy(weight_subset_d)
    return weight_subset_h


"""
cupy attempt - more success
"""

if len(sys.argv) != 3: 
    print('Please Input only 2 Arguments')
    print(sys.argv)
    exit(0)

ISI = sys.argv[1]
NUM_RESET = sys.argv[2]

# isi 500, 30000 weights seem to be what is necessary to prevent forgetting if they are frozen...
num_weights = 2 ** 20
num_trials = 10
num_t_step_per_trial = 5000
save_step = 1
# simulating the most significant weights look kinda half-gaussy around 0:
# here the scale of course is not accurate to what the big sim produces
weights_path = 'CbmSim/data/outputs/reset_isi_' + ISI + '/reset_w_delta_isi_' + ISI + '_' + NUM_RESET + '.pfpcw'
mask_path = 'CbmSim/data/outputs/freeze_isi_' + ISI + '/freeze_w_delta_isi_' + ISI + '_' + NUM_RESET + '.mask'


weights, mask = load_weights_mask(weights_path, mask_path)
#weights = halfnorm.rvs(loc = 0, scale = 0.05, size=num_weights).astype(np.float32)
#weights_to_walk = select_weights(weights, mask)
#random_weight_values = np.random.random(size=len(weights_to_walk))
#WALK

# # we are forgor
walked_weights = cupy_random_walk(weights, mask, num_trials, num_t_step_per_trial)
# new_weights = set_weights(random_weight_values, weights, mask)
#new_weights = set_weights(walked_weights, weights, mask)
new_weights = walked_weights
print(np.array_equal(weights, new_weights))

# plot them shits
bin_size = 0.0025
binz = np.arange(0, 1, bin_size)
plt.xlim(0, 1)
plt.hist(weights, bins=binz, alpha=0.5, label='before forget')
plt.hist(new_weights, bins=binz, alpha=0.5, label='after forget')
plt.legend(loc='upper right')
plt.show() 


"""
TODO:
    - read in weights from file
    - modify script so that we're only modifying most significant 30K or so weights 
      (might read in mask for this purpose)
    - save weights to file every M time steps (or you can think of M in intervals of trials,
      like save every 500 trials) For now, does not have to be too-fine grained, as we need
      to compare this forgetting to the big sim forgetting

    msg me with Qs :-)
"""
