#!/usr/bin/env python

import os
import sys
import numpy as np
import scipy.stats as sts
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NUM_GR = 2 ** 20

def get_weights_from_file(w_file):
    try:
        with open(w_file, 'rb') as w_fd:
            raw_data = np.fromfile(w_fd, np.single)
            w_fd.close()
    except FileNotFoundError:
        print(f"[ERROR] '{w_file}' could not be opened.")
        print('[ERROR]: Exiting...')
        sys.exit(1)
    return raw_data

eq_weight_path = "./eq_ISI_500_TRIAL_499.pfpcw"
acq_weight_path = "./acq_ISI_500_TRIAL_749.pfpcw"

eq_weights = get_weights_from_file(eq_weight_path)
acq_weights = get_weights_from_file(acq_weight_path)

acq_weights_and_ids = np.row_stack((np.arange(len(acq_weights)), acq_weights))

#fig = plt.figure()
#fig.suptitle("Un-Sorted Acquisition Weights", fontsize=14)
#ax = plt.subplot(111)
#ax.set_xlabel('weight id', fontsize=12)
#ax.set_ylabel('weight val', fontsize=12)
#ax.plot(acq_weights_and_ids[1], 'ko', ms=0.25)
#plt.show()
#plt.close(fig)

"""
These two lines are for if you are resetting wrt to the sorted
acquisition weight vals
"""
#sorted_ids = np.argsort(acq_weights)
#sorted_acq_ws_ids = acq_weights_and_ids[:, sorted_ids]

"""
These two lines are for if you are resetting wrt to the sorted *deltas*
between acq and eq weight vals
"""
deltas = acq_weights-eq_weights
ws_deltas_and_ids = np.row_stack((np.arange(NUM_GR), deltas))
sorted_ids = np.argsort(deltas)
sorted_ws_deltas_and_ids = ws_deltas_and_ids[:, sorted_ids]

#fig = plt.figure()
#fig.suptitle("Weights Deltas (acq-eq)", fontsize=14)
#ax = plt.subplot(111)
#ax.set_xlabel('weight id', fontsize=12)
#ax.set_ylabel('weight val', fontsize=12)
#ax.plot(ws_deltas_and_ids[1], 'ko', ms=0.25)
#plt.show()
#plt.close(fig)
#
#fig = plt.figure()
#fig.suptitle("Sorted Weights Deltas (acq-eq)", fontsize=14)
#ax = plt.subplot(111)
#ax.set_xlabel('weight id', fontsize=12)
#ax.set_ylabel('weight val', fontsize=12)
#ax.plot(sorted_ws_deltas_and_ids[1], 'ko', ms=0.25)
#plt.show()
#plt.close(fig)

#fig = plt.figure()
#fig.suptitle("Sorted Acquisition Weights", fontsize=14)
#ax = plt.subplot(111)
#ax.set_xlabel('weight id', fontsize=12)
#ax.set_ylabel('weight val', fontsize=12)
#ax.plot(sorted_acq_ws_ids[1], 'ko', ms=0.25)
#plt.show()
#plt.close(fig)

# reset time boi
# initial block is playing around with the alg
#max_reset = 50000
#reset_weights = np.copy(acq_weights)
#ids_to_reset = sorted_acq_ws_ids[0, :max_reset]
#ids_to_reset = ids_to_reset.astype(np.uintc)
#reset_weights[ids_to_reset] = eq_weights[ids_to_reset]
#
#sorted_ids = np.argsort(reset_weights)
#sorted_reset_weights = reset_weights[sorted_ids]

# checking the sorted weights to see where the lowest ones are reset to
# fig = plt.figure()
# fig.suptitle("Sorted Reset Weights", fontsize=14)
# ax = plt.subplot(111)
# ax.set_xlabel('weight id', fontsize=12)
# ax.set_ylabel('weight val', fontsize=12)
# ax.plot(sorted_reset_weights, 'ko', ms=0.25)
# plt.show()
# plt.close(fig)

# checking reset distribution
#fig = plt.figure()
#fig.suptitle("Reset Distribution", fontsize=14)
#ax = plt.subplot(111)
#ax.set_xlabel('weight val', fontsize=12)
#ax.set_ylabel('num in bin', fontsize=12)
#ax.hist(reset_weights, bins=400, edgecolor='black')
#plt.show()
#plt.close(fig)

def create_freeze_mask(ids):
    mask = np.zeros(NUM_GR, dtype=np.uint8)
    mask[ids] = 1
    return mask

# the final countdown
max_reset = 50000
step = 500
ISI = 500
reset_weight_nums = np.arange(0, max_reset + step, step)

for num in reset_weight_nums:
    print(f"reseting {num} lowest weights...")
    reset_weights = np.copy(acq_weights)
    """
    below is for if you are resetting based off sorted acq weight value
    """
    #ids_to_reset = sorted_acq_ws_ids[0, :num]
    """
    below is for if you are resetting based off sorted delta (acq-eq) weight value
    """
    ids_to_reset = sorted_ws_deltas_and_ids[0, :num]
    ids_to_reset = ids_to_reset.astype(np.uintc)
    print("creating mask array for those weights")
    freeze_mask = create_freeze_mask(ids_to_reset)
    reset_weights[ids_to_reset] = eq_weights[ids_to_reset]

    #reset_weight_file = f"reset_isi_{ISI}_{num}.pfpcw"
    reset_weight_file = f"reset_w_delta_isi_{ISI}_{num}.pfpcw"
    print(f"Saving {num} reset weights to '{reset_weight_file}'...")
    reset_weights.tofile(reset_weight_file)
    #freeze_mask_file = f"freeze_isi_{ISI}_{num}.mask"
    freeze_mask_file = f"freeze_w_delta_isi_{ISI}_{num}.mask"
    print(f"finished all tasks for reset num {num}")
    freeze_mask.tofile(freeze_mask_file)
print("all done. have nice day :-)")
