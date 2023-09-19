import numpy as np
import os
import copy 

def import_order_weights(acq_file_path, eq_file_path):
    isi = acq_file_path.split('/')[2].split('_')[2]

    acq_pfpc_weights = np.fromfile(acq_file_path, dtype=np.single)
    eq_pfpc_weights = np.fromfile(eq_file_path, dtype=np.single)

    index_weights = np.arange(0, acq_pfpc_weights.shape[0])
    weights_to_order = np.column_stack((acq_pfpc_weights, eq_pfpc_weights, index_weights))
    ordered_weights = weights_to_order[np.argsort(weights_to_order[:, 0])]
    return ordered_weights, isi

def TEST_import_order_weights():
    acq_pfpc_weights = np.random.randint(low=1, high=100, size=1000000)
    eq_pfpc_weights = np.random.randint(low=100, high=200, size=1000000)

    index_weights = np.arange(0, acq_pfpc_weights.shape[0])
    weights_to_order = np.column_stack((acq_pfpc_weights, eq_pfpc_weights, index_weights))
    ordered_weights = weights_to_order[np.argsort(weights_to_order[:, 0])]
    return ordered_weights

def set_reset_weights(ordered_weights, isi):

    arr_indicies = np.argwhere(ordered_weights[:, 0] < .45)
    weights_to_reset = ordered_weights[arr_indicies[0][0] : arr_indicies[-1][0]]
    
    point = arr_indicies[-1][0]
    start = arr_indicies[0][0]
    step = (arr_indicies[-1][0] - arr_indicies[0][0]) // 10
    
    cwd = os.getcwd()
    try:
        os.mkdir(cwd + '/ISI_' + isi + '/')
    except FileExistsError:
        pass

    finally:
        os.chdir(cwd + '/ISI_' + isi)
        cwd_isi = os.getcwd()

    try:
        os.mkdir(cwd_isi + '/pfpc_weights/')
        os.mkdir(cwd_isi + '/reset_mask/')
    except FileExistsError:
        pass

    finally:
        compare_sub_1 = np.ndarray([])
        compare_sub_2 = np.ndarray([])
        while start + step <= point:
            reset_weights = np.column_stack((np.concatenate((ordered_weights[start:point, 0], ordered_weights[point:, 1])), ordered_weights[:, 2]))
            
            print("START:", start, "STOP:", point)
            index_reset = np.sort(ordered_weights[start:point, 2])
            index_ordered_reset = reset_weights[np.argsort(reset_weights[:, 1])][:, 0].astype(np.single)
            print("AMT CHANGED BY INDEX", (point-start)/index_ordered_reset.shape[0])
            print("ELEMENT-WISE COMPARISON SUB 1 ARR", (np.sum(np.equal(compare_sub_1, index_ordered_reset))/index_ordered_reset.shape[0]))
            print("ELEMENT-WISE COMPARISON SUB 2 ARR", (np.sum(np.equal(compare_sub_2, index_ordered_reset))/index_ordered_reset.shape[0]))
            compare_sub_2 = copy.deepcopy(compare_sub_1)
            compare_sub_1 = copy.deepcopy(index_ordered_reset)
            
            # print(index_ordered_reset[0], index_ordered_reset[-1])
            print(os.getcwd())
            os.chdir(cwd_isi + '/pfpc_weights/') 
            weights_filepath = 'pfpcw_' + str(isi) + '_' + str(point) + '.pfpcw'
            index_ordered_reset.tofile(weights_filepath)        
            os.chdir(cwd_isi)
            
            # print(reset_weights)
            # print(index_reset)
            mask = create_mask(index_ordered_reset.shape[0], index_reset)
            os.chdir(cwd_isi + '/reset_mask/')
            mask_filename = 'reset_mask_' + str(isi) + '_' + str(point) + '.mask'
            mask.tofile(mask_filename)
            os.chdir(cwd_isi)
            point = point - step
                
        os.chdir(cwd)
        return "DONE"



def TEST_set_reset_weights(ordered_weights):
    # print(ordered_weights.shape)

    print(ordered_weights[:, 0])
    arr_indicies = np.argwhere(ordered_weights[:, 0] < .45)
    print(arr_indicies[0], arr_indicies[-1])
    weights_to_reset = ordered_weights[arr_indicies[0][0] : arr_indicies[-1][0]]
    print(weights_to_reset)
    
    point = arr_indicies[-1][0]
    start = arr_indicies[0][0]
    step = (arr_indicies[-1][0] - arr_indicies[0][0]) // 10
    

    # cwd = os.getcwd()
    # os.mkdir(cwd + '/ISI_' + isi + '/')
    # os.chdir(cwd + '/ISI_' + isi)
    # cwd_isi = os.getcwd()
    # os.mkdir(cwd_isi + '/pfpc_weights/')
    # os.mkdir(cwd_isi + '/reset_mask/')

    while start + step <= point:
        reset_weights = np.column_stack((np.concatenate((ordered_weights[start:point, 0], ordered_weights[point:, 1])), ordered_weights[:, 2]))
        index_reset = np.sort(ordered_weights[start:point, 2])
        index_ordered_reset = reset_weights[np.argsort(reset_weights[:, 1])]
        print(index_ordered_reset.shape)

        # weights_filepath = '/pfpc_weights/pfpcw_' + str(isi) + '_' + str(point) + '.pfpcw'
        # weights.tofile(weights_filepath)        
        
        print("START:", start, "STOP:", point)
        # print(reset_weights)
        # print(index_reset)
        mask = create_mask(index_ordered_reset.shape[0], index_reset)

        # mask_filename = '/reset_mask/reset_mask_' + str(isi) + '_' + str(point) + '.mask'
        # mask.tofile(mask_filename)
        point = point - step
            
    # os.chdir(cwd)
    # return "DONE"

def create_mask(len_weights, index_reset):    
    mask = np.zeros(len_weights, dtype=np.ubyte)
    mask[index_reset.astype(int)] = 1

    return mask

print(os.getcwd())

# weights returned is a col stack of these: (acq_pfpc_weights, eq_pfpc_weights, index_weights)
# so weights should have shape (num_gr, 3)

weights, isi = import_order_weights('acq_data/acq_ISI_500/acq_ISI_500_TRIAL_749.pfpcw', 'eq_data/eq_ISI_500/eq_ISI_500_TRIAL_499.pfpcw')
set_reset_weights(weights, isi)

# weights = TEST_import_order_weights()

# TEST_set_reset_weights(weights)


