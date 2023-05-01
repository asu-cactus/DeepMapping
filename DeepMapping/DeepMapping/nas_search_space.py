import numpy as np

def generate_search_space(num_task):
    search_space = dict()
    search_space['l1'] = np.arange(100, 2000, 400)
    search_space['l2'] = np.arange(100, 2000, 400)
    search_space['l1_input'] = np.arange(0, 1)
    search_space['l2_input'] = np.arange(1, 2)
    for i in range(num_task):
        private_layer1_name = 'l{}'.format(i+3)
        private_layer1_input_name = 'l{}_input'.format(i+3)
        search_space[private_layer1_name] = np.arange(100, 2000, 400)
        search_space[private_layer1_input_name] = np.arange(0, 3)
        task_output_name = 'task{}_input'.format(i)
        search_space[task_output_name] = np.concatenate((np.arange(0, 3), np.arange(i+3, i+4)))
    return search_space