import ctypes
import numpy as np
import os
import pandas as pd
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import zstd
from DeepMapping import ndb_utils
from DeepMapping.nas_search_space import generate_search_space
from DeepMapping.ndb_utils import df_preprocess, data_manipulation
from sklearn import preprocessing
from tqdm.auto import tqdm


"""
CREDIT: The implementation code of ENAS is modified based on the version implemented in Microsoft NNI AutoML toolkit.
https://github.com/microsoft/nni/blob/master/nni/nas/oneshot/pytorch/enas.py
"""


def search_model(DATA_SUB_PATH, NUM_ITER=2000, NUM_EPOCHS_MODEL_TRAIN=5, NUM_ITR_CTRL_TRAIN=50, TASK_NAME='nas', CUDA_ID=0,
                 IS_DATA_MANIPULATION=False, EARLY_STOP_DELTA=0.0001, NUM_EARLYSTOP_PATIENT=3):
    """Search model with given configurations
    
    Args: 

    DATA_SUB_PATH : str            
        path to dataset for NAS, you can also try others, like tpch-s1/customer, tpcds-s1/customer_demographics
    NUM_ITER : int
        number of total search iteration, default=2000
    NUM_EPOCHS_MODEL_TRAIN : int
        number of model training epochs per model training iteration, default=5
    NUM_ITR_CTRL_TRAIN : int
        the gap between the iteration the controller will get trained, default=50
    TASK_NAME : str
        task name, default=nas
    CUDA_ID : str                  
        cuda id, default=0
    IS_DATA_MANIPULATION : bool
        is the model searched for data manipulation task
    
    """

    ESTIMATE_SIZE_ON_BATCH = True
    torch.cuda.set_device(CUDA_ID)
    torch.backends.cudnn.benchmark = True
    print('[INFO]: Run in CUDA: {}'.format(CUDA_ID))

    FOLDER_PREFIX = os.path.join('result/nas_result', TASK_NAME)
    if os.path.exists(FOLDER_PREFIX):
        shutil.rmtree(FOLDER_PREFIX)
    os.makedirs(FOLDER_PREFIX)

    log_file_name = '{}_{}_MODEL_TRAIN_{}_CONTROLLER_TRAIN_{}.csv'.format(
        TASK_NAME, NUM_ITER, NUM_EPOCHS_MODEL_TRAIN, NUM_ITR_CTRL_TRAIN)
    log_file_name = os.path.join(FOLDER_PREFIX, log_file_name)
    if os.path.exists(log_file_name):
        os.remove(log_file_name)


    log_message = 'N_EPOCHS|CTRL_STEP|SAMPLED_STRUT|NUM_PARAMS|MODEL_SIZE|MODEL_RATIO|AUX_SIZE|AUX_RATIO|TOTAL_SIZE|REDUCE_SIZE_RATIO|LOSS|REWARD|MODE\n'            
    with open(log_file_name, 'a') as f:
        print(log_message)
        f.write(log_message)
        f.flush()


    def encode_label(arr):
        label_encoder = preprocessing.LabelEncoder().fit(arr)
        arr_encode = label_encoder.transform(arr)
        return arr_encode, label_encoder

    # load shared library for feature extraction
    shared_utils = ctypes.CDLL(os.path.abspath("shared_utils.so")) # Or full path to file 
    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.bool_, 
                                            ndim=1,
                                            flags="C")
    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, 
                                        ndim=1,
                                        flags="C")
    shared_utils.create_fetures.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int]
    shared_utils.create_fetures_mutlt_thread_mgr.argtypes = [ND_POINTER_1, ND_POINTER_2, ctypes.c_long, ctypes.c_int32, ctypes.c_int32]

    def create_features_c_multi_thread(shared_utils, x, num_record, max_len, num_threads=4):
        # feature extraction, multi-thread implementation
        x_features_arr = np.zeros(num_record * max_len * 10, dtype=np.bool_)
        x_features_arr_ptr = shared_utils.create_fetures_mutlt_thread_mgr(x_features_arr, x, num_record, max_len, num_threads)
        features = np.frombuffer(x_features_arr_ptr.contents, dtype=np.bool_).reshape(num_record, -1).copy()
        return features
            
    def create_features_c_single_thread(shared_utils, x, num_record, max_len):
        # feature extraction, single-thread implementation
        x_features_arr = np.zeros(num_record * max_len * 10, dtype=np.bool_)
        x_features_arr_ptr = shared_utils.create_fetures(x_features_arr, x, num_record, max_len)
        x_features_sampled = np.frombuffer(x_features_arr_ptr.contents, dtype=np.bool_).reshape(num_record, -1).copy()
        return x_features_sampled

    class CustomDataset(torch.utils.data.Dataset):
        """Custom Dataset Object
        """
        def __init__(self, x, list_y, max_len, chunk_size):
            self.chunk_size = chunk_size
            self.x = x
            self.num_task = len(list_y)
            self.list_y = list_y
            self.max_len = max_len
            self.len = int(np.ceil(len(x) / self.chunk_size))
            
        def __len__(self):
            return self.len
        
        def __getitem__(self, index):
            data = []
            idx_start = index*self.chunk_size
            idx_end = (index+1)*self.chunk_size
            data_x = self.x[idx_start:idx_end]
            num_record = len(data_x)
            max_len = self.max_len
            
            shared_utils.create_fetures.restype = ctypes.POINTER(ctypes.c_bool * (num_record * max_len * 10))
            shared_utils.create_fetures_mutlt_thread_mgr.restype = ctypes.POINTER(ctypes.c_bool * (num_record * max_len * 10))

            x_features = create_features_c_multi_thread(shared_utils, data_x, num_record, max_len)
            
            data.append(torch.Tensor(x_features).type(torch.float32))
            for i in range(self.num_task):
                data_y = self.list_y[i][idx_start:idx_end]
                data.append(torch.Tensor(data_y).type(torch.long))        
            return data
        
    class StackedLSTMCell(nn.Module):
        """
        CREDIT: The implementation code of ENAS is modified based on the version implemented in Microsoft NNI AutoML toolkit.
        https://github.com/microsoft/nni/blob/master/nni/nas/oneshot/pytorch/enas.py
        """

        def __init__(self, layers, size, bias):
            super().__init__()
            self.lstm_num_layers = layers
            self.lstm_modules = nn.ModuleList([nn.LSTMCell(size, size, bias=bias)
                                            for _ in range(self.lstm_num_layers)])

        def forward(self, inputs, hidden):
            prev_h, prev_c = hidden
            next_h, next_c = [], []
            for i, m in enumerate(self.lstm_modules):
                curr_h, curr_c = m(inputs, (prev_h[i], prev_c[i]))
                next_c.append(curr_c)
                next_h.append(curr_h)
                # current implementation only supports batch size equals 1,
                # but the algorithm does not necessarily have this limitation
                inputs = curr_h[-1].view(1, -1)
            return next_h, next_c


    class ReinforceField:
        """
        A field with ``name``, with ``total`` choices. ``choose_one`` is true if one and only one is meant to be
        selected. Otherwise, any number of choices can be chosen.

        CREDIT: The implementation code of ENAS is modified based on the version implemented in Microsoft NNI AutoML toolkit.
        https://github.com/microsoft/nni/blob/master/nni/nas/oneshot/pytorch/enas.py

        """

        def __init__(self, name, total, choose_one):
            self.name = name
            self.total = total
            self.choose_one = choose_one

        def __repr__(self):
            return f'ReinforceField(name={self.name}, total={self.total}, choose_one={self.choose_one})'
        
    class ReinforceController(nn.Module):
        """
        A controller that mutates the graph with RL.

        CREDIT: The implementation code of ENAS is modified based on the version implemented in Microsoft NNI AutoML toolkit.
        https://github.com/microsoft/nni/blob/master/nni/nas/oneshot/pytorch/enas.py

        Parameters
        ----------
        fields : list of ReinforceField
            List of fields to choose.
        lstm_size : int
            Controller LSTM hidden units.
        lstm_num_layers : int
            Number of layers for stacked LSTM.
        tanh_constant : float
            Logits will be equal to ``tanh_constant * tanh(logits)``. Don't use ``tanh`` if this value is ``None``.
        skip_target : float
            Target probability that skipconnect will appear.
        temperature : float
            Temperature constant that divides the logits.
        entropy_reduction : str
            Can be one of ``sum`` and ``mean``. How the entropy of multi-input-choice is reduced.
        """

        def __init__(self, fields, lstm_size=64, lstm_num_layers=1, tanh_constant=1.5,
                    skip_target=0.4, temperature=None, entropy_reduction='sum'):
            super(ReinforceController, self).__init__()
            self.fields = fields
            self.lstm_size = lstm_size
            self.lstm_num_layers = lstm_num_layers
            self.tanh_constant = tanh_constant
            self.temperature = temperature
            self.skip_target = skip_target

            self.lstm = StackedLSTMCell(self.lstm_num_layers, self.lstm_size, False)
            self.attn_anchor = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
            self.attn_query = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
            self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
            self.g_emb = nn.Parameter(torch.randn(1, self.lstm_size) * 0.1)
            self.skip_targets = nn.Parameter(torch.tensor([1.0 - self.skip_target, self.skip_target]),  # pylint: disable=not-callable
                                            requires_grad=False)
            assert entropy_reduction in ['sum', 'mean'], 'Entropy reduction must be one of sum and mean.'
            self.entropy_reduction = torch.sum if entropy_reduction == 'sum' else torch.mean
            self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
            self.soft = nn.ModuleDict({
                field.name: nn.Linear(self.lstm_size, field.total, bias=False) for field in fields
            })
            self.embedding = nn.ModuleDict({
                field.name: nn.Embedding(field.total, self.lstm_size) for field in fields
            })

        def resample(self):
            self._initialize()
            result = dict()
            for field in self.fields:
                result[field.name] = self._sample_single(field)
            return result

        def _initialize(self):
            self._inputs = self.g_emb.data
            self._c = [torch.zeros((1, self.lstm_size),
                                dtype=self._inputs.dtype,
                                device=self._inputs.device) for _ in range(self.lstm_num_layers)]
            self._h = [torch.zeros((1, self.lstm_size),
                                dtype=self._inputs.dtype,
                                device=self._inputs.device) for _ in range(self.lstm_num_layers)]
            self.sample_log_prob = 0
            self.sample_entropy = 0
            self.sample_skip_penalty = 0

        def _lstm_next_step(self):
            self._h, self._c = self.lstm(self._inputs, (self._h, self._c))

        def _sample_single(self, field):
            self._lstm_next_step()
            logit = self.soft[field.name](self._h[-1])
            if self.temperature is not None:
                logit /= self.temperature
            if self.tanh_constant is not None:
                logit = self.tanh_constant * torch.tanh(logit)
            if field.choose_one:
                sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
                log_prob = self.cross_entropy_loss(logit, sampled)
                self._inputs = self.embedding[field.name](sampled)
            else:
                logit = logit.view(-1, 1)
                logit = torch.cat([-logit, logit], 1)  # pylint: disable=invalid-unary-operand-type
                sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
                skip_prob = torch.sigmoid(logit)
                kl = torch.sum(skip_prob * torch.log(skip_prob / self.skip_targets))
                self.sample_skip_penalty += kl
                log_prob = self.cross_entropy_loss(logit, sampled)
                sampled = sampled.nonzero().view(-1)
                if sampled.sum().item():
                    self._inputs = (torch.sum(self.embedding[field.name](sampled.view(-1)), 0) / (1. + torch.sum(sampled))).unsqueeze(0)
                else:
                    self._inputs = torch.zeros(1, self.lstm_size, device=self.embedding[field.name].weight.device)

            sampled = sampled.cpu().detach().numpy().tolist()
            self.sample_log_prob += self.entropy_reduction(log_prob)
            entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
            self.sample_entropy += self.entropy_reduction(entropy)
            if len(sampled) == 1:
                sampled = sampled[0]
            return sampled

        
    def create_nn_linear(in1, in1_size, in2, in2_size, layer_dict):
        layer_name = '{}_{}_{}_{}'.format(in1, in1_size, in2, in2_size)
        layer_dict[layer_name] = nn.Linear(in1_size, in2_size)

    def init_all(model, init_func, *params, **kwargs):
        for p in model.parameters():
            init_func(p, *params, **kwargs)

    def compute_total_num_params(sampled_layers):
        num_total_param = 0
        for layer in sampled_layers:
            input_size = int(layer.split('_')[1])
            output_size = int(layer.split('_')[3])
            num_total_param += (input_size + 1)*output_size
        return num_total_param


    def constructure_layers(sampled_structure_interpreted, num_in, list_num_outs):
        current_layer_index = 1 #  1 is the first hidden layer
        layers = []
        layers_name = list(sampled_structure_interpreted.keys())
        layers_size = list(sampled_structure_interpreted.values())
        for layer_name, layer_choice in sampled_structure_interpreted.items():
            if 'input' in layer_name and 'task' not in layer_name:
                curr_layer = layer_name.split('_')[0]
                # case: intermediate layeer
                if layer_choice == 0:
                    # input layer
                    layer_name = '{}_{}_{}_{}'.format('in', num_in, curr_layer, sampled_structure_interpreted[curr_layer])
                else:
                    prev_layer_index = sampled_structure_interpreted[layer_name]
                    prev_layer = 'l' + str(prev_layer_index)
                    layer_name = '{}_{}_{}_{}'.format(prev_layer, sampled_structure_interpreted[prev_layer], curr_layer, sampled_structure_interpreted[curr_layer])
                layers.append(layer_name)
                
            elif 'input' in layer_name and 'task' in layer_name:
                task_layer = layer_name.split('_')[0]
                task_index = int(task_layer[4:])
                
                if layer_choice == 0:
                    # input layer
                    layer_name = '{}_{}_{}_{}'.format('in', num_in, task_layer, list_num_outs[task_index])
                else:            
                    prev_layer_index = layer_choice
                    prev_layer = 'l' + str(prev_layer_index)
                    layer_name = '{}_{}_{}_{}'.format(prev_layer, sampled_structure_interpreted[prev_layer], task_layer, list_num_outs[task_index])
                
                layers.append(layer_name)
        return layers

    def interprete_structure(sampled_structre, search_space):
        used_by_task_layers = set()
        sampled_structure_interpreted = dict()
        
        for layer_name, layer_choice in sampled_structre.items():
            curr_layer = layer_name.split('_')[0]
            prev_layer_index = search_space[layer_name][layer_choice]
            sampled_structure_interpreted[layer_name] = prev_layer_index
            
            if 'task' in layer_name:
                used_layer_name = 'l' + str(prev_layer_index)
                used_by_task_layers.add(used_layer_name)

                # track it source 
                curr_iter_layer_index = prev_layer_index
                while curr_iter_layer_index != 0:
                    # get current layer input choice
                    curr_iter_input_layer = 'l' + str(curr_iter_layer_index) + '_input'
                    curr_iter_input_layer_index = search_space[curr_iter_input_layer][sampled_structre[curr_iter_input_layer]]
                    used_layer_name = 'l' + str(curr_iter_input_layer_index)
                    used_by_task_layers.add(used_layer_name)
                    curr_iter_layer_index = curr_iter_input_layer_index
                
        for layer_name, layer_choice in sampled_structure_interpreted.copy().items():
            curr_layer = layer_name.split('_')[0]
            if curr_layer not in used_by_task_layers and 'task' not in curr_layer:
                # delete unused branch
                del sampled_structure_interpreted[layer_name]
                # pass
        
        return sampled_structure_interpreted

    class NN(nn.Module):
        def __init__(self, num_in, list_num_outs, search_space):
            super(NN, self).__init__()
            self.layers_dict = dict()
            # temp fix
            search_space_keys = list(search_space.keys())
            search_space_values = list(search_space.values())
            for i in range(len(search_space)):
                iter_key = search_space_keys[i]
                if 'input' in iter_key and 'task' not in iter_key:
                    # intermediate layer
                    curr_layer = iter_key.split('_')[0]
                    # obtain the layer input indexes
                    prev_layer_idxs = search_space_values[i]
                    for curr_layer_size in search_space[curr_layer]:
                        for prev_layer_idx in prev_layer_idxs:
                            # case: prev layer is the input layer
                            if prev_layer_idx == 0:
                                create_nn_linear('in', num_in, curr_layer, curr_layer_size, self.layers_dict)   
                            else:
                            # case: prev layer is non-input layer
                                prev_layer = 'l' + str(prev_layer_idx)
                                for prev_layer_size in search_space[prev_layer]:
                                    create_nn_linear(prev_layer, prev_layer_size, curr_layer, curr_layer_size, self.layers_dict)   
                elif 'input' in iter_key and 'task' in iter_key:            
                    # case: output layer
                    curr_task = iter_key.split('_')[0]
                    curr_task_idx = int(curr_task[4:])
                    curr_task_size = list_num_outs[curr_task_idx]
                    prev_layer_idxs = search_space_values[i]
                    for prev_layer_idx in prev_layer_idxs:
                        # case: prev layer is the input layer
                        if prev_layer_idx == 0:
                            create_nn_linear('in', num_in, curr_task, curr_task_size, self.layers_dict)   
                        else:
                        # case: prev layer is non-input layer
                            prev_layer = 'l' + str(prev_layer_idx)
                            for prev_layer_size in search_space[prev_layer]:
                                create_nn_linear(prev_layer, prev_layer_size, curr_task, curr_task_size, self.layers_dict)   
            
            self.layers_dict = nn.ModuleDict(self.layers_dict)
            
        def forward(self, x, layers):
            outputs = []
            temp_result = dict()
            for layer in layers:
                layer_split = layer.split('_')
                layer_in = layer_split[0] + '_' + layer_split[1]
                layer_out = layer_split[2] + '_' + layer_split[3]
                if 'in' in layer and 'task' not in layer:
                    temp_result[layer_out] = F.relu(self.layers_dict[layer](x))
                elif 'task' not in layer:
                    temp_result[layer_out] = F.relu(self.layers_dict[layer](temp_result[layer_in]))
                elif 'in' in layer and 'task' in layer:
                    out = F.log_softmax(self.layers_dict[layer](x), dim=-1)
                    outputs.append(out)
                else:
                    out = F.log_softmax(self.layers_dict[layer](temp_result[layer_in]), dim=-1)
                    outputs.append(out)
            return outputs


    # pre-process data, like encoding
    df = pd.read_csv('dataset/{}.csv'.format(DATA_SUB_PATH), sep=',')
    df,_ = df_preprocess(df, benchmark='deepmapping')
    for col in df.columns:
            if df[col].dtypes == np.int64:
                df[col] = df[col].astype(np.int32)

    if IS_DATA_MANIPULATION:
        df, data_ori = data_manipulation(df, ops="Default")

    print(df.head(2))
    df_key = [df.columns[0]]
    list_y_encoded = []
    list_y_encoder = []

    for col in df.columns:
        if col not in df_key:
            encoded_val, encoder = encode_label(df[col])
            list_y_encoded.append(encoded_val)
            list_y_encoder.append(encoder)
    num_tasks = len(list_y_encoded)
    print('NUM task', num_tasks)

    for encoder in list_y_encoder:
        print(len(encoder.classes_))

    # load data
    x = df[df_key[0]].values.astype(np.int32)
    max_len = len(str(np.max(x)))
    y = np.array(list_y_encoded).T.astype(np.int32)
    data = np.concatenate((x.reshape(-1,1), y), axis=1, dtype=np.int32)
    data_size = df.to_records(index=False).nbytes/1024/1024
    
    # load data into pytorch dataloader object
    # the batch size can be changed to fully utilize your available GPU memory
    mydata = CustomDataset(x, list_y_encoded, max_len, chunk_size=1024*16)
    mydata_ctrl = CustomDataset(x, list_y_encoded, max_len, chunk_size=1024*2)
    train_dataloader = torch.utils.data.DataLoader(mydata, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    ctrl_dataloader = torch.utils.data.DataLoader(mydata_ctrl, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    # define search space and controller
    num_in = max_len*10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    list_num_outs = [len(encoder.classes_) for encoder in list_y_encoder]
    search_space = generate_search_space(len(list_num_outs))
    model = NN(num_in, list_num_outs, search_space)
    nas_fields = [ReinforceField(name=key, total=len(values), choose_one=True) for key, values in search_space.items()]
    controller = ReinforceController(nas_fields)
    
    # init model and controller
    model_loss = nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-07, betas=(0.9, 0.999), amsgrad=False)
    controller_optimizer = optim.Adam(controller.parameters(), lr=0.00035)
    ctrl_accuracy = torchmetrics.Accuracy().to(device)
    entropy_weight = 0.0001
    ctrl_steps_aggregate = 20
    skip_weight = 0.8
    baseline_decay = 0.999
    baseline = 0.
    grad_clip = 5

    # init model weights
    init_all(model, torch.nn.init.normal_, mean=0., std=0.05)
    init_all(controller, torch.nn.init.normal_, mean=0., std=0.05)

    # you can uncomment this to improve performance if you are using pytorch 2
    # model = torch.compile(model)
    # controller = torch.compile(controller)

    model = model.to(device)
    controller = controller.to(device)

    list_acc_metrics = [torchmetrics.Accuracy().to(device) for _ in range(num_tasks)]
    ctrl_accuracy = torchmetrics.Accuracy().to(device)

    stop_train_flag = False

    timer_nas = ndb_utils.Timer()
    timer_nas.tic()

    for n_epochs in tqdm(range(NUM_ITER)):
        # sample a model from the search space at each iteration
        sampled_structure = controller.resample()
        sampled_structure_interpreted = interprete_structure(sampled_structure, search_space)
        sampled_layers = constructure_layers(sampled_structure_interpreted, num_in, list_num_outs)

        m_acc = torchmetrics.Accuracy().to(device)
        # train each sampled model for NUM_EPOCHS_MODEL_TRAIN  
        for m_epoch in range(NUM_EPOCHS_MODEL_TRAIN):
            model.train()
            controller.eval()
            for step, batch_data in enumerate(train_dataloader):
                for i in range(len(batch_data)):
                    batch_data[i] = batch_data[i].to(device)
                x = batch_data[0]
                x = x.view(-1, num_in)
                y = batch_data[1:]
                model_optimizer.zero_grad()
                logits = model(x, sampled_layers)
                loss = 0
                for i in range(num_tasks):
                    loss += model_loss(logits[i], y[i].view(-1))
                    list_acc_metrics[i].update(logits[i], y[i].view(-1))
                loss.backward()
                model_optimizer.step()

                # migrate from nni code
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            acc_message = ''
            list_acc = []
            for i in range(num_tasks):
                task_acc = list_acc_metrics[i].compute()
                list_acc_metrics[i].reset()
                acc_message += '{:.4f},'.format(task_acc)
                list_acc.append(task_acc.cpu().detach().numpy())

            if ESTIMATE_SIZE_ON_BATCH: 
                batch_data_len = len(x)
                num_misclassfied_batch = int(batch_data_len*(1-np.min(list_acc)))
                aux_data_size_batch = sys.getsizeof(zstd.compress(data[:batch_data_len][:num_misclassfied_batch].tobytes()))/1024/1024
                # scaled to approximate all misclassified
                aux_data_size = aux_data_size_batch*len(data) / batch_data_len
            elif not ESTIMATE_SIZE_ON_BATCH:
                # approximate the all misclassified
                num_misclassfied = int(len(data)*(1-np.min(list_acc)))
                aux_data_size = sys.getsizeof(zstd.compress(data[:num_misclassfied].tobytes()))/1024/1024

            aux_size_ratio = aux_data_size / data_size
            total_num_param = compute_total_num_params(sampled_layers)
            model_size = total_num_param*4/1024/1024
            model_ratio = model_size / data_size
            total_size = model_size + aux_data_size
            reduced_size_ratio = (data_size - total_size)/data_size
            my_reward = reduced_size_ratio

            # dump intermediate result
            log_message = '{}|{}|{}|{}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{}|\n'.format(
                n_epochs, m_epoch, sampled_structure_interpreted, 
                total_num_param, model_size, model_ratio, 
                aux_data_size, aux_size_ratio, 
                total_size, reduced_size_ratio, 
                loss, my_reward, 'model')
            with open(log_file_name, 'a') as f:
                print(log_message)
                f.write(log_message)
                f.flush()
        
        if n_epochs % NUM_ITR_CTRL_TRAIN == (NUM_ITR_CTRL_TRAIN-1):
            # train controller
            model.eval()
            controller.train()
            count_meet_early_stop_condition = 0
            loss_prev = 0
            for ctrl_step, batch_data in enumerate(ctrl_dataloader):
                for i in range(len(batch_data)):
                    batch_data[i] = batch_data[i].to(device)
                x = batch_data[0]
                x = x.view(-1, num_in)
                y = batch_data[1:]
                
                sampled_structure = controller.resample()
                sampled_structure_interpreted = interprete_structure(sampled_structure, search_space)
                sampled_layers = constructure_layers(sampled_structure_interpreted, num_in, list_num_outs)
                with torch.no_grad():
                    logits = model(x, sampled_layers)
                
                acc_message = ''
                list_acc = []
                for i in range(num_tasks):
                    task_acc = ctrl_accuracy(logits[i], y[i].view(-1))
                    acc_message += '{:.4f},'.format(task_acc)
                    list_acc.append(task_acc.cpu().detach().numpy())

                if ESTIMATE_SIZE_ON_BATCH: 
                    batch_data_len = len(x)
                    num_misclassfied_batch = int(batch_data_len*(1-np.min(list_acc)))
                    aux_data_size_batch = sys.getsizeof(zstd.compress(data[:batch_data_len][:num_misclassfied_batch].tobytes()))/1024/1024
                    # scaled to approximate all misclassified
                    aux_data_size = aux_data_size_batch*len(data) / batch_data_len
                elif not ESTIMATE_SIZE_ON_BATCH:
                    # approximate the all misclassified
                    num_misclassfied = int(len(data)*(1-np.min(list_acc)))
                    aux_data_size = sys.getsizeof(zstd.compress(data[:num_misclassfied].tobytes()))/1024/1024

                aux_size_ratio = aux_data_size / data_size
                total_num_param = compute_total_num_params(sampled_layers)
                model_size = total_num_param*4/1024/1024
                model_ratio = model_size / data_size
                total_size = model_size + aux_data_size
                reduced_size_ratio = (data_size - total_size)/data_size
            
                reward = reduced_size_ratio
            
                extra_message = ''
                extra_message += 'origin reward {},'.format(reward)
                
                if entropy_weight:
                    reward += entropy_weight * controller.sample_entropy.item()
                extra_message += 'weighted reward {},'.format(reward)
                baseline = baseline * baseline_decay + reward * (1 - baseline_decay)
                extra_message += 'baseline {},'.format(baseline)
                loss = controller.sample_log_prob * (reward - baseline)     
                extra_message += 'sum loss {},'.format(loss)
                if skip_weight:
                        loss += skip_weight * controller.sample_skip_penalty
                extra_message += 'skip loss {},'.format(loss)
                loss /= ctrl_steps_aggregate
                loss.backward()
                loss_detached = loss.cpu()
                if np.abs(loss_detached-loss_prev) <= EARLY_STOP_DELTA:
                    count_meet_early_stop_condition += 1
                else:
                    count_meet_early_stop_condition = 0
                loss_prev = loss_detached
                if (ctrl_step + 1) % ctrl_steps_aggregate == 0:
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(controller.parameters(), 5)
                    controller_optimizer.step()
                    controller_optimizer.zero_grad()
                    
                    
                log_message = '{}|{}|{}|{}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{}\n'.format(
                n_epochs, ctrl_step, sampled_structure_interpreted, 
                total_num_param, model_size, model_ratio,
                aux_data_size, aux_size_ratio, 
                total_size, reduced_size_ratio, 
                loss, reward, 'Controller')
                
                with open(log_file_name, 'a') as f:
                    print(log_message)
                    f.write(log_message)
                    f.flush()
                
                if count_meet_early_stop_condition >= NUM_EARLYSTOP_PATIENT:
                    stop_train_flag = True
                    break
            
        if stop_train_flag == True:
            print("Early Stop")
            break

    t_nas = timer_nas.toc()
    print("[INFO] nas search time: {}".format(t_nas))
    best_model = interprete_structure(controller.resample(), search_space)

    # you can try to sample multiple times and select most voted searched structure
    print("[INFO] Intermediate result is written to: {}".format(log_file_name))
    return best_model    
