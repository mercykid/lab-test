# Lab 2

***

## Question 1: Functionality of `report_graph_analysis_pass`


The function `report_graph_analysis_pass` provide a convenient way to generate a summary report of the operations and structure of a computational graph. A `placeholder` represents a function input.The `get_attr`, `call_function`, `call_module`, and `call_method` nodes represent the operations in the method. A `get_attr` node represents the fetch of an attribute from the Module hierarchy. A `call_function` node represents a call to a Python callable, specified by the function name. A `call_module` node represents a call to the forward() function of a Module in the Module hierarchy. A `call_method` node represents a call to a given method on the 0th element of args.
***
## Question 2: Functionality of `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass`


The function `profile_statistics_analysis_pass` encapsulates the process of analyzing a computational graph to collect statistics related to weights and activations based on the provided parameters. The `report_node_meta_param_analysis_pass` function conducts meta-parameter analysis on nodes within a graph and generates a report summarizing the analysis. 
***

## Question 3: Quantize_transform_pass

The configuration of the quantize transform pass specifies that only Linear layers will undergo quantization. Given the hierarchical structure of the MaseGraph, which contains only one Linear layer, so there is only one operation (OP) will be altered after applying the `quantize_transform_pass`.
***

## Question 4: Traverse `mg` and `ori_mg`

### Code for traverse the `ori_mg`:

```
pass_args = {
    "by": "type",                                                            # collect statistics by node name
    "target_weight_nodes": ["linear"],                                       # collect weight statistics for linear layers
    "target_activation_nodes": ["relu"],                                     # collect activation statistics for relu layers
    "weight_statistics": {
        "variance_precise": {"device": "cpu", "dims": "all"},                # collect precise variance of the weight
    },
    "activation_statistics": {
        "range_quantile": {"device": "cpu", "dims": "all", "quantile": 0.97} # collect 97% quantile of the activation range
    },
    "input_generator": input_generator,                                      # the input generator for feeding data to the model
    "num_samples": 32,                                                       # feed 32 samples to the model
}
_ = report_graph_analysis_pass(ori_mg)
ori_mg, _ = profile_statistics_analysis_pass(ori_mg, pass_args)
ori_mg, _ = report_node_meta_param_analysis_pass(ori_mg, {"which": ("software",)})
```

### Result for graph analysis pass of `ori_mg`:
```
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    return seq_blocks_3
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 4, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]
```
### Profile statistics analysis for `ori_mg`:

```
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| Node name    | Fx Node op   | Mase type           | Mase op      | Software Param                                                                           |
+==============+==============+=====================+==============+==========================================================================================+
| x            | placeholder  | placeholder         | placeholder  | {'results': {'data_out_0': {'stat': {}}}}                                                |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_0 | call_module  | module              | batch_norm1d | {'args': {'bias': {'stat': {}},                                                          |
|              |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|              |              |                     |              |           'running_mean': {'stat': {}},                                                  |
|              |              |                     |              |           'running_var': {'stat': {}},                                                   |
|              |              |                     |              |           'weight': {'stat': {}}},                                                       |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_1 | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 512,                        |
|              |              |                     |              |                                                     'max': 2.2610883712768555,           |
|              |              |                     |              |                                                     'min': -1.723684310913086,           |
|              |              |                     |              |                                                     'range': 3.9847726821899414}}}},     |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_2 | call_module  | module_related_func | linear       | {'args': {'bias': {'stat': {'variance_precise': {'count': 5,                             |
|              |              |                     |              |                                                  'mean': -0.002756827976554632,          |
|              |              |                     |              |                                                  'variance': 0.07052876800298691}}},     |
|              |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|              |              |                     |              |           'weight': {'stat': {'variance_precise': {'count': 80,                          |
|              |              |                     |              |                                                    'mean': 0.008668387308716774,         |
|              |              |                     |              |                                                    'variance': 0.028130000457167625}}}}, |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_3 | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 160,                        |
|              |              |                     |              |                                                     'max': 1.6148279905319214,           |
|              |              |                     |              |                                                     'min': -1.4658207893371582,          |
|              |              |                     |              |                                                     'range': 3.080648899078369}}}},      |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| output       | output       | output              | output       | {'args': {'data_in_0': {'stat': {}}}}                                                    |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
```
### Code for traverse the `mg`:

```
pass_args = {
    "by": "type",                                                            # collect statistics by node name
    "target_weight_nodes": ["linear"],                                       # collect weight statistics for linear layers
    "target_activation_nodes": ["relu"],                                     # collect activation statistics for relu layers
    "weight_statistics": {
        "variance_precise": {"device": "cpu", "dims": "all"},                # collect precise variance of the weight
    },
    "activation_statistics": {
        "range_quantile": {"device": "cpu", "dims": "all", "quantile": 0.97} # collect 97% quantile of the activation range
    },
    "input_generator": input_generator,                                      # the input generator for feeding data to the model
    "num_samples": 32,                                                       # feed 32 samples to the model
}
_ = report_graph_analysis_pass(mg)
mg, _ = profile_statistics_analysis_pass(mg, pass_args)
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})
```

### Result for graph analysis pass of `mg`:
```
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    return seq_blocks_3
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 4, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]
```
### Profile statistics analysis for `mg`:

```
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| Node name    | Fx Node op   | Mase type           | Mase op      | Software Param                                                                           |
+==============+==============+=====================+==============+==========================================================================================+
| x            | placeholder  | placeholder         | placeholder  | {'results': {'data_out_0': {'stat': {}}}}                                                |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_0 | call_module  | module              | batch_norm1d | {'args': {'bias': {'stat': {}},                                                          |
|              |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|              |              |                     |              |           'running_mean': {'stat': {}},                                                  |
|              |              |                     |              |           'running_var': {'stat': {}},                                                   |
|              |              |                     |              |           'weight': {'stat': {}}},                                                       |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_1 | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 512,                        |
|              |              |                     |              |                                                     'max': 2.3464934825897217,           |
|              |              |                     |              |                                                     'min': -1.7889368534088135,          |
|              |              |                     |              |                                                     'range': 4.135430335998535}}}},      |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_2 | call_module  | module_related_func | linear       | {'args': {'bias': {'stat': {'variance_precise': {'count': 5,                             |
|              |              |                     |              |                                                  'mean': -0.002756827976554632,          |
|              |              |                     |              |                                                  'variance': 0.07052876800298691}}},     |
|              |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|              |              |                     |              |           'weight': {'stat': {'variance_precise': {'count': 80,                          |
|              |              |                     |              |                                                    'mean': 0.008668387308716774,         |
|              |              |                     |              |                                                    'variance': 0.028130000457167625}}}}, |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_3 | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 160,                        |
|              |              |                     |              |                                                     'max': 2.4945321083068848,           |
|              |              |                     |              |                                                     'min': -1.2733203172683716,          |
|              |              |                     |              |                                                     'range': 3.767852306365967}}}},      |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| output       | output       | output              | output       | {'args': {'data_in_0': {'stat': {}}}}                                                    |
+--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
```
***
## Question 5: Quantisation flow for JSC-New network

### set up for JSC-New model and dataset
```
batch_size = 64
model_name = "jsc-new"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()
# 📝️ change this CHECKPOINT_PATH to the one you trained in Lab1
CHECKPOINT_PATH = "../mase_output/jsc-new_classification_jsc_2024-01-24/software/training_ckpts/best-v1.ckpt"
model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)
```
### quantize transform for JSC-New network

```
| Original type   | OP           |   Total |   Changed |   Unchanged |
|-----------------+--------------+---------+-----------+-------------|
| BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
| Linear          | linear       |       3 |         3 |           0 |
| ReLU            | relu         |       4 |         0 |           4 |
| output          | output       |       1 |         0 |           1 |
| x               | placeholder  |       1 |         0 |           1 |
```
***
## Question 6: Weight analysis for `ori_mg` and `mg`

### Weight e.g. of JSC-New network:
```
tensor([ 0.0031,  0.0985, -0.2065, -0.2162, -0.0902,  0.0881,  0.0138,  0.1775,
        -0.0033,  0.0792, -0.0429, -0.0349, -0.2325, -0.1502, -0.1796, -0.0337,
         0.0378,  0.0627, -0.2438, -0.1465,  0.0151,  0.1431, -0.0279,  0.1770,
        -0.0169,  0.0261,  0.2241, -0.1816, -0.1420, -0.0396, -0.1486,  0.2870,
        -0.0951, -0.0612, -0.1287, -0.2231, -0.1301,  0.1888,  0.0923,  0.1009,
        -0.0062, -0.1405,  0.0069, -0.2801, -0.1974, -0.1532,  0.2418,  0.1766,
        -0.1086, -0.0104,  0.1853,  0.2566,  0.0220,  0.0217,  0.1274, -0.1504,
         0.0064, -0.2080, -0.1760, -0.1299,  0.0780,  0.0751, -0.1867,  0.1006,
         0.1269, -0.0857,  0.0236,  0.0928,  0.1805,  0.2870, -0.1584, -0.0382,
         0.1329,  0.1791,  0.1661,  0.2116,  0.0317, -0.1826,  0.0075, -0.2081,
        -0.1825,  0.1358,  0.1449, -0.3112,  0.0181, -0.0647, -0.0368, -0.1414,
         0.1008, -0.1841,  0.0580,  0.1066,  0.1395,  0.0648, -0.3318, -0.1712,
         0.1592, -0.0329, -0.2641, -0.2212,  0.1935,  0.1275, -0.0462, -0.0644,
        -0.1749,  0.0016, -0.1576, -0.1639, -0.0061,  0.0493, -0.2012,  0.0641,
        -0.1103,  0.1859,  0.1489, -0.3122,  0.0029,  0.0207,  0.0240,  0.0537,
         0.0858,  0.0146, -0.1304,  0.0906, -0.2346, -0.1313, -0.0350, -0.1634])
```
<left>
    <img src=ori_mg.png width="40%" />
    <br/>
    <font color="AAAAAA">Figure 1: Weight distribution e.g. for JSC-New network</font>
</left>

### Quantized Weight e.g. of JSC-New network:
```
tensor([ 0.0000,  0.1250, -0.1875, -0.1875, -0.0625,  0.0625,  0.0000,  0.1875,
        -0.0000,  0.0625, -0.0625, -0.0625, -0.2500, -0.1250, -0.1875, -0.0625,
         0.0625,  0.0625, -0.2500, -0.1250,  0.0000,  0.1250, -0.0000,  0.1875,
        -0.0000,  0.0000,  0.2500, -0.1875, -0.1250, -0.0625, -0.1250,  0.3125,
        -0.1250, -0.0625, -0.1250, -0.2500, -0.1250,  0.1875,  0.0625,  0.1250,
        -0.0000, -0.1250,  0.0000, -0.2500, -0.1875, -0.1250,  0.2500,  0.1875,
        -0.1250, -0.0000,  0.1875,  0.2500,  0.0000,  0.0000,  0.1250, -0.1250,
         0.0000, -0.1875, -0.1875, -0.1250,  0.0625,  0.0625, -0.1875,  0.1250,
         0.1250, -0.0625,  0.0000,  0.0625,  0.1875,  0.3125, -0.1875, -0.0625,
         0.1250,  0.1875,  0.1875,  0.1875,  0.0625, -0.1875,  0.0000, -0.1875,
        -0.1875,  0.1250,  0.1250, -0.3125,  0.0000, -0.0625, -0.0625, -0.1250,
         0.1250, -0.1875,  0.0625,  0.1250,  0.1250,  0.0625, -0.3125, -0.1875,
         0.1875, -0.0625, -0.2500, -0.2500,  0.1875,  0.1250, -0.0625, -0.0625,
        -0.1875,  0.0000, -0.1875, -0.1875, -0.0000,  0.0625, -0.1875,  0.0625,
        -0.1250,  0.1875,  0.1250, -0.3125,  0.0000,  0.0000,  0.0000,  0.0625,
         0.0625,  0.0000, -0.1250,  0.0625, -0.2500, -0.1250, -0.0625, -0.1875])
```
<left>
    <img src=mg.png width="40%" />
    <br/>
    <font color="AAAAAA">Figure 2: Quantized Weight distribution e.g. for JSC-New network</font>
</left>

### Summary:
The provided weight tensors showcase the transformation of weights after applying the quantized transform pass. In the originial JSC-New weight tensor, the values are floating-point numbers, whereas the quantized JSC-New weight tensor demonstrates quantized weights. The quantization process involves converting floating-point values to fixed-point representations. In this case, it appears that the quantized weights are represented with a precision where the values after the decimal point are all multiples of 0.0625, indicating a quantization scheme with a step size of 0.0625.
***
## Question7: Command line interface transform

### Configuration for `jsc_new_by_type.toml`:
```
# basics
model = "jsc-new"
dataset = "jsc"
task = "cls"

max_epochs = 5
batch_size = 512
learning_rate = 1e-2
accelerator = "cuda"
project = "jsc-new"
seed = 42
log_every_n_steps = 5
# load_name = "../mase_output/jsc-new_classification_jsc_2024-01-24/software/training_ckpts/best.ckpt"
# load_name = "../mase_output/jsc_toy_tiny/software/training_ckpts/best.ckpt"
load_type = "pl"

[passes.quantize]
by = "type"
[passes.quantize.default.config]
name = "NA"
[passes.quantize.linear.config]
name = "integer"
"data_in_width" = 8
"data_in_frac_width" = 4
"weight_width" = 8
"weight_frac_width" = 4
"bias_width" = 8
"bias_frac_width" = 4

[search.search_space]
name = "graph/quantize/mixed_precision_ptq"

[search.search_space.setup]
by = "name"

[search.search_space.seed.default.config]
# the only choice "NA" is used to indicate that layers are not quantized by default
name = ["NA"]

[search.search_space.seed.linear.config]
# if search.search_space.setup.by = "type", this seed will be used to quantize all torch.nn.Linear/ F.linear
name = ["integer"]
data_in_width = [4, 8]
data_in_frac_width = ["NA"] # "NA" means data_in_frac_width = data_in_width // 2
weight_width = [2, 4, 8]
weight_frac_width = ["NA"]
bias_width = [2, 4, 8]
bias_frac_width = ["NA"]

[search.search_space.seed.seq_blocks_2.config]
# if search.search_space.setup.by = "name", this seed will be used to quantize the mase graph node with name "seq_blocks_2"
name = ["integer"]
data_in_width = [4, 8]
data_in_frac_width = ["NA"]
weight_width = [2, 4, 8]
weight_frac_width = ["NA"]
bias_width = [2, 4, 8]
bias_frac_width = ["NA"]

[search.strategy]
name = "optuna"
eval_mode = true

[search.strategy.sw_runner.basic_evaluation]
data_loader = "val_dataloader"
num_samples = 512

[search.strategy.hw_runner.average_bitwidth]
compare_to = 32 # compare to FP32

[search.strategy.setup]
n_jobs = 1
n_trials = 20
timeout = 20000
sampler = "tpe"
# sum_scaled_metrics = true # single objective
# direction = "maximize"
sum_scaled_metrics = false # multi objective

[search.strategy.metrics]
# loss.scale = 1.0
# loss.direction = "minimize"
accuracy.scale = 1.0
accuracy.direction = "maximize"
average_bitwidth.scale = 0.2
average_bitwidth.direction = "minimize"
```

### Result for command line interface transform:
```
+-------------------------+--------------------------+--------------+-----------------+--------------------------+
| Name                    |         Default          | Config. File | Manual Override |        Effective         |
+-------------------------+--------------------------+--------------+-----------------+--------------------------+
| task                    |      classification      |     cls      |       cls       |           cls            |
| load_name               |           None           |              |                 |           None           |
| load_type               |            mz            |      pl      |                 |            pl            |
| batch_size              |           128            |     512      |                 |           512            |
| to_debug                |          False           |              |                 |          False           |
| log_level               |           info           |              |                 |           info           |
| report_to               |       tensorboard        |              |                 |       tensorboard        |
| seed                    |            0             |      42      |                 |            42            |
| quant_config            |           None           |              |                 |           None           |
| training_optimizer      |           adam           |              |                 |           adam           |
| trainer_precision       |         16-mixed         |              |                 |         16-mixed         |
| learning_rate           |          1e-05           |     0.01     |                 |           0.01           |
| weight_decay            |            0             |              |                 |            0             |
| max_epochs              |            20            |      5       |                 |            5             |
| max_steps               |            -1            |              |                 |            -1            |
| accumulate_grad_batches |            1             |              |                 |            1             |
| log_every_n_steps       |            50            |      5       |                 |            5             |
| num_workers             |            32            |              |        0        |            0             |
| num_devices             |            1             |              |                 |            1             |
| num_nodes               |            1             |              |                 |            1             |
| accelerator             |           auto           |     cuda     |                 |           cuda           |
| strategy                |           auto           |              |                 |           auto           |
| is_to_auto_requeue      |          False           |              |                 |          False           |
| github_ci               |          False           |              |                 |          False           |
| disable_dataset_cache   |          False           |              |                 |          False           |
| target                  |   xcu250-figd2104-2L-e   |              |                 |   xcu250-figd2104-2L-e   |
| num_targets             |           100            |              |                 |           100            |
| is_pretrained           |          False           |              |                 |          False           |
| max_token_len           |           512            |              |                 |           512            |
| project_dir             | /home/mercy_kid/project/ |              |                 | /home/mercy_kid/project/ |
|                         | content/mase/mase_output |              |                 | content/mase/mase_output |
| project                 |           None           |   jsc-new    |                 |         jsc-new          |
| model                   |           None           |   jsc-new    |                 |         jsc-new          |
| dataset                 |           None           |     jsc      |                 |           jsc            |
+-------------------------+--------------------------+--------------+-----------------+--------------------------+
```

## Optional: Write a pass

```
import torch
from thop import profile

def count_bitwise_ops(model):
    total_bitwise_ops = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):

            input_size = (1,) + tuple(module.weight.shape[1:])
            output_size = (1,) + tuple(module.forward(torch.zeros(input_size)).shape[1:])
            num_ops = input_size[1] * input_size[2] * input_size[3] * output_size[1] * output_size[2] * output_size[3]
            total_bitwise_ops += num_ops
    return total_bitwise_ops

def flop_bitwise_pass(graph, pass_args={'dummy_in':None}):
    record_result={"FLOP":None,"Bit-op":None}
    input_tensor = pass_args['dummy_in']
    flops, params = profile(graph.model, inputs=(input_tensor,))
    record_result["FLOP"]=flops
    bit_ops = count_bitwise_ops(graph.model)
    record_result["Bit-op"]=bit_ops
    return graph,record_result

```