# Lab 4

## Question 1: Edit of `redefine_linear_transform_pass`

```
def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    print(main_config)
    default = main_config.pop('default', None)
    print(default)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    pre_in=1
    pre_out=1
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        
        if name is not None:
            ori_module = graph.modules[node.target]
            # Initially assign in_features and out_features to be 16
            # otherwise the size of in_features and out_features will be enlarged each time applying pass
            in_features = 16
            out_features = 16
            bias = ori_module.bias
            if name == "output_only":
                in_features = ori_module.in_features
                out_features = out_features * config["channel_multiplier"]
                pre_out=config["channel_multiplier"]
            elif name == "both":
                in_features = in_features * pre_out
                out_features = out_features * config["channel_multiplier"]
                pre_out = pre_in
                pre_in = config["channel_multiplier"]
            elif name == "input_only":
                in_features = in_features * pre_in
                out_features = ori_module.out_features
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}



pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "both",
        "channel_multiplier":4,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        }
    },
}

# this performs the architecture transformation based on the config
mg, _ = redefine_linear_transform_pass(
    graph=mg, pass_args={"config": pass_config})

num_para = sum(p.numel() for p in mg.model.parameters() if p.requires_grad)
mg.model
```

It is also important to note that `nn.ReLU`, it will automatically adapts to the output channel size (out_features) of the previous layers. The following is the example model output after editing the `redefine_linear_transform_pass`:

```
GraphModule(
  (seq_blocks): Module(
    (0): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=16, out_features=32, bias=True)
    (3): ReLU(inplace=True)
    (4): Linear(in_features=32, out_features=64, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=64, out_features=5, bias=True)
    (7): ReLU(inplace=True)
  )
)
```
## Question 2-4

First step, the training procedure was carried out using the following command line:
```
!./ch train jsc-d jsc --max-epochs 10 --batch-size 256
```
<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src = train_acc_three.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 1: Training Accuracy for JSC-Three-Linear-Layer network</font>
			</center>
		</td>
		<td>
			<center>
				<img src = val_acc_three.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 2: Validation Accuracy for JSC-Three-Linear-Layer network</font>
			</center>
		</td>
	</tr>
</table>
Subsequently, the additional pass `redefine_linear_transform_pass` described in Question 1 is added into the `quantize.py` script. This pass is introduced to facilitate the transformation of linear layers within the network for different `in_features` and `out_features` channels. 

After adding the pass, the search space configuration is specified in `jsc-three_by_type_grid.toml` with a series of `channel_multiplier` with values of [1, 2, 4, 6, 8]. 

```
# basics
model = "jsc-d"
dataset = "jsc"
task = "cls"

max_epochs = 5
batch_size = 512
learning_rate = 1e-2
accelerator = "cuda"
project = "jsc-threeLinear"
seed = 42
log_every_n_steps = 5
# load_name = "../mase_output/toy_toy_tiny/software/training_ckpts/best.ckpt"
load_type = "pl"

[passes.quantize]
by = "type"
[passes.quantize.default.config]
name = "NA"

[search.search_space]
name = "graph/quantize/mixed_precision_ptq"

[search.search_space.setup]
by = "name"

[search.search_space.seed.default.config]
# the only choice "NA" is used to indicate that layers are not quantized by default
name = ["NA"]


[search.search_space.seed.seq_blocks_2.config]
# if search.search_space.setup.by = "name", this seed will be used to quantize the mase graph node with name "seq_blocks_2"
name = ["output_only"]
channel_multiplier = [1,2,4,6,8,10]

[search.search_space.seed.seq_blocks_4.config]
name = ["both"]
channel_multiplier = [1,2,4,6,8,10]

[search.search_space.seed.seq_blocks_6.config]
name = ["input_only"]


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
n_trials = 40
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

Eventually, we could get the best trial by running the search algorithm. The detailed information of best trial is given below:

```
{
    "0":{
        "number":0,
        "value":[
            0.3267064095,
            6.4
        ],
        "software_metrics":{
            "loss":1.6031441689,
            "accuracy":0.3267064095
        },
        "hardware_metrics":{
            "average_bitwidth":32,
            "memory_density":1.0
        },
        "scaled_metrics":{
            "accuracy":0.3267064095,
            "average_bitwidth":6.4
        },
        "sampled_config":{
            "seq_blocks_1":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_2":{
                "config":{
                    "name":"output_only",
                    "channel_multiplier":6
                }
            },
            "seq_blocks_3":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_4":{
                "config":{
                    "name":"both",
                    "channel_multiplier":6
                }
            },
            "seq_blocks_5":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_6":{
                "config":{
                    "name":"input_only"
                }
            },
            "seq_blocks_7":{
                "config":{
                    "name":null
                }
            },
            "by":"name",
            "accelerator":"cuda"
        }
    }
}
```
## Optional Task: search space for VGG

### write a pass to redefine vgg7 network by changing the channel dimension
```
def instantiate_Conv(in_features, out_features, kernel_size=3,padding=1):
    return nn.Conv2d(
        in_channels=in_features,
        out_channels=out_features,
        kernel_size=kernel_size,
        padding=padding)
def instantiate_BatchNorm(out_features):
    return nn.BatchNorm2d(
        num_features=out_features
    )
    
def redefine_vgg_pass(graph, pass_args=None):
    main_config = pass_args# .pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    pre_out=1
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        #print(node.target in graph.modules.keys())
        
        if 'feature_layers' in get_parent_name(node.target):
            ori_module = graph.modules[node.target]
            if isinstance(ori_module,nn.Conv2d):
                in_features = 128
                out_features = 128
                if name is not None:
                    if name == "output_only":
                        in_features = ori_module.in_channels
                        out_features = out_features * config["channel_multiplier"]
                        pre_out=config["channel_multiplier"]
                    elif name == "both":
                        in_features = in_features * pre_out
                        out_features = out_features * config["channel_multiplier"]
                        pre_out = config["channel_multiplier"]
                    elif name == "input_only":
                        in_features = in_features * pre_out
                        out_features = ori_module.out_channels
                    new_module = instantiate_Conv(in_features, out_features)
                    parent_name, name = get_parent_name(node.target)
                    setattr(graph.modules[parent_name], name, new_module)
            elif isinstance(ori_module,nn.BatchNorm2d):
                new_module =instantiate_BatchNorm(out_features=out_features)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
    return graph, {}
```
### Configuration of `vgg7_by_type.toml`
```
# basics
model = "vgg7"
dataset = "cifar10"
task = "cls"

max_epochs = 10
batch_size = 128
learning_rate = 1e-1
accelerator = "gpu"
project = "vgg7-brute"
seed = 42
log_every_n_steps = 5
# load_name = "../mase_output/toy_toy_tiny/software/training_ckpts/best.ckpt"
load_type = "pl"

[passes.quantize]
by = "type"
[passes.quantize.default.config]
name = "NA"

[search.search_space]
name = "graph/quantize/mixed_precision_ptq"

[search.search_space.setup]
by = "name"

[search.search_space.seed.default.config]
# the only choice "NA" is used to indicate that layers are not quantized by default
name = ["NA"]


[search.search_space.seed.feature_layers_0.config]
# if search.search_space.setup.by = "name", this seed will be used to quantize the mase graph node with name "seq_blocks_2"
name = ["output_only"]
channel_multiplier = [1,2]

[search.search_space.seed.feature_layers_3.config]
name = ["both"]
channel_multiplier = [1,2,4,8]

[search.search_space.seed.feature_layers_7.config]
name = ["both"]
channel_multiplier = [1,2,4,8]

[search.search_space.seed.feature_layers_10.config]
name = ["both"]
channel_multiplier = [1,2,4,8]

[search.search_space.seed.feature_layers_14.config]
name = ["both"]
channel_multiplier = [1,2,4,8]

[search.search_space.seed.feature_layers_17.config]
name = ["input_only"]


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
n_trials = 100
timeout = 20000
sampler = "brute"
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

### Result of best trial
```
{
    "0":{
        "number":3,
        "value":[
            0.1168477312,
            6.4
        ],
        "software_metrics":{
            "loss":2.3963468075,
            "accuracy":0.1168477312
        },
        "hardware_metrics":{
            "average_bitwidth":32,
            "memory_density":1.0
        },
        "scaled_metrics":{
            "accuracy":0.1168477312,
            "average_bitwidth":6.4
        },
        "sampled_config":{
            "feature_layers_0":{
                "config":{
                    "name":"output_only",
                    "channel_multiplier":2
                }
            },
            "feature_layers_2":{
                "config":{
                    "name":null
                }
            },
            "feature_layers_3":{
                "config":{
                    "name":"both",
                    "channel_multiplier":2
                }
            },
            "feature_layers_5":{
                "config":{
                    "name":null
                }
            },
            "feature_layers_7":{
                "config":{
                    "name":"both",
                    "channel_multiplier":2
                }
            },
            "feature_layers_9":{
                "config":{
                    "name":null
                }
            },
            "feature_layers_10":{
                "config":{
                    "name":"both",
                    "channel_multiplier":4
                }
            },
            "feature_layers_12":{
                "config":{
                    "name":null
                }
            },
            "feature_layers_14":{
                "config":{
                    "name":"both",
                    "channel_multiplier":4
                }
            },
            "feature_layers_16":{
                "config":{
                    "name":null
                }
            },
            "feature_layers_17":{
                "config":{
                    "name":"input_only"
                }
            },
            "feature_layers_19":{
                "config":{
                    "name":null
                }
            },
            "classifier_0":{
                "config":{
                    "name":null
                }
            },
            "classifier_1":{
                "config":{
                    "name":null
                }
            },
            "classifier_2":{
                "config":{
                    "name":null
                }
            },
            "classifier_3":{
                "config":{
                    "name":null
                }
            },
            "last_layer":{
                "config":{
                    "name":null
                }
            },
            "by":"name",
            "accelerator":"cuda"
        }
    }
}
```