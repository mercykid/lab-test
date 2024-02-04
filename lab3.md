# Lab 3

## Question 1 

### Code for implementation

```
import torch
import numpy as np
from thop import profile
from torchmetrics.classification import MulticlassAccuracy
import time
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5

recorded_accs = []
recorded_latency=[]
recorded_FLOP = []
recorded_model_size = []
recorded_result = []
for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
    j = 0
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    latencys = []
    flops, params = [] , []

    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        net = mg.model
        start_time = time.time()
        preds = mg.model(xs)
        end_time = time.time()
        latency = end_time - start_time
        latencys.append(latency)
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss.detach())
        flop, param = profile(net, (xs,))
        flops.append(flop)
        if j > num_batchs:
            break
        j += 1
    
    torch.save(mg.model.state_dict(), "temp_model.pth")
    model_size = os.path.getsize("temp_model.pth")
    os.remove("temp_model.pth")
    recorded_model_size.append(model_size)
    latency_avg = np.mean(latencys)
    recorded_latency.append(np.mean(latencys))
    acc_avg = np.mean(accs)
    loss_avg = np.mean(losses)
    recorded_accs.append(acc_avg)
    flop_avg = np.mean(flops)
    recorded_FLOP.append(flop_avg)
    recorded_result.append({"acc":acc_avg,"loss":loss_avg,"latency":latency_avg,"flop":flop_avg,"model-size":model_size})
```
***
## Question 2

### Metrics result record

```
[{'acc': 0.112500004,
  'loss': 1.861436,
  'latency': 0.0007328987121582031,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.19619048,
  'loss': 1.7432919,
  'latency': 0.0007834093911307198,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.19000001,
  'loss': 1.7187221,
  'latency': 0.0006403923034667969,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.2309524,
  'loss': 1.609438,
  'latency': 0.0005945818764822823,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.10238095,
  'loss': 1.8458731,
  'latency': 0.0006569794246128627,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.25357142,
  'loss': 1.8158171,
  'latency': 0.0006076608385358538,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.08095239,
  'loss': 1.9151993,
  'latency': 0.0005749634334019252,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.1404762,
  'loss': 1.609438,
  'latency': 0.0005112716129847936,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.17142858,
  'loss': 1.8138891,
  'latency': 0.0005198206220354353,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.18571429,
  'loss': 1.7135665,
  'latency': 0.000549112047467913,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.17619048,
  'loss': 1.7541769,
  'latency': 0.0005979878561837333,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.1642857,
  'loss': 1.609438,
  'latency': 0.0005438668387276786,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.22238097,
  'loss': 1.763243,
  'latency': 0.0006195817674909319,
  'flop': 2976.0,
  'model-size': 16674},
 {'acc': 0.17380953,
  'loss': 1.7434618,
  'latency': 0.0006977149418422154,
  'flop': 2976.0,
  'mode}]
  ```

## Question 3

### Implementation of brute-force search in `optuna.oy`
```
def sampler_map(self, name):
        match name.lower():
            case "random":
                sampler = optuna.samplers.RandomSampler()
            case "tpe":
                sampler = optuna.samplers.TPESampler()
            case "nsgaii":
                sampler = optuna.samplers.NSGAIISampler()
            case "nsgaiii":
                sampler = optuna.samplers.NSGAIIISampler()
            case "qmc":
                sampler = optuna.samplers.QMCSampler()
            case "brute":
                sampler = optuna.samplers.BruteForceSampler()
            case _:
                raise ValueError(f"Unknown sampler name: {name}")
        return sampler
```
### Configuration of `jsc_new_by_type_brute.toml`
```
# basics
model = "jsc-new"
dataset = "jsc"
task = "cls"

max_epochs = 5
batch_size = 512
learning_rate = 1e-2
accelerator = "cuda"
project = "jsc-new-brute"
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
***
## Question 4: TPE search vs brute-force search

### best trials using TPE search:

```
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        1 | {'loss': 1.2, 'accuracy': 0.553}   | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.553, 'average_bitwidth': 1.6} |
|  1 |        2 | {'loss': 1.35, 'accuracy': 0.463}  | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.463, 'average_bitwidth': 0.4} |
|  2 |        7 | {'loss': 1.309, 'accuracy': 0.492} | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.492, 'average_bitwidth': 0.8} |
```
```
{
    "0":{
        "number":1,
        "value":[
            0.5530865192,
            1.6
        ],
        "software_metrics":{
            "loss":1.2000683546,
            "accuracy":0.5530865192
        },
        "hardware_metrics":{
            "average_bitwidth":8.0,
            "memory_density":4.0
        },
        "scaled_metrics":{
            "accuracy":0.5530865192,
            "average_bitwidth":1.6
        },
        "sampled_config":{
            "seq_blocks_1":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_2":{
                "config":{
                    "name":"integer",
                    "data_in_width":4,
                    "data_in_frac_width":null,
                    "weight_width":8,
                    "weight_frac_width":null,
                    "bias_width":8,
                    "bias_frac_width":null
                }
            },
            "seq_blocks_4":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_5":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_7":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_8":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_10":{
                "config":{
                    "name":null
                }
            },
            "default":{
                "config":{
                    "name":"integer",
                    "bypass":true,
                    "bias_frac_width":5,
                    "bias_width":8,
                    "data_in_frac_width":5,
                    "data_in_width":8,
                    "weight_frac_width":3,
                    "weight_width":8
                }
            },
            "accelerator":"cuda"
        }
    },
    "1":{
        "number":2,
        "value":[
            0.4634926319,
            0.4
        ],
        "software_metrics":{
            "loss":1.3501318693,
            "accuracy":0.4634926319
        },
        "hardware_metrics":{
            "average_bitwidth":2.0,
            "memory_density":16.0
        },
        "scaled_metrics":{
            "accuracy":0.4634926319,
            "average_bitwidth":0.4
        },
        "sampled_config":{
            "seq_blocks_1":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_2":{
                "config":{
                    "name":"integer",
                    "data_in_width":8,
                    "data_in_frac_width":null,
                    "weight_width":2,
                    "weight_frac_width":null,
                    "bias_width":8,
                    "bias_frac_width":null
                }
            },
            "seq_blocks_4":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_5":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_7":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_8":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_10":{
                "config":{
                    "name":null
                }
            },
            "default":{
                "config":{
                    "name":"integer",
                    "bypass":true,
                    "bias_frac_width":5,
                    "bias_width":8,
                    "data_in_frac_width":5,
                    "data_in_width":8,
                    "weight_frac_width":3,
                    "weight_width":8
                }
            },
            "accelerator":"cuda"
        }
    },
    "2":{
        "number":7,
        "value":[
            0.4918774366,
            0.8
        ],
        "software_metrics":{
            "loss":1.3092877865,
            "accuracy":0.4918774366
        },
        "hardware_metrics":{
            "average_bitwidth":4.0,
            "memory_density":8.0
        },
        "scaled_metrics":{
            "accuracy":0.4918774366,
            "average_bitwidth":0.8
        },
        "sampled_config":{
            "seq_blocks_1":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_2":{
                "config":{
                    "name":"integer",
                    "data_in_width":8,
                    "data_in_frac_width":null,
                    "weight_width":4,
                    "weight_frac_width":null,
                    "bias_width":4,
                    "bias_frac_width":null
                }
            },
            "seq_blocks_4":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_5":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_7":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_8":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_10":{
                "config":{
                    "name":null
                }
            },
            "default":{
                "config":{
                    "name":"integer",
                    "bypass":true,
                    "bias_frac_width":5,
                    "bias_width":8,
                    "data_in_frac_width":5,
                    "data_in_width":8,
                    "weight_frac_width":3,
                    "weight_width":8
                }
            },
            "accelerator":"cuda"
        }
    }
}
```
### best trials using brute-force search:

```
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        8 | {'loss': 1.495, 'accuracy': 0.408} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.408, 'average_bitwidth': 0.4} |
|  1 |       16 | {'loss': 1.391, 'accuracy': 0.458} | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.458, 'average_bitwidth': 0.8} |
|  2 |       17 | {'loss': 1.386, 'accuracy': 0.46}  | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.46, 'average_bitwidth': 1.6}  |
```
```
{
    "0":{
        "number":8,
        "value":[
            0.4078707099,
            0.4
        ],
        "software_metrics":{
            "loss":1.494977951,
            "accuracy":0.4078707099
        },
        "hardware_metrics":{
            "average_bitwidth":2.0,
            "memory_density":16.0
        },
        "scaled_metrics":{
            "accuracy":0.4078707099,
            "average_bitwidth":0.4
        },
        "sampled_config":{
            "seq_blocks_1":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_2":{
                "config":{
                    "name":"integer",
                    "data_in_width":4,
                    "data_in_frac_width":null,
                    "weight_width":2,
                    "weight_frac_width":null,
                    "bias_width":2,
                    "bias_frac_width":null
                }
            },
            "seq_blocks_4":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_5":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_7":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_8":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_10":{
                "config":{
                    "name":null
                }
            },
            "default":{
                "config":{
                    "name":"integer",
                    "bypass":true,
                    "bias_frac_width":5,
                    "bias_width":8,
                    "data_in_frac_width":5,
                    "data_in_width":8,
                    "weight_frac_width":3,
                    "weight_width":8
                }
            },
            "accelerator":"cuda"
        }
    },
    "1":{
        "number":16,
        "value":[
            0.4582196176,
            0.8
        ],
        "software_metrics":{
            "loss":1.3913549185,
            "accuracy":0.4582196176
        },
        "hardware_metrics":{
            "average_bitwidth":4.0,
            "memory_density":8.0
        },
        "scaled_metrics":{
            "accuracy":0.4582196176,
            "average_bitwidth":0.8
        },
        "sampled_config":{
            "seq_blocks_1":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_2":{
                "config":{
                    "name":"integer",
                    "data_in_width":4,
                    "data_in_frac_width":null,
                    "weight_width":4,
                    "weight_frac_width":null,
                    "bias_width":2,
                    "bias_frac_width":null
                }
            },
            "seq_blocks_4":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_5":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_7":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_8":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_10":{
                "config":{
                    "name":null
                }
            },
            "default":{
                "config":{
                    "name":"integer",
                    "bypass":true,
                    "bias_frac_width":5,
                    "bias_width":8,
                    "data_in_frac_width":5,
                    "data_in_width":8,
                    "weight_frac_width":3,
                    "weight_width":8
                }
            },
            "accelerator":"cuda"
        }
    },
    "2":{
        "number":17,
        "value":[
            0.4602891803,
            1.6
        ],
        "software_metrics":{
            "loss":1.3858699799,
            "accuracy":0.4602891803
        },
        "hardware_metrics":{
            "average_bitwidth":8.0,
            "memory_density":4.0
        },
        "scaled_metrics":{
            "accuracy":0.4602891803,
            "average_bitwidth":1.6
        },
        "sampled_config":{
            "seq_blocks_1":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_2":{
                "config":{
                    "name":"integer",
                    "data_in_width":4,
                    "data_in_frac_width":null,
                    "weight_width":8,
                    "weight_frac_width":null,
                    "bias_width":2,
                    "bias_frac_width":null
                }
            },
            "seq_blocks_4":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_5":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_7":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_8":{
                "config":{
                    "name":null
                }
            },
            "seq_blocks_10":{
                "config":{
                    "name":null
                }
            },
            "default":{
                "config":{
                    "name":"integer",
                    "bypass":true,
                    "bias_frac_width":5,
                    "bias_width":8,
                    "data_in_frac_width":5,
                    "data_in_width":8,
                    "weight_frac_width":3,
                    "weight_width":8
                }
            },
            "accelerator":"cuda"
        }
    }
}
```

Comparing the TPE search result with brute-force search result, TPE search find the best trial with less trial number of 1,2 and 7 and brute-force search used more trials to find the best trials with trial number of 8,16 and 17. Therefore, TPE search has higher sample efficiency than brute-force search.