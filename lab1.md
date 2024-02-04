# Lab 1

## Varying the parameters

In the first section, we were asked to figure out the impact of hyper parameters on the performance of learning network. The three parameters are: Max epochs, Batch size and learning rate. 

The setting of baseline hyper parameters of "JSC-tiny" network are 40 epochs, 256 batch size and 1e-5 learning rate.

****

## Question 1: Batch Size

The 'batch-size' parameter determines size of training batchs utilised in each iteration. The batch size is set to 128, 256, 512 and 1024 to investigate the impact of batch size on the performance of JSC-Tiny network.

<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src=Epochs_bs.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 1a: Number of epoch versus time steps for different 'batch-size' values</font>
			</center>
		</td>
		<td>
			<center>
				<img src=Learning_rate_bs.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 1b: Learning rate for different 'batch-size' values</font>
			</center>
		</td>
	</tr>
</table>

<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src=Training_acc_bs.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 1c: Training Accuracy for different 'batch-size' values</font>
			</center>
		</td>
		<td>
			<center>
				<img src=Training_loss_bs.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 1d: Training Loss for different 'batch-size' values</font>
			</center>
		</td>
	</tr>
</table>
<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src=val_acc_bs.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 1e: Validation Accuracy for different 'batch-size' values</font>
			</center>
		</td>
		<td>
			<center>
				<img src=val_loss_epoch.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 1f: Validation Loss for different 'batch-size' values</font>
			</center>
		</td>
	</tr>
</table>

Figure 2a illustrates that the time taken for training process reduced as the batch size increases as larger batch size facilitates the computation of training process which could improve the efficiency of training process. From Figure 2c to Figure 2f, the training and validation accuracy(loss) graph shows that smaller batch size would lead to higher accuracy and lower loss as smaller batch sizes might provide more smooth update for model's trainable parameters and leading to faster convergence.

## Question 2: Max epochs

The 'max-epochs' parameter decides the number of times the training dataset is processed by the learning network. By increasing the maximum number of epochs, the learning network would have more oppotunities to improve itself by learning from the training data. However, it could also increas the risk of overfitting. The 'max-epochs' parameter is adjusted to 10,20,40 and 60 to investigate the influence of 'max-epoch' parameter.

<!-- <center class="half">
    <img src=Training_acc_epoch.png alt="Image 1" width="400"/> 
    <figcaption>Figure 1a: Training Accuracy for different 'max-epoch' values</figcaption>
    <img src=Training_loss_epoch.png alt="Image 2" width="400"/>
    <figcaption>Figure 1b: Training Loss for different 'max-epoch' values</figcaption>
</center> -->

<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src=Epochs_epoch.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 2a: Different maximum number of epoch</font>
			</center>
		</td>
		<td>
			<center>
				<img src=Learning_rate_epoch.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 2b: Learning rate for different 'max-epoch' values</font>
			</center>
		</td>
	</tr>
</table>

<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src=Training_acc_epoch.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 2c: Training Accuracy for different 'max-epoch' values</font>
			</center>
		</td>
		<td>
			<center>
				<img src=Training_loss_epoch.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 2d: Training Loss for different 'max-epoch' values</font>
			</center>
		</td>
	</tr>
</table>
<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src=val_acc_epoch.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 2e: Validation Accuracy for different 'max-epoch' values</font>
			</center>
		</td>
		<td>
			<center>
				<img src=val_loss_epoch.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 2f: Validation Loss for different 'max-epoch' values</font>
			</center>
		</td>
	</tr>
</table>

From training and validation accuracy(loss) graph, as the maximum number of epochs increases, the performance of the JSC-Tiny network became better.

## Question 3: Learning Rate
The 'learning-rate' parameter determines the step size during the optimization process. The value of learning rate is set to 1e-3, 1e-4, 1e-5 and 1e-6 to investigate its impact on the performance of JSC-tiny network.
<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src=Epochs_lr.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 3a: Number of epoch versus time steps for different 'learning-rate' values</font>
			</center>
		</td>
		<td>
			<center>
				<img src=Learning_rate_lr.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 3b: Learning rate for different 'learning-rate' values</font>
			</center>
		</td>
	</tr>
</table>

<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src=Training_acc_lr.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 3c: Training Accuracy for different 'learning-rate' values</font>
			</center>
		</td>
		<td>
			<center>
				<img src=Training_loss_lr.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 3d: Training Loss for different 'learning-rate' values</font>
			</center>
		</td>
	</tr>
</table>
<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src=val_acc_lr.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 3e: Validation Accuracy for different 'learning-rate' values</font>
			</center>
		</td>
		<td>
			<center>
				<img src=val_loss_lr.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 3f: Validation Loss for different 'learning-rate' values</font>
			</center>
		</td>
	</tr>
</table>
In Figure 3b, it is evident that when the initial learning rate is set to a higher value, such as 1e-3, the learning rate experiences pronounced adjustments (reductions) with each update iteration, eventually converging towards the value of 1e-6. Conversely, for a lower initial learning rate, like 1e-6, the learning rate remains relatively stable throughout the training process, exhibiting minimal fluctuations with each update.
Illustrated in training and validation accuracy (loss) graph, with larger initial learning rate, eventually both training and validation accuracy is higher. On the contrary, with smaller learning rate, both training and validation accuracy is lower. A larger initial learning rate could facilitate faster exploration throughout the parameter space, allowing the optimization algorithm to make more significant adjustments to the model parameters during each update step. Larger initial learning rate could help the model escape the local minima and find the region of the parameter assoicated with optimal weight. 

Both learning rate and batch size would affect the convegence and stablity of training. Larger learning rate leads to faster convergence during training but also introduce instability due to large weight updating. Therefore, choosing a large initial learning rate and generally reduce the learning rate could significantly improve the efficiency of training. Similiarly, smaller batch size results in more frequent updating for parameters which eventually would cause faster convergence during the training but also increase the risk of overfitting.

## Question 4: Implement of new network with 10X more parameters
```
class JSC_New(nn.Module):
    def __init__(self, info):
        super(JSC_New, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 8),  # linear              # 2
            nn.BatchNorm1d(8),  # output_quant       # 3
            nn.ReLU(8),  # 4
            # 2nd LogicNets Layer
            nn.Linear(8, 64),  # 5
            nn.BatchNorm1d(64),  # 6
            nn.ReLU(64),  # 7
            # 3rd LogicNets Layer
            nn.Linear(64, 5),  # 8
            nn.BatchNorm1d(5),  # 9
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)
```
The architecture of JSC-New network with around 10X more parameters than JSC-toy network(130 params) is illustrated in the code above. The brief information of JSC-New network is shown in the following:

```
  | Name      | Type               | Params
-------------------------------------------------
0 | model     | JSC_New            | 1.2 K 
1 | loss_fn   | CrossEntropyLoss   | 0     
2 | acc_train | MulticlassAccuracy | 0     
3 | acc_val   | MulticlassAccuracy | 0     
4 | acc_test  | MulticlassAccuracy | 0     
5 | loss_val  | MeanMetric         | 0     
6 | loss_test | MeanMetric         | 0     
-------------------------------------------------
1.2 K     Trainable params
0         Non-trainable params
1.2 K     Total params
0.005     Total estimated model params size (MB)
```

Illustrated in training and validation accuracy(loss) graph, the training and validation accuracy of JSC-new network with 10X more parameters is higher than JSC-toy network with approximate 67% validation accuracy.
<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src=JSC_new_train_acc.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 4a: Training Accuracy for JSC-New and JSC-toy network</font>
			</center>
		</td>
		<td>
			<center>
				<img src=JSC_new_train_loss.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 4b: Training Loss for JSC-New and JSC-toy network</font>
			</center>
		</td>
	</tr>
</table>
<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src=JSC_new_val_acc.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 4c: Validation Accuracy for JSC-New and JSC-toy network</font>
			</center>
		</td>
		<td>
			<center>
				<img src=JSC_new_val_loss.png width="100%" />
				<br/>
				<font color="AAAAAA">Figure 4d: Validation Loss for JSC-New and JSC-toy network</font>
			</center>
		</td>
	</tr>
</table>