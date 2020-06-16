# PatchUp

## A Regularization Technique for Convolutional Neural Networks

In this work, we propose PatchUp which is a regularization technique that operates in the hidden space by masking out contiguous blocks of the feature map of a random pair of samples and either mixes (Soft PatchUp) or swaps (Hard PatchUp) these selected contiguous blocks.
Our experiments verify that Hard PatchUp achieves a better generalization performance in comparison to other state-of-the-art regularization techniques for CNNs like Mixup, cutout, CutMix and ManifoldMixup on CIFAR-10, CIFAR-100, and SVHN datasets. Soft PatchUp provides the second-best performance on CIFAR-10, CIFAR-100 with PreactResnet18, PreactResnet34, and WideResnet-28-10 models and comparable result in SVHN with PreactResnet18, PreactResnet34 with ManifoldMixup. PatchUp provides significant improvements in the generalization on deformed images and better robustness against FGSM adversarial attack. 

Yo can find further detail on PatchUp in [https://arxiv.org/abs/2006.07794](https://arxiv.org/abs/2006.07794). 

### PatchUp Process:
Following image briefly describes how PatchUp works. It is the PatchUp process for two hidden representations associated with two samples randomly selected in the minibatch (a, b). X<sub>1</sub> = g<sub>k</sub><sup>(i)</sup>(a) and X<sub>2</sub> = g<sub>k</sub><sup>(i)</sup>(b) where "i" is the feature map index. Right top shows Hard PatchUp output and the right bottom shows the interpolated samples with Soft PatchUp. The yellow continuous blocks represent the interpolated selected blocks. 

(Figure 1. from the [PatchUp Paper](https://arxiv.org/abs/2006.07794).)
<br/>

![patchup_process](https://user-images.githubusercontent.com/38594307/83678128-3b3c0000-a5ab-11ea-8f38-d919ecce8d29.png)

<br/>

### Citation:

If you find this work useful and use it in your own research, please consider citing our [paper](https://arxiv.org/abs/2006.07794).
```
@misc{faramarzi2020patchup,
    title={PatchUp: A Regularization Technique for Convolutional Neural Networks},
    author={Mojtaba Faramarzi and Mohammad Amini and Akilesh Badrinaaraayanan and Vikas Verma and Sarath Chandar},
    year={2020},
    eprint={2006.07794},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
<br/>
     
## Project Structure:
Following shows the project structure and modules and files that we have in this project:

<img width="900" alt="project structure" src="https://user-images.githubusercontent.com/38594307/83950930-37eb8300-a7fc-11ea-800a-ebc03bcf41d6.jpg">

### Requirements:
In this implementation some packages were used that you can find names and their version in the requirements.txt.
To install the requirements you can do:

```
env/bin/pip install -r requirements.txt
```

## How to run experiments for CIFAR

Following are the experiment commands for CIFAR-10 for the PatchUp, ManifoldMixup, CutMix, cutout, Mixup, and DropBlock.

We first show how you can run them for CIFAR-10. And then, we indicate the parameter changes that allows you to run experiments on CIFAR-100 and SVHN.

## PatchUp 

### CIFAR-10

#### Soft PatchUp:

For PreActResent models you can run the following command.

Note: X is either preactresnet18 or preactresnet34 and at the end JobID is your job id. 
```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/patchup/soft/ --labels_per_class 5000 --valid_labels_per_class 500 --arch <X>  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --step_factors 0.1 0.1 0.1 --train patchup --alpha 2.0 --patchup_type soft --patchup_block 7 --patchup_prob 1.0 --gamma 0.75 --job_id <JobID>
```
For WideResNet-28-10:

```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/patchup/soft/ --labels_per_class 5000 --valid_labels_per_class 500 --arch wrn28_10  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --step_factors 0.1 0.1 --train patchup --alpha 2.0 --patchup_type soft --patchup_block 7 --patchup_prob 1.0 --gamma 0.75 --job_id <JobID>
```

#### Hard PatchUp:

For PreActResent models you can run the following command.

Note: X is either preactresnet18 or preactresnet34 and at the end JobID is your job id. 

```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/patchup/hard/ --labels_per_class 5000 --valid_labels_per_class 500 --arch <X>  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --step_factors 0.1 0.1 0.1 --train patchup --alpha 2.0 --patchup_type hard --patchup_block 7 --patchup_prob 0.7 --gamma 0.5 --job_id <JobID>
```

For WideResNet-28-10:

```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/patchup/soft/ --labels_per_class 5000 --valid_labels_per_class 500 --arch wrn28_10  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --step_factors 0.1 0.1 --train patchup --alpha 2.0 --patchup_type hard --patchup_block 7 --patchup_prob 0.7 --gamma 0.5 --job_id <JobID>
```
<br/>
<hr/>
<br/>

## ManifoldMixup 

### CIFAR-10

For PreActResent models you can run the following command.

Note: X is either preactresnet18 or preactresnet34 and at the end JobID is your job id. 
```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/manifold/ --labels_per_class 5000 --valid_labels_per_class 500 --arch <X>  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --step_factors 0.1 0.1 0.1 --train manifold_mixup --alpha 1.5 --job_id <JobID>
```
For WideResNet-28-10:

```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/patchup/soft/ --labels_per_class 5000 --valid_labels_per_class 500 --arch wrn28_10  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --step_factors 0.1 0.1 --train manifold_mixup --alpha 1.5 --job_id <JobID>
```
<br/>
<hr/>
<br/>

## CutMix 

### CIFAR-10

For PreActResent models you can run the following command.

Note: X is either preactresnet18 or preactresnet34 and at the end JobID is your job id. 
```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/manifold/ --labels_per_class 5000 --valid_labels_per_class 500 --arch <X>  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --step_factors 0.1 0.1 0.1 --train cutmix --cutmix_prob 0.4 --job_id <JobID>
```
For WideResNet-28-10:

```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/patchup/soft/ --labels_per_class 5000 --valid_labels_per_class 500 --arch wrn28_10  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --step_factors 0.1 0.1 --train cutmix --cutmix_prob 0.4 --job_id <JobID>
```
<br/>
<hr/>
<br/>

## cutout 

### CIFAR-10

For PreActResent models you can run the following command.

Note: X is either preactresnet18 or preactresnet34 and at the end JobID is your job id. 
```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/manifold/ --labels_per_class 5000 --valid_labels_per_class 500 --arch <X>  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --step_factors 0.1 0.1 0.1 --train cutout --cutout 16 --job_id <JobID>
```
For WideResNet-28-10:

```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/patchup/soft/ --labels_per_class 5000 --valid_labels_per_class 500 --arch wrn28_10  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --step_factors 0.1 0.1 --train cutout --cutout 8 --job_id <JobID>
```

Note: For running the cutout experiment on SVHN, you should set --cutout 20
<br/>
<hr/>
<br/>

## Mixup 

### CIFAR-10

For PreActResent models you can run the following command.

Note: X is either preactresnet18 or preactresnet34 and at the end JobID is your job id. 
```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/manifold/ --labels_per_class 5000 --valid_labels_per_class 500 --arch <X>  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --step_factors 0.1 0.1 0.1 --train mixup --alpha 1.0 --job_id <JobID>
```
For WideResNet-28-10:

```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/patchup/soft/ --labels_per_class 5000 --valid_labels_per_class 500 --arch wrn28_10  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --step_factors 0.1 0.1 --train mixup --alpha 1.0 --job_id <JobID>
```
<br/>
<hr/>
<br/>

## DropBlock 

### CIFAR-10

For PreActResent models you can run the following command.

Note: X is either preactresnet18 or preactresnet34 and at the end JobID is your job id. 
```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/manifold/ --labels_per_class 5000 --valid_labels_per_class 500 --arch <X>  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --step_factors 0.1 0.1 0.1 --train dropblock --drop_block 7 --keep_prob 0.9 --drop_block_all True --job_id <JobID>
```
For WideResNet-28-10:

```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/patchup/soft/ --labels_per_class 5000 --valid_labels_per_class 500 --arch wrn28_10  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --step_factors 0.1 0.1 --train dropblock --drop_block 7 --keep_prob 0.9 --drop_block_all True --job_id <JobID>
```
<br/>
<hr/>
<br/>

## Vanilla Model (Model without regularization) 

### CIFAR-10

For PreActResent models you can run the following command.

Note: X is either preactresnet18 or preactresnet34 and at the end JobID is your job id. 
```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/manifold/ --labels_per_class 5000 --valid_labels_per_class 500 --arch <X>  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --step_factors 0.1 0.1 0.1 --train vanilla --job_id <JobID>
```
For WideResNet-28-10:

```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --root_dir ./experiments/patchup/soft/ --labels_per_class 5000 --valid_labels_per_class 500 --arch wrn28_10  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --step_factors 0.1 0.1 --train vanilla --job_id <JobID>
```
<br/>
<hr/>
<br/>

## CIFAR-100

For running experiment on CiFAR-100, you can use above command. But you should change only following parameters:
```
* --dataset cifar100 
* --data_dir ./data/cifar100/
* --labels_per_class 500 
* --valid_labels_per_class 50
```
<br/>
<hr/>
<br/>

### SVHN

For running experiment you can use above command. But you should change only following parameters:
```
* --dataset svhn 
* --data_dir ./data/svhn/
* --labels_per_class 7325 
* --valid_labels_per_class 733

Note: To run the cutout experiment on SVHN, you should  also set --cutout 20
```
<br/>
<hr/>
<br/>

# Experiments on Deformed Images

First, we need to create an affine transformed test set by running the following command:
```
python ./load_data.py --affine_path ./data/test/affine/
```
We create affine transformed test set described in the paper for the CIFAR-100.
After creating the the Deformed Images test set, 
we can run generalization experiment on Deformed Images (affine transformed test set) with same commands to train model with a regularization technique with two more parameters. Following is one command example that is used in Soft PatchUp.
```
python ./main.py --dataset cifar100 --data_dir ./data/cifar100/ --affine_test --affine_path ./data/test/affine/  --root_dir ./experiments/patchup/soft/ --labels_per_class 500 --valid_labels_per_class 50 --arch wrn28_10  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --step_factors 0.1 0.1 --train patchup --alpha 2.0 --patchup_type soft --patchup_block 7 --patchup_prob 1.0 --gamma 0.25 --job_id <JobID>
```
Note: Use the above as a pattern to create a command to run an experiment for evaluating the performance of other approaches in this task.
<br/>
<hr/>
<br/>

# Robustness to Adversarial Examples

in order to see the regularized models' robustness against the FGSM attack, we can use following parameter:
* --fsgm_attack True

The following command runs this experiment on PreActResNet18 in CIFAR-10 with Soft PatchUp and evaluate its robustness against the FGSM attack.
```
python ./main.py --dataset cifar10 --data_dir ./data/cifar10/ --fsgm_attack True --root_dir ./experiments/patchup/soft/ --labels_per_class 5000 --valid_labels_per_class 500 --arch <X>  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --step_factors 0.1 0.1 0.1 --train patchup --alpha 2.0 --patchup_type soft --patchup_block 7 --patchup_prob 1.0 --gamma 0.25 --job_id <JobID>
```
Note: Use above as an example command for running experiment on evaluating other approaches performance.




