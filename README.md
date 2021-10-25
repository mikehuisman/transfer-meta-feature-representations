# The Feature Representations of Transfer Learning and Gradient-Based Meta-Learning Techniques 

This is the code repository associated with the 2021 NeurIPS meta-learning workshop paper "A Preliminary Study on the Feature Representations of Transfer Learning and Gradient-Based Meta-Learning Techniques". 

## Meta-learning / pre-training

The pre-training and meta-learning in few-shot image classifications settings can be run by executing the *main.py* script. Here, we include several examples with the best found hyperparameters (according to random search with a bduget of 30 function evaluations).

**MAML**
```
python -u main.py --problem min --k_test 16 --backbone conv4 --model maml --validate --val_after 2500 --T 4 --k 1 --N 5 --cross_eval --T_test 11 --T_val 11 --meta_batch_size 2 --runs 5 --lr 0.000331 --base_lr 0.011446 --model_spec maml-tuned

python -u main.py --problem min --k_test 16 --backbone resnet10 --model maml --validate --val_after 2500 --T 5 --k 1 --N 5 --cross_eval --T_test 9 --T_val 9 --meta_batch_size 1 --runs 5 --lr 0.000185 --base_lr 0.069478 --model_spec maml-tuned

python -u main.py --problem min --k_test 16 --backbone resnet18 --model maml --validate --val_after 2500 --T 4 --k 1 --N 5 --cross_eval --T_test 13 --T_val 13 --meta_batch_size 2 --runs 5 --lr 0.000050 --base_lr 0.129186 --model_spec maml-tuned

python -u main.py --problem cub --k_test 16 --backbone conv4 --model maml --validate --val_after 2500 --T 8 --k 1 --N 5 --cross_eval --T_test 14 --T_val 14 --meta_batch_size 3 --runs 5 --lr 0.000152 --base_lr 0.009475 --model_spec maml-tuned

python -u main.py --problem cub --k_test 16 --backbone resnet10 --model maml --validate --val_after 2500 --T 5 --k 1 --N 5 --cross_eval --T_test 12 --T_val 12 --meta_batch_size 3 --runs 5 --lr 0.000048 --base_lr 0.005875 --model_spec maml-tuned

python -u main.py --problem cub --k_test 16 --backbone resnet18 --model maml --validate --val_after 2500 --T 5 --k 1 --N 5 --cross_eval --T_test 5 --T_val 5 --meta_batch_size 1 --runs 5 --lr 0.000011 --base_lr 0.110121 --model_spec maml-tuned


```

**Finetuning**
```
python -u main.py --problem min --backbone conv4 --model finetuning --T_val 95 --T_test 95 --train_batch_size 32 --test_batch_size 80 --test_lr 0.010449 --k 1 --N 5 --k_test 16 --val_after 2500 --validate --train_iters 60000 --model_spec finetuning-tuned --cross_eval --runs 5 --test_opt adam

python -u main.py --problem min --backbone resnet10 --model finetuning --T_val 97 --T_test 97 --train_batch_size 32 --test_batch_size 80 --test_lr 0.003148 --k 1 --N 5 --k_test 16 --val_after 2500 --validate --train_iters 60000 --model_spec finetuning-tuned --cross_eval --runs 5 --test_opt adam

python -u main.py --problem min --backbone resnet18 --model finetuning --T_val 114 --T_test 114 --train_batch_size 32 --test_batch_size 16 --test_lr 0.001130 --k 1 --N 5 --k_test 16 --val_after 2500 --validate --train_iters 60000 --model_spec finetuning-tuned --cross_eval --runs 5 --test_opt adam

python -u main.py --problem cub --backbone conv4 --model finetuning --T_val 148 --T_test 148 --train_batch_size 8 --test_batch_size 32 --test_lr 0.009957 --k 1 --N 5 --k_test 16 --val_after 2500 --validate --train_iters 60000 --model_spec finetuning-tuned --cross_eval --runs 5 --test_opt adam

python -u main.py --problem cub --backbone resnet10 --model finetuning --T_val 114 --T_test 114 --train_batch_size 32 --test_batch_size 16 --test_lr 0.001130 --k 1 --N 5 --k_test 16 --val_after 2500 --validate --train_iters 60000 --model_spec finetuning-tuned --cross_eval --runs 5 --test_opt adam

python -u main.py --problem cub --backbone resnet18 --model finetuning --T_val 127 --T_test 127 --train_batch_size 64 --test_batch_size 32 --test_lr 0.000485 --k 1 --N 5 --k_test 16 --val_after 2500 --validate --train_iters 60000 --model_spec finetuning-tuned --cross_eval --runs 5 --test_opt adam
```

**Reptile**
Note that we use the same hyperparameters for different backbones, so you can replace conv4 by resnet10 and resnet18
```
python -u main.py --problem min --backbone conv4 --model reptile --k 1 --cross_eval --k_train 15 --N 5 --k_test 16 --T 8 --T_test 50 --model_spec reptile --val_after 10000 --validate --train_batch_size 10 --meta_batch_size 5 --train_iters 500000 --runs 5
```

### Output files and further analysis
These scripts produce output files in the directory ./results/{dataset}/N{num_ways}k{num_shots}test{num_shots_test}/{model_specifier} that can be used for further analysis.
For computing the CKA similarity scores, we use the *similarities.py* script, which can be called with the same arguments as the otehr scripts for the respective techniques. The output is stored in ./similarities/.
The joint classificationa ccuracy can be measured by running *joint.py* with the same arguments as the scripts above. 


### Questions/feedback
For questions or feedback, please send an email to m.huisman@liacs.leidenuniv.nl
