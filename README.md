# The Feature Representations of Transfer Learning and Gradient-Based Meta-Learning Techniques 

This is the code repository associated with the 2021 NeurIPS meta-learning workshop paper "A Preliminary Study on the Feature Representations of Transfer Learning and Gradient-Based Meta-Learning Techniques". 


The pre-training and meta-learning in few-shot image classifications settings can be run by executing the *main.py* script. Here, we include several examples with the best found hyperparameters (according to random search with a bduget of 30 function evaluations).

```
python -u main.py --problem min --k_test 16 --backbone conv4 --model maml --validate --val_after 2500 --T 4 --k 1 --N 5 --cross_eval --T_test 11 --T_val 11 --meta_batch_size 2 --runs 5 --lr 0.000331 --base_lr 0.011446 --model_spec maml-tuned

python -u main.py --problem min --k_test 16 --backbone resnet10 --model maml --validate --val_after 2500 --T 5 --k 1 --N 5 --cross_eval --T_test 9 --T_val 9 --meta_batch_size 1 --runs 5 --lr 0.000185 --base_lr 0.069478 --model_spec maml-tuned

python -u main.py --problem min --k_test 16 --backbone resnet18 --model maml --validate --val_after 2500 --T 4 --k 1 --N 5 --cross_eval --T_test 13 --T_val 13 --meta_batch_size 2 --runs 5 --lr 0.000050 --base_lr 0.129186 --model_spec maml-tuned
```
