"""
Script to run experiments with a single algorithm of choice.
The design allows for user input and flexibility. 

Command line options are:
------------------------
runs : int, optional
    Number of experiments to perform (using different random seeds)
    (default = 1)
N : int, optional
    Number of classes per task
k : int
    Number of examples in the support sets of tasks
k_test : int
    Number of examples in query sets of meta-validation and meta-test tasks
T : int
    Number of optimization steps to perform on a given task
train_batch_size : int, optional
    Size of minibatches to sample from META-TRAIN tasks (or size of flat minibatches
    when the model requires flat data and batch size > k)
    Default = k (no minibatching, simply use entire set)
test_batch_size : int, optional
    Size of minibatches to sample from META-[VAL/TEST] tasks (or size of flat minibatches
    when the model requires flat data and batch size > k)
    Default = k (no minibatching, simply use entire set)
logfile : str
    File name to write results in (does not have to exist, but the containing dir does)
seed : int, optional
    Random seed to use
cpu : boolean, optional
    Whether to use cpu

Usage:
---------------
python main.py --arg=value --arg2=value2 ...
"""

import argparse
import csv
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import random
import torchmeta
import torchmeta.datasets as datasets
import matplotlib.pyplot as plt

from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm #Progress bars
from networks import SineNetwork, Conv4, BoostedConv4, ConvX, ResNet, LinearNet, TransformSineNetwork
from algorithms.metalearner_lstm import LSTMMetaLearner  
from algorithms.train_from_scratch import TrainFromScratch
from algorithms.finetuning import FineTuning
from algorithms.specialft import SpecialFT
from algorithms.moso import MOSO
from algorithms.turtle import Turtle
from algorithms.reptile import Reptile
from algorithms.mamlft import MAMLFT
from algorithms.maml import MAML
from algorithms.sap import SAP
from algorithms.ownlstm import LSTM
from algorithms.turtle_loss import TurtleLoss 
from algorithms.modules.utils import get_init_score_and_operator
from sine_loader import SineLoader
from image_loader import ImageLoader
from linear_loader import LinearLoader
from misc import BANNER, NAMETAG
from configs import TFS_CONF, FT_CONF, CFT_CONF, LSTM_CONF,\
                    MAML_CONF, MOSO_CONF, TURTLE_CONF, LSTM_CONF2,\
                    REPTILE_CONF, SPFT_CONF
from batch_loader import BatchDataset, cycle, Data

FLAGS = argparse.ArgumentParser()

# Required arguments
FLAGS.add_argument("--problem", choices=["sine", "min", "cub", "linear"], required=True,
                   help="Which problem to address?")

FLAGS.add_argument("--k", type=int, required=True,
                   help="Number examples per task set during meta-validation and meta-testing."+\
                   "Also the number of examples per class in every support set.")

FLAGS.add_argument("--k_train", type=int, default=None,
                   help="Number examples per task set during meta-training."+\
                   "Also the number of examples per class in every support set.")

FLAGS.add_argument("--k_test", type=int, required=True,
                   help="Number examples per class in query set")

FLAGS.add_argument("--model", choices=["tfs", "finetuning", "centroidft", 
                   "lstm", "maml", "moso", "lstm2", "turtle", "reptile", "tloss", "spft", "mamlft", "sap"], required=True,
                   help="Which model to use?")

# Optional arguments
FLAGS.add_argument("--N", type=int, default=None,
                   help="Number of classes (only applicable when doing classification)")   

FLAGS.add_argument("--meta_batch_size", type=int, default=1,
                   help="Number of tasks to compute outer-update")   

FLAGS.add_argument("--val_after", type=int, default=None,
                   help="After how many episodes should we perform meta-validation?")

FLAGS.add_argument("--decouple", type=int, default=None,
                   help="After how many train tasks switch from meta-mode to base-mode?")

FLAGS.add_argument("--lr", type=float, default=None,
                   help="Learning rate for (meta-)optimizer")

FLAGS.add_argument("--cpe", type=float, default=0.5,
                   help="#Times best weights get reconsidered per episode (only for baselines)")

FLAGS.add_argument("--T", type=int, default=None,
                   help="Number of weight updates per training set")

FLAGS.add_argument("--T_val", type=int, default=None,
                   help="Number of weight updates at validation time")

FLAGS.add_argument("--T_test", type=int, default=None,
                   help="Number of weight updates at test time")

FLAGS.add_argument("--history", choices=["none", "grads", "updates"], default="none",
                   help="Historical information to use (only applicable for TURTLE): none/grads/updates")

FLAGS.add_argument("--beta", type=float, default=None,
                   help="Beta value to use (only applies when model=TURTLE)")

FLAGS.add_argument("--train_batch_size", type=int, default=None,
                   help="Size of minibatches for training "+\
                         "only applies for flat batch models")

FLAGS.add_argument("--test_batch_size", type=int, default=None,
                   help="Size of minibatches for testing (default = None) "+\
                   "only applies for flat-batch models")

FLAGS.add_argument("--activation", type=str, choices=["relu", "tanh", "sigmoid"],
                   default=None, help="Activation function to use for TURTLE/MOSO")

FLAGS.add_argument("--runs", type=int, default=30, 
                   help="Number of runs to perform")

FLAGS.add_argument("--devid", type=int, default=None, 
                   help="CUDA device identifier")

FLAGS.add_argument("--second_order", action="store_true", default=False,
                   help="Use second-order gradient information for TURTLE")

FLAGS.add_argument("--batching_eps", action="store_true", default=False,
                   help="Batching from episodic data")

FLAGS.add_argument("--input_type", choices=["raw_grads", "raw_loss_grads", 
                   "proc_grads", "proc_loss_grads", "maml"], default=None, 
                   help="Input type to the network (only for MOSO and TURTLE"+\
                   " choices = raw_grads, raw_loss_grads, proc_grads, proc_loss_grads, maml")

FLAGS.add_argument("--layer_wise", action="store_true", default=False,
                   help="Whether TURTLE should use multiple meta-learner networks: one for every layer in the base-learner")

FLAGS.add_argument("--param_lr", action="store_true", default=False,
                   help="Whether TURTLE should learn a learning rate per parameter")

FLAGS.add_argument("--base_lr", type=float, default=None,
                   help="Inner level learning rate")

FLAGS.add_argument("--train_base_lr", type=float, default=None,
                   help="Inner level learning rate for meta-training")

FLAGS.add_argument("--train_iters", type=int, default=None,
                    help="Number of meta-training iterations")

FLAGS.add_argument("--model_spec", type=str, default=None,
                   help="Store results in file ./results/problem/k<k>test<k_test>/<model_spec>/")

FLAGS.add_argument("--layers", type=str, default=None,
                   help="Neurons per hidden/output layer split by comma (e.g., '10,10,1')")

FLAGS.add_argument("--cross_eval", default=False, action="store_true",
                   help="Evaluate on tasks from different dataset (cub if problem=min, else min)")

FLAGS.add_argument("--backbone", type=str, default=None,
                    help="Backbone to use (format: convX)")

FLAGS.add_argument("--seed", type=int, default=1337,
                   help="Random seed to use")

FLAGS.add_argument("--single_run", action="store_true", default=False,
                   help="Whether the script is run independently of others for paralellization. This only affects the storage technique.")

FLAGS.add_argument("--no_annealing", action="store_true", default=False,
                   help="Whether to not anneal the meta learning rate for reptile")

FLAGS.add_argument("--random", action="store_true", default=False,
                   help="Use random new layer for MAML in eval phase")

FLAGS.add_argument("--cpu", action="store_true",
                   help="Use CPU instead of GPU")

FLAGS.add_argument("--time_input", action="store_true", default=False,
                   help="Add a timestamp as input to TURTLE")                   

FLAGS.add_argument("--validate", action="store_true", default=False,
                   help="Validate performance on meta-validation tasks")


FLAGS.add_argument("--no_freeze", action="store_true", default=False,
                   help="Whether to freeze the weights in the finetuning model of earlier layers")

FLAGS.add_argument("--eval_on_train", action="store_true", default=False,
                    help="Whether to also evaluate performance on training tasks")

FLAGS.add_argument("--test_adam", action="store_true", default=False,
                   help="Optimize weights with Adam, LR = 0.001 at test time.")

FLAGS.add_argument("--test_opt", choices=["adam", "sgd"], default=None,
                   help="Optimizer to use at meta-validation or meta-test time for the finetuning model")

FLAGS.add_argument("--test_lr", type=float, default=None, help="LR to use at meta-val/test time for finetuning")

FLAGS.add_argument("--special", action="store_true", default=False,
                   help="Train MAML on 64 classes")

FLAGS.add_argument("--log_norm", action="store_true", default=None,
                   help="Log grad norms")

FLAGS.add_argument("--log_test_norm", action="store_true", default=None,
                   help="Log grad norms of final model")
                   
FLAGS.add_argument("--var_updates", action="store_true", default=False,
                   help="Use variable number of updates")

FLAGS.add_argument("--gamma", type=float, default=None, help="Gamma value for mamlft")

FLAGS.add_argument("--transform", type=str, choices=["interp", "scale"], default="interp", help="Type of transform integration for SAP")

FLAGS.add_argument("--reg", type=str, choices=["num_params", "l2", "null"], default="num_params", help="Type of regularization method for transforms in SAP")


FLAGS.add_argument("--pretrain", action="store_true", default=False,
                   help="pretrain sap")

RESULT_DIR = "./results/"

def create_dir(dirname):
    """
    Create directory <dirname> if not exists
    """
    
    if not os.path.exists(dirname):
        print(f"[*] Creating directory: {dirname}")
        try:
            os.mkdir(dirname)
        except FileExistsError:
            # Dir created by other parallel process so continue
            pass

def print_conf(conf):
    """Print the given configuration
    
    Parameters
    -----------
    conf : dictionary
        Dictionary filled with (argument names, values) 
    """
    
    print(f"[*] Configuration dump:")
    for k in conf.keys():
        print(f"\t{k} : {conf[k]}")

def set_batch_size(conf, args, arg_str):
    value = getattr(args, arg_str)
    # If value for argument provided, set it in configuration
    if not value is None:
        conf[arg_str] = value
    else:
        try:
            # Else, try to fetch it from the configuration
            setattr(args, arg_str, conf[arg_str]) 
            args.train_batch_size = conf["train_batch_size"]
        except:
            # In last case (nothing provided in arguments or config), 
            # set batch size to N*k
            num = args.k
            if not args.N is None:
                num *= args.N
            setattr(args, arg_str, num)
            conf[arg_str] = num             

def overwrite_conf(conf, args, arg_str):
    # If value provided in arguments, overwrite the config with it
    value = getattr(args, arg_str)
    if not value is None:
        conf[arg_str] = value
    else:
        # Try to fetch argument from config, if it isnt there, then the model
        # doesn't need it
        try:
            setattr(args, arg_str, conf[arg_str])
        except:
            return
        
def setup(args):
    """Process arguments and create configurations
        
    Process the parsed arguments in order to create corerct model configurations
    depending on the specified user-input. Load the standard configuration for a 
    given algorithm first, and overwrite with explicitly provided input by the user.

    Parameters
    ----------
    args : cmd arguments
        Set of parsed arguments from command line
    
    Returns
    ----------
    args : cmd arguments
        The processed command-line arguments
    conf : dictionary
        Dictionary defining the meta-learning algorithm and base-learner
    data_loader
        Data loader object, responsible for loading data
    """
    
    if args.k_train is None:
        args.k_train = args.k

    # Mapping from model names to configurations
    mod_to_conf = {
        "tfs": (TrainFromScratch, TFS_CONF),
        "finetuning": (FineTuning, FT_CONF),
        "centroidft": (FineTuning, CFT_CONF), 
        "lstm": (LSTMMetaLearner, LSTM_CONF),
        "lstm2": (LSTM, LSTM_CONF2),
        "maml": (MAML, MAML_CONF),
        "mamlft": (MAMLFT, MAML_CONF),
        "sap": (SAP, MAML_CONF),
        "moso": (MOSO, MOSO_CONF),
        "turtle": (Turtle, TURTLE_CONF),
        "tloss": (TurtleLoss, TURTLE_CONF), 
        "reptile": (Reptile, REPTILE_CONF),
        "spft": (SpecialFT, SPFT_CONF),
    }

    baselines = {"tfs", "finetuning", "centroidft"}#, "spft"}
    
    # Get model constructor and config for the specified algorithm
    model_constr, conf = mod_to_conf[args.model]

    # Set batch sizes
    set_batch_size(conf, args, "train_batch_size")
    set_batch_size(conf, args, "test_batch_size")
        
    # Set values of T, lr, and input type
    overwrite_conf(conf, args, "T")
    overwrite_conf(conf, args, "lr")
    overwrite_conf(conf, args, "input_type")
    overwrite_conf(conf, args, "beta")
    overwrite_conf(conf, args, "meta_batch_size")
    overwrite_conf(conf, args, "time_input")
    conf["no_annealing"] = args.no_annealing
    conf["test_adam"] = args.test_adam
    conf["var_updates"] = args.var_updates
    conf["gamma"] = args.gamma
    conf["transform"] = args.transform
    conf["reg"] = args.reg
    conf["sine_constr"] = SineNetwork
    conf["pretrain"] = args.pretrain

    if not args.test_opt is None or not args.test_lr is None:
        assert args.model == "finetuning", "test_opt and test_lr arguments only suited for finetuning model"
        conf["test_opt"] = args.test_opt
        conf["test_lr"] = args.test_lr
    
    # Parse the 'layers' argument
    if not args.layers is None:
        try:
            layers = [int(x) for x in args.layers.split(',')]
        except:
            raise ValueError(f"Error while parsing layers argument {args.layers}")
        conf["layers"] = layers
    
    # Make sure argument 'val_after' is specified when 'validate'=True
    if args.validate:
        assert not args.val_after is None,\
                    "Please specify val_after (number of episodes after which to perform validation)"
    
    # If using multi-step maml, perform gradient clipping with -10, +10
    if not conf["T"] is None:
        if conf["T"] > 1 and (args.model=="maml" or args.model=="turtle"):# or args.model=="reptile"):
            conf["grad_clip"] = 10
        elif args.model == "lstm" or args.model == "lstm2":
            conf["grad_clip"] = 0.25 # it does norm clipping
        else:
            conf["grad_clip"] = None
    
    # If MOSO or TURTLE is selected, set the activation function
    if args.activation:
        act_dict = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(), 
            "sigmoid": nn.Sigmoid()
        }
        conf["act"] = act_dict[args.activation]
    
    # Set the number of reconsiderations of best weights during meta-training episodes,
    # and the device to run the algorithms on 
    conf["cpe"] = args.cpe
    conf["dev"] = args.dev
    conf["second_order"] = args.second_order
    conf["history"] = args.history
    conf["layer_wise"] = args.layer_wise
    conf["param_lr"] = args.param_lr
    conf["decouple"] = args.decouple
    conf["batching_eps"] = args.batching_eps
    conf["freeze"] = not args.no_freeze
    conf["special"] = args.special
    conf["random"] = args.random

    if not args.log_norm is None:
        conf["log_norm"] = True

    if args.T_test is None:
        conf["T_test"] = conf["T"]
    else:
        conf["T_test"] = args.T_test
    
    if args.T_val is None:
        conf["T_val"] = conf["T"]
    else:
        conf["T_val"] = args.T_val

    if not args.base_lr is None:
        conf["base_lr"] = args.base_lr

    if not args.train_base_lr is None:
        conf["train_base_lr"] = args.train_base_lr
    else:
        if args.test_lr is not None:
            conf["test_lr"] = args.test_lr
            conf["train_base_lr"] = args.test_lr
        else:
            try:
                conf["train_base_lr"] = conf["base_lr"]
            except:
                pass

    assert not (args.input_type == "maml" and args.history != "none"), "input type 'maml' and history != none are not compatible"
    assert not (conf["T"] == 1 and args.history != "none"), "Historical information cannot be used when T == 1" 

    # Different data set loader to test domain shift robustness
    cross_loader = None
    
    # Pick appropriate base-learner model for the chosen problem [sine/image]
    # and create corresponding data loader obejct
    if args.problem == "linear":
        data_loader = LinearLoader(k=args.k, k_test=args.k_test, seed=args.seed)
        conf["baselearner_fn"] = LinearNet
        conf["baselearner_args"] = {"criterion":nn.MSELoss(), "dev":args.dev}
        conf["generator_args"] = {
            "batch_size": args.train_batch_size, # Only applies for baselines
        }
        train_loader, val_loader, test_loader, cross_loader = data_loader, None, None, None
    elif args.problem == "sine":
        data_loader = SineLoader(k=args.k, k_test=args.k_test, seed=args.seed)
        if not "sap" in args.model:
            conf["baselearner_fn"] = SineNetwork
        else:
            conf["baselearner_fn"] = TransformSineNetwork
        conf["baselearner_args"] = {"criterion":nn.MSELoss(), "dev":args.dev}
        conf["generator_args"] = {
            "batch_size": args.train_batch_size, # Only applies for baselines
            "reset_ptr": True,
        }
        train_loader, val_loader, test_loader, cross_loader = data_loader, None, None, None
    else:
        assert not args.N is None, "Please provide the number of classes N per set"
        

        normalize = False
        # Image problem
        if args.backbone is None:
            if args.model == "centroidft":
                conf["baselearner_fn"] = BoostedConv4
                lowerstr = "Bconv4"
            else:    
                conf["baselearner_fn"] = ConvX
                lowerstr = "conv4"
            img_size = (84,84)
        else:
            lowerstr = args.backbone.lower()    
            args.backbone = lowerstr        
            if "resnet" in lowerstr:
                modelstr = "resnet"
                constr = ResNet
                img_size = (224,224)
            elif "conv" in lowerstr:
                modelstr = "conv"
                constr = ConvX
                img_size = (84,84)
            else:
                raise ValueError("Could not parse the provided backbone argument")
            
            num_blocks = int(lowerstr.split(modelstr)[1])
            print(f"Using backbone: {modelstr}{num_blocks}")
            conf["baselearner_fn"] = constr
            if num_blocks > 4:
                normalize = True

        if normalize:
            transform = Compose([Resize(size=img_size), ToTensor(), 
                                 Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                           np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
            # transform = Compose([Resize(size=img_size), ToTensor(), 
            #                      Normalize(np.array([0.485, 0.456, 0.406]),
            #                                np.array([0.229, 0.224, 0.225]))])
            #mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]
            print("Normalizing using MTL values")
        else:
            transform = Compose([Resize(size=img_size), ToTensor()])

        if args.train_iters is None:
            if args.k >= 5:
                train_iters = 40000
            else:
                train_iters = 60000
        else:
            train_iters = args.train_iters

        eval_iters = 600
        args.eval_iters = 600
        args.train_iters = train_iters

        if "min" in args.problem:
            ds = datasets.MiniImagenet
            cds = datasets.CUB
            dataset_specifier = Data.MIN
        elif "cub" in args.problem:
            ds = datasets.CUB
            cds = datasets.MiniImagenet
            dataset_specifier = Data.CUB

        val_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=False, 
                        meta_val=True, meta_test=False, meta_split="val",
                        transform=transform,
                        target_transform=Compose([Categorical(args.N)]),
                        download=True)
        val_loader = ClassSplitter(val_loader, shuffle=True, num_train_per_class=args.k, num_test_per_class=args.k_test)
        val_loader = BatchMetaDataLoader(val_loader, batch_size=1, num_workers=2, shuffle=True)


        test_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=False, 
                        meta_val=False, meta_test=True, meta_split="test",
                        transform=transform,
                        target_transform=Compose([Categorical(args.N)]),
                        download=True)
        test_loader = ClassSplitter(test_loader, shuffle=True, num_train_per_class=args.k, num_test_per_class=args.k_test)
        test_loader = BatchMetaDataLoader(test_loader, batch_size=1, num_workers=2, shuffle=True)


        cross_loader = None
        if args.cross_eval:
            cross_loader = cds(root="./data/", num_classes_per_task=args.N, meta_train=False, 
                               meta_val=False, meta_test=True, meta_split="test",
                               transform=transform,
                               target_transform=Compose([Categorical(args.N)]),
                               download=True)
            cross_loader = ClassSplitter(cross_loader, shuffle=True, num_train_per_class=args.k, num_test_per_class=args.k_test)
            cross_loader = BatchMetaDataLoader(cross_loader, batch_size=1, num_workers=2, shuffle=True)


        train_class_per_problem = {
            "min": 64,
            "cub": 140
        }

        problem_to_root = {
            "min": "./data/miniimagenet/",
            "cub": "./data/cub/"
        }

        if args.model in baselines:
            if not args.model == "tfs":
                train_classes = train_class_per_problem[args.problem.lower()]
            else:
                train_classes = args.N # TFS does not train, so this enforces the model to have the correct output dim. directly

            train_loader = BatchDataset(root_dir=problem_to_root[args.problem],
                                        dataset_spec=dataset_specifier, transform=transform)
            train_loader = iter(cycle(DataLoader(train_loader, batch_size=conf["train_batch_size"], shuffle=True, num_workers=4)))
            args.batchmode = True
            print("Using custom made BatchDataset")
        else:
            if args.special:
                train_classes = 64
                train_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=True, 
                          meta_val=False, meta_test=False, meta_split="train",
                          transform=transform,
                          target_transform=Compose([Categorical(64)]),
                          download=True)
                train_loader = ClassSplitter(train_loader, shuffle=True, num_train_per_class=args.k_train, num_test_per_class=args.k_test)
                train_loader = iter(cycle(BatchMetaDataLoader(train_loader, batch_size=1, num_workers=4, shuffle=True)))
            else:
                train_classes = args.N
                
                if args.model == "spft":
                    train_classes = train_class_per_problem[args.problem.lower()]
                    train_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=True, 
                          meta_val=False, meta_test=False, meta_split="train",
                          transform=transform,
                          target_transform=None,
                          download=True)
                    
                    print(train_loader.dataset._labels, len(train_loader.dataset._labels))
                    class_map = dict()
                    for cid, label in enumerate(train_loader.dataset._labels):
                        assert label not in class_map, "Duplicate label found"
                        class_map[label] = cid 
                    args.labels = class_map
                else:
                    train_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=True, 
                            meta_val=False, meta_test=False, meta_split="train",
                            transform=transform,
                            target_transform=Compose([Categorical(args.N)]),
                            download=True)
                    if not args.gamma is None:
                        train_loader2 = BatchDataset(root_dir=problem_to_root[args.problem],
                                            dataset_spec=dataset_specifier, transform=transform)

                        train_loader2 = iter(cycle(DataLoader(train_loader2, batch_size=conf["train_batch_size"], shuffle=True, num_workers=4)))
            
                train_loader = ClassSplitter(train_loader, shuffle=True, num_train_per_class=args.k_train, num_test_per_class=args.k_test)
                train_loader = BatchMetaDataLoader(train_loader, batch_size=1, num_workers=4, shuffle=True)
            args.batchmode = False
            
        conf["baselearner_args"] = {
            "train_classes": train_classes,
            "eval_classes": args.N, 
            "criterion": nn.CrossEntropyLoss(),
            "dev":args.dev
        }

        if not args.backbone is None:
            conf["baselearner_args"]["num_blocks"] = num_blocks
        
        args.backbone = lowerstr
        
    # Print the configuration for confirmation
    print_conf(conf)
    

    if args.problem == "linear" or args.problem == "sine":
        episodic = True
        args.batchmode = False
        if args.model in baselines:
            episodic = False
            args.batchmode = True
        
        print(args.train_batch_size)
        args.data_loader = data_loader
        val_loader = data_loader.generator(episodic=episodic, batch_size=args.test_batch_size, mode="val")
        test_loader = data_loader.generator(episodic=episodic, batch_size=args.test_batch_size, mode="test")
        train_loader = data_loader.generator(episodic=episodic, batch_size=args.train_batch_size, mode="train")
        print("train batch size:", args.train_batch_size)
        print("test batch size:", args.test_batch_size)
        args.linear = True
        args.sine = True
        args.eval_iters = 1000
    else:
        args.linear = False
        args.sine = False



    
    args.resdir = RESULT_DIR
    bstr = args.backbone if not args.backbone is None else ""
    # Ensure that ./results directory exists
    create_dir(args.resdir)
    args.resdir += args.problem + '/'
    # Ensure ./results/<problem> exists
    create_dir(args.resdir)
    if args.N:
        args.resdir += 'N' + str(args.N) + 'k' + str(args.k) + "test" + str(args.k_test) + '/' 
    else:
        args.resdir += 'k' + str(args.k) + "test" + str(args.k_test) + '/' 
    # Ensure ./results/<problem>/k<k>test<k_test> exists
    create_dir(args.resdir)
    if args.model_spec is None:
        args.resdir += args.model + '/'
    else:
        args.resdir += args.model_spec + '/'
    # Ensure ./results/<problem>/k<k>test<k_test>/<model>/ exists
    create_dir(args.resdir)

    # If args.single_run is true, we should store the results in a directory runs
    if args.single_run or args.runs < 30:
        args.resdir += f"{bstr}-runs/"
        create_dir(args.resdir)
        args.resdir += f"run{args.seed}-" 


    test_loaders = [test_loader]
    filenames = [args.resdir+f"{args.backbone}-test_scores.csv"]
    loss_filenames = [args.resdir+f"{args.backbone}-test_losses-T{conf['T_test']}.csv"]

    if args.eval_on_train:
        train_classes = args.N

        loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=True, 
                        meta_val=False, meta_test=False, meta_split="train",
                        transform=transform,
                        target_transform=Compose([Categorical(args.N)]),
                        download=True)
        loader = ClassSplitter(loader, shuffle=True, num_train_per_class=args.k_train, num_test_per_class=args.k_test)
        loader = BatchMetaDataLoader(loader, batch_size=1, num_workers=4, shuffle=True)
        test_loaders.append(loader)
        filenames.append(args.resdir+f"{args.backbone}-train_scores.csv")
        loss_filenames.append(args.resdir+f"{args.backbone}-train_losses-T{conf['T_test']}.csv")
    if args.cross_eval:
        test_loaders.append(cross_loader)
        filenames.append(args.resdir+f"{args.backbone}-cross_scores.csv")
        loss_filenames.append(args.resdir+f"{args.backbone}-cross_losses-T{conf['T_test']}.csv")        

    if args.gamma is None or args.model == "sap":
        return args, conf, train_loader, val_loader, test_loaders, [filenames, loss_filenames], model_constr
    else:
        return args, conf, [train_loader, train_loader2], val_loader, test_loaders, [filenames, loss_filenames], model_constr


def validate(model, data_loader, best_score, best_state, conf, args):
    """Perform meta-validation
        
    Create meta-validation data generator obejct, and perform meta-validation.
    Update the best_loss and best_state if the current loss is lower than the
    previous best one. 

    Parameters
    ----------
    model : Algorithm
        The chosen meta-learning model
    data_loader : DataLoader
        Data container which can produce a data generator
    best_score : float
        Best validation performance obtained so far
    best_state : nn.StateDict
        State of the meta-learner which gave rise to best_loss
    args : cmd arguments
        Set of parsed arguments from command line
    
    Returns
    ----------
    best_loss
        Best obtained loss value so far during meta-validation
    best_state
        Best state of the meta-learner so far
    score 
        Performance score on this validation run
    """
    
    print("[*] Validating performance...")
    scores = []
    c = 0
    if not args.sine:
        for epoch in val_loader:
            (train_x, train_y), (test_x, test_y) = epoch['train'], epoch['test']

            acc, loss_history = model.evaluate(train_x = train_x[0], 
                                train_y = train_y[0], 
                                test_x = test_x[0], 
                                test_y = test_y[0])
            scores.append(acc)
            c+=1
            if c == args.eval_iters:
                break
    else:
        loader = args.data_loader.generator(mode="val", batch_size=args.test_batch_size, episodic=True, reset_ptr=True)
        for epoch in loader:
            train_x, train_y, test_x, test_y = epoch
            acc, loss_history = model.evaluate(train_x = train_x, 
                                train_y = train_y, 
                                test_x = test_x, 
                                test_y = test_y)
            scores.append(acc)

    score = np.mean(scores)
    # Compute min/max (using model.operator) of new score and best score 
    tmp_score = model.operator(score, best_score)
    # There was an improvement, so store info
    if tmp_score != best_score and not math.isnan(tmp_score):
        best_score = score
        best_state = model.dump_state()
    print("validation loss:", score)
    return best_score, best_state, score
        
def body(args, conf, train_loader, val_loader, test_loaders, files, model_constr):
    """Create and apply the meta-learning algorithm to the chosen data
    
    Backbone of all experiments. Responsible for:
    1. Creating the user-specified model
    2. Performing meta-training
    3. Performing meta-validation
    4. Performing meta-testing
    5. Logging and writing results to output channels
    
    Parameters
    -----------
    args : arguments
        Parsed command-line arguments
    conf : dictionary
        Configuration dictionary with all model arguments required for construction
    data_loader : DataLoader
        Data loder object which acts as access point to the problem data
    model_const : constructor fn
        Constructor function for the meta-learning algorithm to use
    
    """
        
    # Write learning curve to file "curves<val_after>.csv"    
    curvesfile = args.resdir+f"{args.backbone}-curves"+str(args.val_after)+".csv"
    
    overall_best_score = get_init_score_and_operator(conf["baselearner_args"]["criterion"])[0]
    overall_best_state = None
    print("overall best score:", overall_best_score)
    
    seeds = [random.randint(0, 100000) for _ in range(args.runs)]
    print("Actual seed:", seeds)

    if not args.gamma is None and not args.model == "sap":
        train_loader, train_loader2 = train_loader

    for run in range(args.runs):
        stime = time.time()
        print("\n\n"+"-"*40)
        print(f"[*] Starting run {run}")
        # Set torch seed to ensure same base-learner initialization across techniques
        torch.manual_seed(seeds[run])
        model = model_constr(**conf)
        if args.model == "spft":
            model.setmap(args.labels)

        # num_params = sum([p.numel() for p in model.baselearner.parameters()])
        # print("Number of parameters:", num_params)
        # import sys; sys.exit()

        if model.operator == max:
            logstr = "accuracy"
        else:
            logstr = "loss"
        
        vtime = time.time()
        # Start with validation to ensure non-trainable model get 
        # validated at least once
        if args.validate:
            best_score, best_state = model.init_score, None
            best_score, best_state, score = validate(model, val_loader, 
                                                     best_score, best_state, 
                                                     conf, args)
            print(f"[*] Done validating, cost: {time.time()-vtime} seconds")
            # Stores all validation performances over time (learning curve) 
            learning_curve = [score]
            
        
        if model.trainable:
            dcounter = [1,0] if conf["decouple"] else [0]

            print('\n[*] Training...')
            ttime = time.time()
            for el in dcounter:
                c = 0
                history = []
                if args.sine:
                    train_loader = args.data_loader.generator(episodic=True, batch_size=args.train_batch_size, reset_ptr=True, mode="train")
                
                for eid, epoch in enumerate(train_loader):
                    if not args.gamma is None and not args.model == "sap":
                        joint_x, joint_y = train_loader2.__next__()

                    #task_time = time.time()
                    # Unpack the episode. If the model is non-episodic in nature, test_x and 
                    # test_y will be None
                    if args.batchmode:
                        if args.linear:
                            train_x, train_y, _, _ = epoch
                            model.train(train_x=train_x, train_y=train_y, test_x=None, test_y=None)
                        else:
                            train_x, train_y = epoch
                            model.train(train_x=train_x, train_y=train_y.view(-1), test_x=None, test_y=None)
                    else:
                        if args.linear:
                            (train_x, train_y, test_x, test_y) = epoch
                            model.train(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
                        else:
                            (train_x, train_y), (test_x, test_y) = epoch['train'], epoch['test']
                            # Perform update using selected batch
                            if not args.gamma is None:
                                model.train(train_x=train_x[0], train_y=train_y[0], test_x=test_x[0], test_y=test_y[0], joint_x=joint_x, joint_y=joint_y)
                            else:
                                model.train(train_x=train_x[0], train_y=train_y[0], test_x=test_x[0], test_y=test_y[0])
                    #print(time.time() - task_time, "seconds")
                    #task_time = time.time()

                    if eid % 1000 == 0 and args.linear and not args.sine:
                        model_params = np.array(model._get_params())
                        history.append(model_params)
                        print(model_params)
                        print(np.mean(model.train_losses[-1000:]))
                    
                    if args.linear and eid%10000==0 and not args.sine:
                        hist = np.array(history)
                        x = hist[:,0]
                        y = hist[:,1]
                        plt.figure()
                        plt.xlim((0,6))
                        plt.ylim((0,6))
                        plt.plot(x,y, color='green')
                        plt.scatter(x[-1], y[-1], color='blue', label='End')
                        plt.scatter(x[0], y[0], color='red', label='Start')
                        tasks = [(1,1), (1,2), (2,2), (2,1), (3,4), (4,4), (4,3), (3,3)]
                        plt.scatter([t[0] for t in tasks], [t[1] for t in tasks], color='black', label='optimal') 
                        plt.legend()
                        plt.show()     

                    # Perform meta-validation
                    if args.validate and (eid + 1) % args.val_after == 0 and el != 1:
                        print(f"{time.time() - ttime} seconds for training")
                        ttime = time.time()
                        vtime = time.time()
                        best_score, best_state, score = validate(model, val_loader, 
                                                                best_score, best_state, 
                                                                conf, args)
                        print(f"[*] Done validating, cost: {time.time()-vtime} seconds")
                        # Store validation performance for the learning curve
                        # note that score is more informative than best_score 
                        learning_curve.append(score)
                    c+=1
                    if c == args.train_iters:
                        print("c reached", args.train_iters)
                        break

        if args.linear and not args.sine:
            hist = np.array(history)
            x = hist[:,0]
            y = hist[:,1]
            plt.figure()
            plt.xlim((0,6))
            plt.ylim((0,6))
            plt.plot(x,y, color='green')
            plt.scatter(x[-1], y[-1], color='blue', label='End')
            plt.scatter(x[0], y[0], color='red', label='Start')
            tasks = [(1,1), (1,2), (2,2), (2,1), (3,4), (4,4), (4,3), (3,3)]
            plt.scatter([t[0] for t in tasks], [t[1] for t in tasks], color='black', label='optimal') 
            plt.legend()
            plt.show()       


        if args.validate:
            # Load best found state during meta-validation
            model.load_state(best_state)
            save_path = args.resdir+f"model-{run}.pkl"
            print(f"[*] Writing best model state to {save_path}")
            model.store_file(save_path)
            if args.log_test_norm:
                model.log_test_norm = True

        
        generators = test_loaders
        filenames, loss_filenames = files 
        
        # Set seed and next test seed to ensure test diversity
        set_seed(args.test_seed)
        args.test_seed = random.randint(0,100000)
        if not args.linear or args.sine:
            for idx, (eval_gen, filename) in enumerate(zip(generators, filenames)):
                test_accuracies = []
                print('\n[*] Evaluating test performance...')
                

                loss_info = []
                c = 0
                if args.sine:
                    eval_gen = args.data_loader.generator(episodic=True, batch_size=args.test_batch_size, mode="test", reset_ptr=True)
                for eid, epoch in enumerate(eval_gen):
                    if args.sine:
                        train_x, train_y, test_x, test_y  = epoch
                        acc, loss_history = model.evaluate(
                                train_x = train_x, 
                                train_y = train_y, 
                                test_x = test_x, 
                                test_y = test_y, 
                                val=False #real test! no validation anymore
                        )
                    else:
                        (train_x, train_y), (test_x, test_y)  = epoch['train'], epoch['test'] 
                        acc, loss_history = model.evaluate(
                                train_x = train_x[0], 
                                train_y = train_y[0], 
                                test_x = test_x[0], 
                                test_y = test_y[0], 
                                val=False #real test! no validation anymore
                        )
                    test_accuracies.append(acc)
                    c+=1
                    loss_info.append(loss_history)
                    if not args.sine:
                        if c >= args.eval_iters:
                            break                
                
                # Create files and headers if they do not exist or if we started a new run (and want to overwrite previous results)
                if not os.path.exists(filename) or run == 0:
                    with open(filename, "w+", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["run",f"mean_{logstr}",f"median_{logstr}"])
                # loss file name
                lfname = loss_filenames[idx]
                flat_losses = [item for sublist in loss_info for item in sublist]
                if not os.path.exists(lfname) or run == 0:
                    open_mode = "w+"
                else:
                    open_mode = "a"
                # Write learning curve to file
                with open(lfname, open_mode, newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([str(flat_loss) for flat_loss in flat_losses])

                print(test_accuracies)
                r, mean, median = str(run), str(np.mean(test_accuracies)),\
                                str(np.median(test_accuracies))
                with open(filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([r, mean, median])
                

                # Loss info 
                # linfo = np.array(loss_info)
                # cols = linfo.shape[1]
                # mean_initloss = str(np.mean(linfo[:,0]))
                # median_initloss = str(np.median(linfo[:,0]))
                # mean_multiloss = str(np.mean(np.mean(linfo[:,1:cols-1], axis=1)))
                # median_multiloss = str(np.median(np.mean(linfo[:,1:cols-1], axis=1)))
                # mean_finloss = str(np.mean(linfo[:,cols-1]))
                # median_finloss = str(np.median(linfo[:,cols-1]))


                # loss file name
                # with open(lfname, "a", newline="") as f:
                #     writer = csv.writer(f)
                #     writer.writerow([r, mean_initloss, median_initloss, mean_multiloss, 
                #                      median_multiloss, mean_finloss, median_finloss])

                print(f"Run {run} done, mean {logstr}: {mean}, median {logstr}: {median}")
                print(f"Time used: {time.time() - stime}")
                print("-"*40)
                if args.sine:
                    break
            
            
            if args.validate:
                print(learning_curve)
                # Determine writing mode depending on whether learning curve file already exists
                # and the current run
                if not os.path.exists(curvesfile) or run == 0:
                    open_mode = "w+"
                else:
                    open_mode = "a"
                # Write learning curve to file
                with open(curvesfile, open_mode, newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([str(score) for score in learning_curve])
                
                # Check if the best score is better than the overall best score
                # if so, update best score and state across runs. 
                # It is better if tmp_best != best
                tmp_best_score = model.operator(best_score, overall_best_score)
                if tmp_best_score != overall_best_score and not math.isnan(tmp_best_score):
                    print(f"[*] Updated best model configuration across runs")
                    overall_best_score = best_score
                    overall_best_state = deepcopy(best_state)

            if args.log_test_norm:
                np.save(f"{args.model_spec}-{args.backbone}-losses-{run}.npy", model.test_losses)
                np.save(f"{args.model_spec}-{args.backbone}-norms-{run}.npy", model.test_norms)
                np.save(f"{args.model_spec}-{args.backbone}-perfs-{run}.npy", model.test_perfs)
                np.save(f"{args.model_spec}-{args.backbone}-dists-{run}.npy", model.distances)
                np.save(f"{args.model_spec}-{args.backbone}-angles-{run}.npy", model.angles)
                np.save(f"{args.model_spec}-{args.backbone}-gangles-{run}.npy", model.gangles)
                np.save(f"{args.model_spec}-{args.backbone}-gdistances-{run}.npy", model.gdistances)


    # At the end of all runs, write the best found configuration to file
    if args.validate:            
        save_path = args.resdir+f"model.pkl"
        print(f"[*] Writing best model state to {save_path}")
        model.load_state(overall_best_state)
        model.store_file(save_path)

def set_seed(seed):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if __name__ == "__main__":
    # Parse command line arguments
    args, unparsed = FLAGS.parse_known_args()

    # If there is still some unparsed argument, raise error
    if len(unparsed) != 0:
        raise ValueError(f"Argument {unparsed} not recognized")
    
    # Set device to cpu if --cpu was specified
    if args.cpu:
        args.dev="cpu"
    
    # If cpu argument wasn't given, check access to CUDA GPU
    # defualt device is cuda:1, if that raises an exception
    # cuda:0 is used
    if not args.cpu:
        print("Current device:", torch.cuda.current_device())
        print("Available devices:", torch.cuda.device_count())
        if not args.devid is None:
            torch.cuda.set_device(args.devid)
            args.dev = f"cuda:{args.devid}"
            print("Using cuda device: ", args.dev)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU unavailable.")
            try:
                torch.cuda.set_device(1)
                args.dev="cuda:1"
            except:
                torch.cuda.set_device(0)
                args.dev="cuda:0"

    # Let there be reproducibility!
    set_seed(args.seed)
    print("Chosen seed:", args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.test_seed = random.randint(0,100000)

    # Let there be recognizability!
    print(BANNER)
    print(NAMETAG)

    # Let there be structure!
    pargs, conf, train_loader, val_loader, test_loaders, files, model_constr = setup(args)


    # Let there be beauty!
    body(pargs, conf, train_loader, val_loader, test_loaders, files, model_constr)