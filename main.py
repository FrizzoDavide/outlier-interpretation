"""
this script can perform outlier interpretation method ATON, COIN, SHAP, LIME, and IntGrad
These methods use feature weight as interpretation

@ Author: Hongzuo Xu
@ email: hongzuo.xu@gmail.com or leogarcia@126.com or xuhongzuo13@nudt.edu.cn
"""

import argparse
import ast
import datetime
import glob
import os
import time
import warnings
from typing import Union

import ipdb
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable

from config import eva_root, get_parser, root
from eval.evaluation_od import evaluation_od, evaluation_od_auc
from model_aton.ATON import ATON
from model_aton.ATON_ablation import ATONabla
from model_aton.ATON_ablation2 import ATONabla2
from model_aton.ATON_ablation3 import ATONabla3
from model_coin.COIN import COIN
from model_iml.LIME import LIME
from model_iml.SHAP import SHAP
from utils import model_utils
from utils.eval_print_utils import print_eval_runs
from utils.utils import generate_path, get_most_recent_file, save_element

# from model_iml.IntGrad import IntGrad


warnings.filterwarnings("ignore")

# ------------------- parser ----------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    default="data/",
    help="dirpath where the data are saved",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="pima",
    help="dataset name",
)
parser.add_argument(
    "--algorithm_name",
    type=str,
    default="aton",
    help="name of the interpretation algorithm to use",
)
parser.add_argument(
    "--gpu", action="store_true", help="If set, use the gpu for training"
)
parser.add_argument("--device_num", type=int, default=0, help="CUDA device number")
parser.add_argument(
    "--file_pos",
    type=int,
    default=0,
    help="file position in a directory for get_most_recent_file",
)
parser.add_argument(
    "--eval",
    action="store_true",
    help="If set, evaluate the interpretation results",
)
parser.add_argument(
    "--train",
    action="store_true",
    help="If set, train the model and save the checkpoint",
)
parser.add_argument(
    "--w2s_ratio",
    type=str,
    default="real_len",
    help="'real-len', 'auto', 'pn', or a ratio.",
)
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--record_name", type=str, default="")
args = parser.parse_args()
parser = get_parser(args.algorithm_name, parser)
args = parser.parse_args()

input_root_list = [os.path.join(args.path,f"{args.dataset_name}.csv")]
w2s_ratio = args.w2s_ratio
od_eval_model = [
    "iforest",
    "copod",
    "hbos",
]  # we obtain ground-truth annotations using three outlier detection methods
runs = args.runs
record_name = args.record_name

# ------------------- record ----------------- #
cwd = os.getcwd()
record_path = generate_path(basepath=cwd, folders=["record", args.algorithm_name])
record_path = os.path.join(record_path, f"{args.algorithm_name}.txt")
checkpoints_dirpath = generate_path(
    basepath=cwd, folders=["checkpoints", args.algorithm_name]
)
feat_weight_dirpath = generate_path(
    basepath=cwd, folders=["lfi", args.algorithm_name]
)

doc = open(record_path, "a")
tab1 = PrettyTable(["parameter", "value"])
tab1.add_row(["@ data", str(input_root_list)])
tab1.add_row(["@ algorithm_name", str(args.algorithm_name)])
tab1.add_row(["@ w2s_ratio", str(w2s_ratio)])
tab1.add_row(["@ runs", str(runs)])
tab1.add_row(["@ od_eval_model", str(od_eval_model)])
tab1.add_row(["@ start_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
for k in list(vars(args).keys()):
    tab1.add_row([k, vars(args)[k]])
print(tab1, file=doc)
print(tab1)
doc.close()
time.sleep(0.2)


def load_model(
    X: np.ndarray, y: np.ndarray, algorithm: str = "aton"
) -> Union[ATON, ATONabla, ATONabla2, ATONabla3, SHAP, LIME, COIN]:
    """
    Function to load the model based on the algorithm name

    Args:
        X (np.ndarray): input data
        y (np.ndarray): labels
        algorithm (str): algorithm name

    Returns:
        model (Union[ATON, ATONabla, ATONabla2, ATONabla3, SHAP, LIME, COIN]): the interpretation model
    """

    if algorithm == "aton":
        model = ATON(
            verbose=False,
            gpu=args.gpu,
            device_num=args.device_num,
            nbrs_num=args.nbrs_num,
            rand_num=args.rand_num,
            alpha1=args.alpha1,
            alpha2=args.alpha2,
            n_epoch=args.n_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            n_linear=args.n_linear,
            margin=args.margin,
            train_ad_model=args.train,
            checkpoint_dirpath=checkpoints_dirpath,
        )

    elif algorithm == "aton_ablation":
        model = ATONabla(
            verbose=False,
            nbrs_num=args.nbrs_num,
            rand_num=args.rand_num,
            n_epoch=args.n_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            n_linear=args.n_linear,
            margin=args.margin,
        )

    elif algorithm == "aton_ablation2":
        model = ATONabla2(
            verbose=False,
            nbrs_num=args.nbrs_num,
            rand_num=args.rand_num,
            n_epoch=args.n_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            margin=args.margin,
        )

    elif algorithm == "aton_ablation3":
        model = ATONabla3(
            verbose=False,
            gpu=True,
            nbrs_num=args.nbrs_num,
            rand_num=args.rand_num,
            n_epoch=args.n_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            n_linear=args.n_linear,
            margin=args.margin,
        )

    elif algorithm == "shap":
        model = SHAP(
            kernel=args.kernel, n_sample=args.n_sample, threshold=args.threshold
        )

    elif algorithm == "lime":
        model = LIME(
            discretize_continuous=args.discretize_continuous,
            discretizer=args.discretizer,
        )

    # elif algorithm == "intgrad":
    #     model = IntGrad(n_steps=args.n_steps, method=args.method)
    #     fea_weight_lst = model.fit(X, y)

    elif algorithm == "coin":
        sgnf_prior = 1
        model = COIN(
            X,
            y,
            args.ratio_nbr,
            AUG=args.AUG,
            MIN_CLUSTER_SIZE=args.MIN_CLUSTER_SIZE,
            MAX_NUM_CLUSTER=args.MAX_NUM_CLUSTER,
            VAL_TIMES=args.VAL_TIMES,
            C_SVM=args.C_SVM,
            THRE_PS=args.THRE_PS,
            DEFK=args.DEFK,
        )
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not implemented")
    return model


def fit_model(
    alg: Union[ATON, ATONabla, ATONabla2, ATONabla3, SHAP, LIME, COIN],
    X: np.ndarray,
    y: np.ndarray,
) -> list:
    """
    Function to fit the model to the data and compute the
    feature attribution list

    Args:
        alg (Union[ATON, ATONabla, ATONabla2, ATONabla3, SHAP, LIME, COIN]): interpretation algorithm to fit
        X (np.ndarray): input data
        y (np.ndarray): labels

    Returns:
        fea_weight_lst (list): list of feature attributions for each anomalous sample
    """

    feat_weight_lst = alg.fit(X, y)

    return feat_weight_lst


def main(path, run_times):

    print("#" * 50)
    print(f"eval: {args.eval}")
    print(f"gpu: {args.gpu}")
    print(f"device_num: {args.device_num}")
    print("#" * 50)
    data_name = path.split("/")[-1].split(".")[0]

    # this is to remove the prefix index number of data set name, so that we can match the annotation file.
    data_name = data_name[3:]

    print("# ------------------ %s ------------------ # " % data_name)

    #TODO: Here instead of using the labels from the dataset we can use the
    # predictions produced by an AD model. We can compute them through the
    # save_labels cli argument in lfi_exp.py

    df = pd.read_csv(path)
    X = df.values[:, :-1]
    y = np.array(df.values[:, -1], dtype=int)

    # get the real length of the ground-truth interpretation if the w2s_ratio is true
    real_len_lst = []
    runs_metric_lst = [[] for k in range(len(od_eval_model))]
    if args.eval and args.w2s_ratio == "real_len":
        gt_lst = []
        for eval_m in od_eval_model:
            folder = eva_root + "data_od_evaluation/"
            gt_path = os.path.join(folder, data_name + "_gt_" + eval_m + ".csv")
            if len(glob.glob(gt_path)) == 0:
                raise FileNotFoundError("no such gt file:" + gt_path)
            gt_str = pd.read_csv(gt_path)["exp_subspace"].values
            gt_lst.append([ast.literal_eval(gtt) for gtt in gt_str])

        for gt in gt_lst:
            real_len_lst.append([len(gtt) for gtt in gt])

    t = 0
    for i in range(run_times):
        print("-" * 50)
        print(f"run {i+1}")
        print("-" * 50)
        time1 = time.time()

        # ------------ run the chosen algorithm to get interpretation (feature weight) ------------- #

        alg = load_model(X=X, y=y, algorithm=args.algorithm_name)

        print("-" * 50)
        print("Running the model")
        print("-" * 50)
        fea_weight_lst = fit_model(alg, X, y)

        fea_weight_arr = np.array(fea_weight_lst)
        avg_scores = np.mean(fea_weight_arr, axis=0)
        scores_ranking = np.argsort(avg_scores)[::-1]

        print("-"*50)
        print(f"Feature ranking in decreasing order of importance: {scores_ranking}")
        print(f"Scores sorted: {avg_scores[scores_ranking]}")
        print("-"*50)

        filename = f"{args.algorithm_name}_avg_lfi_scores"
        save_element(
            element=avg_scores,
            directory_path=feat_weight_dirpath,
            filename=filename,
            filetype="npz",
        )
        print("-" * 50)
        print(
            f"Importance ranking saved at {os.path.join(feat_weight_dirpath,filename)}"
        )
        print("-" * 50)

        # TODO: Understand what is happening in this part.
        # It seems a pre processing step to do on fea_weight_lst to produce another version of the
        # importance scores?
        # ------------------- transfer feature weight to subspace ----------------- #
        subspace_outputs = []
        if args.eval:
            for j in range(len(od_eval_model)):
                if w2s_ratio == "real_len":
                    real_len = real_len_lst[j]
                    subspace = model_utils.get_exp_subspace(
                        fea_weight_lst, w2s_ratio=w2s_ratio, real_exp_len=real_len
                    )
                else:
                    subspace = model_utils.get_exp_subspace(
                        fea_weight_lst, w2s_ratio=w2s_ratio
                    )
                subspace_outputs.append(subspace)

        t = time.time() - time1

        # ---------------------- evaluation -------------------------- #
        if args.eval:
            for mm, eval_model in enumerate(od_eval_model):
                p, j, s = evaluation_od(
                    subspace_outputs[mm], X, y, data_name, eval_model
                )
                auroc, aupr = evaluation_od_auc(
                    fea_weight_lst, X, y, data_name, model_name=eval_model
                )
                metric_lst = [p, j, s, auroc, aupr, t]
                runs_metric_lst[mm].append(metric_lst)
                print(
                    "data: {}, eval_model: {}, {}".format(
                        path.split("/")[-1].split(".")[0], eval_model, metric_lst
                    )
                )

    if args.eval:
        name = path.split("/")[-1].split(".")[0]
        for mm in range(len(od_eval_model)):
            txt = print_eval_runs(
                runs_metric_lst[mm], data_name=name, algo_name=args.algorithm_name
            )
            print(txt)

            doc = open(record_path, "a")
            print(txt, file=doc)
            doc.close()
    else:
        txt = data_name + "," + str(round(t, 2)) + "," + args.algorithm_name
        print(txt)
        doc = open(record_path, "a")
        print(txt, file=doc)
        doc.close()
    return

if __name__ == "__main__":
    for input_root in input_root_list:
        if os.path.isdir(input_root):
            for file_name in sorted(os.listdir(input_root)):
                if file_name.endswith(".csv"):
                    input_path = str(os.path.join(input_root, file_name))
                    main(input_path, runs)

        else:
            input_path = input_root
            main(input_path, runs)
