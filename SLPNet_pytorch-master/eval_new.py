import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import math
from argparse import ArgumentParser
from torch.optim import SGD, Adam
from torch.autograd import Variable
from load_data import *
# from module.det_part.detection_head import GaussDistanceLoss
import module.det_part.PostProcessing as postP
import train_config as train_cfg
from model.detection_recognition_pipeline import DetectionRecognitionPipeline, online_distribute_ctc_targets
import pandas as pd
import json


def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


def eval(args, model):
    # ===============================【1】DataSet=================================
    # val dataset loader
    dataset_val = LPDataSet(img_path=train_cfg.test_img_folder_path, txt_path=train_cfg.test_txt_folder_path)
    print("=>Val dataset total images: % d" % dataset_val.__len__())
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size,
                            drop_last=False, shuffle=False, collate_fn=base_lp_collate)
    if args.pretrained is not None:
        print("Load weight from pretrained model ...")
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> Load weight successfully.")
    else:
        print("Please input the model weight!")
        raise ValueError("The args pretrained shouldn't be None!")
    model.eval()
    result_dict = {}
    pred_dict = {}
    if args.mode == 0:  # detection only
        print("The validation mode is: detection only.")
        Tp_all = 0
        Fn_all = 0
        Fp_all = 0
        gauss_all = []
        for step, (images, point_label_list, lpchar_label_list, lpchar_length_list, name_list) in enumerate(loader_val):
                step += 1
                start_time = time.time()
                if args.cuda:
                    images = images.cuda()
                    point_label_list = [label.cuda() for label in point_label_list]
                obj_num_list, scores_tensor, coordinates_tensor = model(images, mode1='det_only', mode2='eval')
                # ======================= test accuracy of val =====================
                start_idx_pred = 0
                print("obj_num_list",obj_num_list)
                for batch_idx, obj_num_pred in enumerate(obj_num_list):
                    print("batch_idx, obj_num_pred",batch_idx, obj_num_pred)
                    if obj_num_pred != 0:
                        # tensor size(obj_num_pred, 8)
                        single_img_coord_preds = coordinates_tensor[start_idx_pred: start_idx_pred + obj_num_pred]
                        pred_dict[str(step)] = single_img_coord_preds.cpu().tolist()

                        result_dict[str(step)] = point_label_list[batch_idx].cpu().tolist()
                        print("single_img_coord_preds",single_img_coord_preds.cpu().tolist())
                        print("point_label_list",point_label_list[batch_idx].cpu().tolist())
                        start_idx_pred = start_idx_pred + obj_num_pred
                        Tp, Fn, Fp, gauss_list = postP.gaussian_eval(single_img_coord_preds, point_label_list[batch_idx])
                        Tp_all += Tp
                        Fn_all += Fn
                        Fp_all += Fp
                        gauss_all.extend(gauss_list)
                    print("batch_idx,Tp_all, Fn_all, Fp_all", batch_idx,Tp_all, Fn_all, Fp_all)

        if Tp_all == 0:
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
            mGauss = 0.0
        else:
            precision = Tp_all * 1.0 / (Tp_all + Fp_all)
            recall = Tp_all * 1.0 / (Tp_all + Fn_all)
            f1_score = (2 * precision * recall) /(precision + recall)
            mGauss = sum(gauss_all) / len(gauss_all)

        # ============================= Total Epoch Print =========================
        print("=> Precision: ", precision)
        print("=> Recall: ", recall)
        print("=> F1-score: ", f1_score)
        print("=> mGauss: %.3f" % (mGauss * 100))
        
        with open('result_target_new.json', 'w') as fp:
            json.dump(result_dict, fp)
        with open('result_pred_new.json', 'w') as fp:
            json.dump(pred_dict, fp)
        
def main(args):
    savedir = os.path.join(cfg.save_parent_folder, str(args.savedir))
    print("The save file path is: " + savedir)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # ============================== Load Model ===========================
    model = DetectionRecognitionPipeline(input_size=train_cfg.INPUT_SIZE,  # (1024, 1024)
                                         det_size=train_cfg.DETECTION_SIZE,   # (512, 512)
                                         reg_size=train_cfg.RECOGNITION_SIZE,   # (144, 48)
                                         class_num=len(CHARS))  # 68
    # =====================================================================
    if args.cuda:
        model = model.cuda()


    print("========== START TRAINING ===========")
    model = eval(args, model)  # Train decoder
    print("========== EVALUATE FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--mode', type=int, default=0)  # 0 or 1
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--steps_interval', type=int, default=50, help='show loss every how many steps')
    parser.add_argument('--savedir', default="ssnetv2_total_2_11")
    # parser.add_argument('--pretrained', default="./weight/weight3_8/model_best.pth")  # "./weight/pretrained_original/model_best.pth"
    # parser.add_argument('--pretrained', default="./weight/SLPNetSave_3/model_best.pth")  # "./weight/pretrained_original/model_best.pth"
    parser.add_argument('--pretrained', default="./weight/SLPNetweight4/model_best.pth")  # "./weight/pretrained_original/model_best.pth"

    main(parser.parse_args())