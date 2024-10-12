from cgi import print_arguments
import os
import numpy as np
import cv2
import argparse
import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
from OF_3DCNN_train import train
import random
from MOL_model import MOL
from train import train
from utils.metrics import calculate_metrics

def set_random_seed(SEED=2023):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--wdecay', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--ldm_weight', type=float, default=0.5)
    parser.add_argument('--of_weight', type=float, default=0.5)
    parser.add_argument('--mer_weight', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting.')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--pretrained_path', default=None)
    parser.add_argument('--version', default='V1.0.0')
    parser.add_argument('--seed',default=2023)
    parser.add_argument('--dataset',default='CASME2', help='CASME2, SAMM or SMIC.')
    parser.add_argument('--cls',default=5, help='3 or 5. (3 for SMIC only)')
    args = parser.parse_args()

    set_random_seed(args.seed)
    if args.cls == 5:
        ConfusionMatrix = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    else:
        ConfusionMatrix = [[0,0,0],[0,0,0],[0,0,0]]

    if args.dataset == "CASME2" and args.cls== 3:
        LOSO = ['17', '26', '16', '09', '05', '24', '02', '13', '04', '23', '11', '12', '08', '14', '03', '19', '01', '10', '20', '21', '22', '15', '06', '25', '07']

    if args.dataset == "SAMM" and args.cls== 3:
        LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','026','028','030','031','032','033','034','035','036','037']

    if args.dataset == "SMIC" and args.cls== 3:
        LOSO = ['s1','s2','s3','s4','s5','s6','s8','s9','s11','s12','s13','s14','s15','s18','s19','s20']
      
    if args.dataset == "CASME2" and args.cls== 5:
        LOSO = ['01', '17', '26', '16', '09', '05', '24', '02', '13', '04', '23', '11', '12', '08', '14', '03', '19', '10', '20', '21', '22', '15', '06', '25', '07']

    if args.dataset == "SAMM" and args.cls== 5:
        LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','025','026','028','030','031','032','033','034','035','036','037']



    test_log_file = open('logs/' + args.version + '_test_log.txt', 'w')
    train_log_file = open('logs/' + args.version + '_train_log.txt', 'w')
    train_log_file.writelines('----------args----------\n')
    test_log_file.writelines('----------args----------\n')

    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
        train_log_file.writelines('%s: %s\n' % (k, vars(args)[k]))
        test_log_file.writelines('%s: %s\n' % (k, vars(args)[k]))
    train_log_file.writelines('----------args----------\n')
    test_log_file.writelines('----------args----------\n')


    for sub in range(len(LOSO)):
        subject = LOSO[sub]
        test_dataset = torch.load('processed_data/'+args.dataset+'/sub'+subject+'_'+str(args.cls)+'cls_test.pth')

        train_sub_list = LOSO-[subject]
        for train_sub in range(len(train_sub_list)):
            train_subject = train_sub_list[train_sub]
            if train_sub == 0:
                train_dataset = torch.load('processed_data/'+args.dataset+'/sub'+train_subject+'_'+str(args.cls)+'cls_train.pth')
            else:
                train_dataset = train_dataset + torch.load('processed_data/'+args.dataset+'/sub'+train_subject+'_'+str(args.cls)+'cls_train.pth')
    
        model = MOL(args)
        if args.pretrained_path is not None:
            print("loading model.....")
            pretrain_weight = torch.load(args.pretrained_path)
            model.load_state_dict(pretrain_weight,strict = False)

        
            
        train_dataset_size = train_dataset.__len__()
        test_dataset_size = test_dataset.__len__()
        train_log_file.writelines('train_dataset_size:' + str(train_dataset_size))
        train_log_file.writelines('test_dataset_size:'+ str(test_dataset_size))
        print('train_dataset.size:{}'.format(len(train_dataset)))
        train_log_file.writelines('LOSO ' +subject+'\n')
        test_log_file.writelines('LOSO '+subject+'\n')
        final_acc, subject_confusion_matrix=train(args=args, model=model, train_dataset=train_dataset, test_dataset=test_dataset,train_log_file=train_log_file, test_log_file=test_log_file)
        test_log_file.writelines('LOSO '+subject+' best_acc:'+str(final_acc)+'\n')

        for i in range(len(ConfusionMatrix)):
            for j in range(len(ConfusionMatrix[0])):
                ConfusionMatrix[i][j] +=subject_confusion_matrix[i][j]
    result = calculate_metrics(ConfusionMatrix)
    test_log_file.writelines(result)
