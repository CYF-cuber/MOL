import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
import torch.nn.functional as F
from utils.pupil_distance import get_pupil_distance
from dataset import CLIP_LENGTH, LANDMARK_NUM

def train(args, model, train_dataset, test_dataset=None, train_log_file=None, test_log_file=None):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW([{"params":filter(lambda p: p.requires_grad, model.parameters())}], lr=args.lr, weight_decay=args.wdecay)
    train_dataloader = DataLoader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    total_steps = 0
    keep_training = True
    epoch = 0
    final_acc = 0
    L1_loss = nn.L1Loss(reduction='sum')
    L2_loss = nn.MSELoss()

    while keep_training:
        #torch.cuda.empty_cache()
        epoch += 1
        args.test_mode = False
        totalsamples = 0
        correct_samples = 0
        acc = 0

        for i, item in enumerate(train_dataloader):
            print('-----epoch:{}  steps:{}/{}-----'.format(epoch, total_steps, args.num_steps))
            video, flow, ldm, label = item
            flow = flow.permute(0,1,3,4,2)
            ldm_loss = torch.zeros(1).requires_grad_(True)
            OF_loss = torch.zeros(1).requires_grad_(True)
            optimizer.zero_grad()
            pred_mer, pred_flow, pred_ldm = model(video.to(device))
            pred_mer = F.log_softmax(pred_mer, dim=1)
            ME_loss = F.nll_loss(pred_mer.to(device), label.to(device))
            _, pred = torch.max(pred_mer, dim=1)
            print('label:{} \n pred:{}'.format(label, pred))
            flow_loss = (L2_loss(pred_flow.to(torch.float32).cpu(), flow.to(torch.float32).cpu())/(CLIP_LENGTH-1))
            for index in range(CLIP_LENGTH - 1):
                ldm_frame_gt = ldm[:,:,index+1]
                ldm_frame_pred = pred_ldm[:,:,index]
                batch_loss =torch.zeros(1).requires_grad_(True)
                for batch in range(len(ldm)):
                    pupil_dis = get_pupil_distance(ldm_frame_gt[batch])
                    frame_loss = L1_loss(ldm_frame_gt[batch].to('cpu'),ldm_frame_pred[batch].to('cpu'))/pupil_dis
                    batch_loss = batch_loss + frame_loss
                #print(frame_loss)
                ldm_loss = ldm_loss+batch_loss
            ldm_loss = ldm_loss/(len(pred_ldm)*LANDMARK_NUM)

            print("ME_LOSS:",ME_loss ,"OF_loss:", flow_loss,"ldm_loss:", ldm_loss)
            final_loss =flow_loss.to(torch.float32) * args.of_weight + ldm_loss.to(torch.float32) * args.ldm_weight + ME_loss.to(torch.float32).to('cpu')* args.mer_weight
            
            final_loss.backward()
            optimizer.step()

            batch_correct_samples = pred.cpu().eq(label).sum()
            correct_samples += pred.cpu().eq(label).sum()
            totalsamples += len(label)
            batch_acc = batch_correct_samples / len(label)
            acc = correct_samples / totalsamples
            print("batch_acc:{}%".format(batch_acc * 100))
            print("acc:{}%".format(acc * 100))
            print('total_loss:{}'.format(final_loss))
            train_log_file.writelines('-----epoch:{}  steps:{}/{}-----\n'.format(epoch, total_steps, args.num_steps))
            train_log_file.writelines(
                'Flow loss:{}\t\tLDM loss:{}\t\tME loss:{}\t\tFinal loss:{}\n'.format(flow_loss,ldm_loss, ME_loss, final_loss))
            train_log_file.writelines('batch acc:{}\t\tacc:{}\n'.format(batch_acc * 100, acc * 100))
            total_steps += 1

            if total_steps > args.num_steps:
                keep_training = False
                break
        
        print("epoch average acc:{}%".format(acc * 100))
        print('=========================')
        train_log_file.writelines('epoch average acc:{}%\n'.format(acc * 100))
        train_log_file.writelines('=========================\n')
        acc, cm = evaluate(args, model, epoch=epoch, test_dataset=test_dataset, test_log_file=test_log_file)
        if acc > final_acc:
            torch.save(model.state_dict(), args.save_path)
            final_acc = acc
    return final_acc, cm


def evaluate(args, model, epoch, test_dataset, test_log_file):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    totalsamples = 0
    correct_samples = 0
    L1_loss = nn.L1Loss(reduction="sum")
    L2_loss = nn.MSELoss()

    if args.cls == 5:
        confusion_matrix = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    else:
        confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]


    test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=args.batch_size)

    with torch.no_grad():
        for i, item in enumerate(test_dataloader):
            print('-----epoch:{}  batch:{}-----'.format(epoch, i))
            OF_loss = torch.zeros(1).requires_grad_(True)
            ldm_loss = torch.zeros(1).requires_grad_(True)
            video, flow, ldm, label = item
            flow = flow.permute(0,1,3,4,2)
            pred_mer,pred_flow, pred_ldm = model(video.to(device))
            pred_mer = F.log_softmax(pred_mer, dim=1)
            ME_loss = F.nll_loss(pred_mer, label.to(device))
            _, pred = torch.max(pred_mer, dim=1)
            pred_list = pred.cpu().numpy().tolist()
            label_list = label.numpy().tolist()
            print('label:{} \n pred:{}'.format(label, pred))
            flow_loss = (L2_loss(pred_flow.to(torch.float32).cpu(), flow.to(torch.float32).cpu())/(CLIP_LENGTH-1))
            for index in range(CLIP_LENGTH - 1):
                ldm_frame_gt = ldm[:,:,index+1]
                ldm_frame_pred = pred_ldm[:,:,index]
                batch_loss =torch.zeros(1).requires_grad_(True)
                for batch in range(len(ldm)):
                    pupil_dis = get_pupil_distance(ldm_frame_gt[batch])
                    frame_loss = L1_loss(ldm_frame_gt[batch].to('cpu'),ldm_frame_pred[batch].to('cpu'))/pupil_dis
                    batch_loss = batch_loss + frame_loss
                #print(frame_loss)
                ldm_loss = ldm_loss+batch_loss
            ldm_loss = ldm_loss/(len(pred_ldm)*LANDMARK_NUM)
            correct_sample, confusion_matrix = cal_corr(label_list, pred_list, confusion_matrix)
            #print(correct_sample, confusion_matrix)
            correct_samples += correct_sample
            totalsamples += len(label_list)
        acc = correct_samples * 100.0 / totalsamples
        print('-----epoch:{}-----'.format(epoch))
        print("acc:{}%".format(acc))

        test_log_file.writelines('\n-----epoch:{}-----\n'.format(epoch))
        test_log_file.writelines('acc:{}'.format(acc))
        final_loss =flow_loss* args.of_weight + ldm_loss * args.ldm_weight + ME_loss.to('cpu') * args.mer_weight
        print('total_loss:{}'.format(final_loss))

        test_log_file.writelines('Flow loss:{}\t\tLDM loss:{}\t\tME loss:{}\t\tFinal loss:{}\n'.format(flow_loss,ldm_loss, ME_loss, final_loss))
        test_log_file.writelines('confusion_matrix:\n{}'.format(confusion_matrix))

    return acc, confusion_matrix

def cal_corr(label_list, pred_list,confusion_matrix):
    corr = 0
    for (a, b) in zip(label_list, pred_list):
        confusion_matrix[a][b]+=1
        if a == b:
            corr += 1
    return corr,confusion_matrix
