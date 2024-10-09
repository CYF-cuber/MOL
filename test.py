import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
import torch.nn.functional as F
from utils.pupil_distance import get_pupil_distance
from sklearn.metrics import f1_score

def evaluate(args, model, epoch, test_dataset, test_log_file):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    totalsamples = 0
    correct_samples = 0
    epoch = 0
    L1_loss = nn.L1Loss()
    sequence_loss = nn.MSELoss(reduction="mean")
    pred_list = []
    label_list = []

    global confusion_matrix 
    confusion_matrix = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

    test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=48)

    with torch.no_grad():
        for i, item in enumerate(test_dataloader):
            print('-----epoch:{}  batch:{}-----'.format(epoch, i))
            OF_loss = torch.zeros(1).requires_grad_(True)
            ldm_loss = torch.zeros(1).requires_grad_(True)
            video, flow, ldm, label = item
            pred_mer,pred_flow, pred_ldm = model(video.to(device))

            pred_mer = F.log_softmax(pred_mer, dim=1)
            ME_loss = F.nll_loss(pred_mer, label.to(device))
            _, pred = torch.max(pred_mer, dim=1)
            pred_list.extend(pred.cpu().numpy().tolist())
            label_list.extend(label.numpy().tolist())

            print('label:{} \n pred:{}'.format(label, pred))

            for index_of in range(len(flow) - 1):

                flow_gt = flow[index_of, :, :, :, :].to(device) #[128,128,23,2]
                for len_ in range(len(pred_flow)):
                    if len_ ==0:
                        flow_pred = pred_flow[0].unsqueeze(-1)
                    else:
                        flow_pred = torch.cat([flow_pred,pred_flow[len_].unsqueeze(-1)],dim=-1)
                flow_pred = flow_pred[index_of,:,:,:].permute(1,2,3,0).to(device)

                loss = sequence_loss(flow_pred.float(), flow_gt.float()).cpu()
                OF_loss = OF_loss + loss

            for index in range(len(pred_ldm)):
                ldm_frame_gt = ldm[:,index+1 , :]
                ldm_frame_pred = pred_ldm[index]
                batch_loss =torch.zeros(1).requires_grad_(True)
                for batch in range(len(ldm)):
                    pupil_dis = get_pupil_distance(ldm_frame_gt[batch])
                    frame_loss = L1_loss(ldm_frame_gt[batch].to('cpu'),ldm_frame_pred[batch].to('cpu'))/pupil_dis
                    batch_loss = batch_loss + frame_loss
                ldm_loss = ldm_loss+batch_loss

        correct_samples += cal_corr(label_list, pred_list)
        totalsamples += len(label_list)
        acc = correct_samples * 100.0 / totalsamples
        weighted_f1_score = f1_score(label_list, pred_list, average="weighted") * 100
        print('-----epoch:{}-----'.format(epoch))
        print("acc:{}%".format(acc))
        print("weighted f1 score:{}".format(weighted_f1_score))

        test_log_file.writelines('\n-----epoch:{}-----\n'.format(epoch))
        test_log_file.writelines('acc:{}\t\tweighted_f1:{}\n'.format(acc, weighted_f1_score))
        final_loss =OF_loss* args.of_weight + ldm_loss * args.ldm_weight + ME_loss.to('cpu') * args.mer_weight
        #final_loss =torch.log10(OF_loss).to(torch.float32) * args.of_weight + torch.log10(ldm_loss).to(torch.float32) * args.ldm_weight + ME_loss.to(torch.float32).to('cpu')

        test_log_file.writelines('OF loss:{}\t\tLDM loss:{}\t\tME loss:{}\t\tFinal loss:{}\n'.format(OF_loss.item(),ldm_loss, ME_loss, final_loss))
        test_log_file.writelines('confusion_matrix:\n{}\n{}\n{}\n{}\n{}\n'.format(confusion_matrix[0],confusion_matrix[1],confusion_matrix[2],confusion_matrix[3],confusion_matrix[4]))
    
        # print('OF_loss:')
    #print(confusion_matrix)
    return acc, confusion_matrix

def cal_corr(label_list, pred_list):
    corr = 0
    for (a, b) in zip(label_list, pred_list):
        confusion_matrix[a][b]+=1
        if a == b:
            corr += 1
    return corr
