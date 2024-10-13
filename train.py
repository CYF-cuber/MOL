import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
import torch.nn.functional as F
from utils.pupil_distance import get_pupil_distance

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
    L1_loss = nn.L1Loss()
    sequence_loss = nn.MSELoss(reduction="mean")

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
            ldm = ldm.permute(0,2,1)
            ldm_loss = torch.zeros(1).requires_grad_(True)
            OF_loss = torch.zeros(1).requires_grad_(True)
            optimizer.zero_grad()

            pred_mer, pred_flow, pred_ldm = model(video.to(device))
            pred_mer = F.log_softmax(pred_mer, dim=1)

            ME_loss = F.nll_loss(pred_mer, label.to(device))
            #loss = F.nll_loss(output, label.to(device))
            _, pred = torch.max(pred_mer, dim=1)
            print('label:{} \n pred:{}'.format(label, pred))
            #print("ldm_len",len(ldm))
            for index_of in range(len(flow) - 1):
                flow_gt = flow[index_of, :, :, :, :].to(device) #[128,128,23,2]
                for len_ in range(len(pred_flow)):
                    if len_ ==0:
                        flow_pred = pred_flow[0].unsqueeze(-1)
                    else:
                        flow_pred = torch.cat([flow_pred,pred_flow[len_].unsqueeze(-1)],dim=-1)

                flow_pred = flow_pred[index_of,:,:,:].permute(1,2,3,0).to(device)
                loss = sequence_loss(flow_pred.to(torch.float32), flow_gt.to(torch.float32)).cpu()#, gamma=args.gamma, test_mode=False)
                

            for index in range(len(pred_ldm)):
                ldm_frame_gt = ldm[:,index+1 , :]
                ldm_frame_pred = pred_ldm[index]
                batch_loss =torch.zeros(1).requires_grad_(True)
                for batch in range(len(ldm)):
                    pupil_dis = get_pupil_distance(ldm_frame_gt[batch])
                    frame_loss = L1_loss(ldm_frame_gt[batch].to('cpu'),ldm_frame_pred[batch].to('cpu'))/pupil_dis
                    batch_loss = batch_loss + frame_loss
                #print(frame_loss)
                ldm_loss = ldm_loss+batch_loss
            ldm_loss = ldm_loss/(len(pred_ldm)*68)

            print("ME_LOSS:",ME_loss ,"OF_loss:", OF_loss,"ldm_loss:", ldm_loss)
            final_loss =OF_loss.to(torch.float32) * args.of_weight + ldm_loss.to(torch.float32) * args.ldm_weight + ME_loss.to(torch.float32).to('cpu')* args.mer_weight
            final_loss.backward()
            optimizer.step()

            batch_correct_samples = pred.cpu().eq(label).sum()
            correct_samples += pred.cpu().eq(label).sum()
            totalsamples += len(label)
            batch_acc = batch_correct_samples / len(label)
            acc = correct_samples / totalsamples
            print("batch_acc:{}%".format(batch_acc * 100))
            print("acc:{}%".format(acc * 100))


            train_log_file.writelines('-----epoch:{}  steps:{}/{}-----\n'.format(epoch, total_steps, args.num_steps))
            train_log_file.writelines(
                'OF loss:{}\t\tLDM loss:{}\t\tME loss:{}\t\tFinal loss:{}\n'.format(OF_loss.item(),ldm_loss.item(), ME_loss, final_loss))
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
    L1_loss = nn.L1Loss()
    sequence_loss = nn.MSELoss(reduction="mean")
    pred_list = []
    label_list = []

    if args.cls == 5:
        confusion_matrix = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    else:
        confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]


    test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=48)

    with torch.no_grad():
        for i, item in enumerate(test_dataloader):
            print('-----epoch:{}  batch:{}-----'.format(epoch, i))
            OF_loss = torch.zeros(1).requires_grad_(True)
            ldm_loss = torch.zeros(1).requires_grad_(True)
            video, flow, ldm, label = item
            ldm = ldm.permute(0,2,1)
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
            ldm_loss = ldm_loss/(len(pred_ldm)*68)

            correct_samples += cal_corr(label_list, pred_list, confusion_matrix)
            totalsamples += len(label_list)
        acc = correct_samples * 100.0 / totalsamples
        print('-----epoch:{}-----'.format(epoch))
        print("acc:{}%".format(acc))

        test_log_file.writelines('\n-----epoch:{}-----\n'.format(epoch))
        test_log_file.writelines('acc:{}'.format(acc))
        final_loss =OF_loss* args.of_weight + ldm_loss * args.ldm_weight + ME_loss.to('cpu') * args.mer_weight

        test_log_file.writelines('OF loss:{}\t\tLDM loss:{}\t\tME loss:{}\t\tFinal loss:{}\n'.format(OF_loss.item(),ldm_loss, ME_loss, final_loss))
        test_log_file.writelines('confusion_matrix:\n{}'.format(confusion_matrix))

    return acc, confusion_matrix

def cal_corr(label_list, pred_list,confusion_matrix):
    corr = 0
    for (a, b) in zip(label_list, pred_list):
        confusion_matrix[a][b]+=1
        if a == b:
            corr += 1
    return corr
