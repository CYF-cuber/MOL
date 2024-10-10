import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
import torch.nn.functional as F
import argparse
from MOL_model import MOL
from dataset import videoDataset
def test(args, model, test_dataset):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=48)

    with torch.no_grad():
        for i, item in enumerate(test_dataloader):
            video, _,_,_ = item
            pred_mer,pred_flow, pred_ldm = model(video.to(device))
            pred_mer = F.log_softmax(pred_mer, dim=1)
            _, pred_mer_cls = torch.max(pred_mer, dim=1)
    return pred_mer_cls, pred_flow, pred_ldm

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='saved_model/V1.0.pth')
    parser.add_argument('--test_dataset_path', help = 'videoDataset class object')
    parser.add_argument('--output_path',default='output/')
    args = parser.parse_args()

    model = MOL(args)
    weight = torch.load(args.model_path)
    model.load_state_dict(weight)

    test_dataset = torch.load(args.test_dataset_path)
    output = test(args, model, test_dataset)

    torch.save(output, args.output_path + '/output.pth')


