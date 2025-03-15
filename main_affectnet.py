import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "5" 
import numpy as np
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import wandb
from src.dataset import NLA_Affecnet
from src.model import NLA_r18
from src.utils import *
from src.resnet import *
from src.loss import * 
from torchsampler import ImbalancedDatasetSampler
from tqdm import tqdm
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="AffecNet", type=str, choices=['rafdb', 'Ferplus', 'AffecNet'], help="experiment dataset")
parser.add_argument('--exp_name', default="NLA_NAW", type=str, choices=['R18', 'r18_L1', 'r18_JSD', 'EAC',  'r18_GA', 'r18_L1_GA', 'EAC_GA', 'r18_JSD_GA'], help="training strategy")
parser.add_argument('--dataset_path', type=str, default='/workspace/dataset/affectnet/', help='raf_dataset_path')
parser.add_argument('--label_path', type=str, default='/workspace/dataset/affectnet/', help='label_path')
parser.add_argument('--imbalanced_path', type=str, default='/workspace/dataset/affectnet/imbalanced.csv')
parser.add_argument('--noise_path', type=str, default='/workspace/dataset/affectnet/noise.csv')
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--feature_embedding', type=int, default=512)
parser.add_argument('--output', default="/workspace/NLA/AAAI", type=str, help="output dir")
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--lr', type=float, default=1e-6, help='lr')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--gpu', type=int, default=0, help='the number of the device')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
parser.add_argument('--save_freq', type=int, default=5, help='save frequency')  
parser.add_argument('--noise', type=bool, default=False, help='learning from noise label')  
parser.add_argument('--noise_name', type=str, default="10_1", help='noise percentage_random seed')  
parser.add_argument('--imbalanced', type=bool, default=False, help='learning from imbalanced label')
parser.add_argument('--imbalanced_name', type=str, default="20_256", help='imbalance factor_random seed')  
parser.add_argument('--seed', type=int, default=11111135) # 11111135
parser.add_argument('--lam_a', type=float, default=0.5)
parser.add_argument('--lam_b', type=float, default=0.5)
parser.add_argument('--lam_c', type=float, default=0.5)
parser.add_argument('--slope', type=float, default=-15)
parser.add_argument('--t_lambda', type=float, default=1)
parser.add_argument('--sch_bool', type=bool, default=True)
parser.add_argument('--mu_x_t', type=float, default=0.5) 
parser.add_argument('--mu_y_t', type=float, default=0.5) 
parser.add_argument('--mu_x_f', type=float, default=0.30) 
parser.add_argument('--mu_y_f', type=float, default=0.20) 
parser.add_argument('--t_std_major', type=float, default=0.85) 
parser.add_argument('--t_std_ratio', type=float, default=2.5) 
parser.add_argument('--f_std_major', type=float, default=0.75) 
parser.add_argument('--f_std_ratio', type=float, default=3) 
args = parser.parse_args()




def train(args, idx, model, train_loader, optimizer, scheduler, device):
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    acc = AccuraryLogger_top2(7)
    model.to(device)
    model.train()    
    # if 'GA' in args.exp_name:
    args.eps = exponential_scheduler(idx, args.epochs, args.slope, args.sch_bool)
        
    for image, labels, image2 in tqdm(train_loader):
        image = image.to(device)
        image2 = image2.to(device)
        labels = labels.to(device)
            
        output  = model(image)
        output2 = model(image2)
        # Loss
        cross_loss = nn.CrossEntropyLoss(reduction='none')(output, labels)
        t_major = args.t_std_major**2 + (args.t_std_major/args.t_std_ratio)**2
        t_minor = -args.t_std_major**2 + (args.t_std_major/args.t_std_ratio)**2
        
        f_major = args.f_std_major**2 + (args.f_std_major/args.f_std_ratio)**2
        f_minor = args.f_std_major**2 - (args.f_std_major/args.f_std_ratio)**2

        loss_naw = NAW(args, mu_x_t=args.mu_x_t, mu_y_t=args.mu_y_t, mu_x_f = args.mu_x_f, mu_y_f = args.mu_y_f, f_minor=f_minor, f_major=f_major, t_minor=t_minor, t_major=t_major, t_lambda=args.t_lambda)(output, labels)
        

        loss_jsd = args.lam_c * jensen_shannon_divergence(output, output2)
        
        loss = (args.lam_a * cross_loss.mean() + args.lam_b * loss_naw.mean()) + loss_jsd
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        iter_cnt += 1
        _, predicts = torch.max(output, 1)
        
        preds = predicts.detach().cpu().numpy() 
        targets = labels.detach().cpu().numpy()
        
        preds_top2 = output.detach().cpu().numpy()
        acc.update(preds, preds_top2, targets)
        
        correct_num = torch.eq(predicts, labels).sum()
        correct_sum += correct_num
        running_loss += loss.item()
    
    scheduler.step()
    running_loss = running_loss / iter_cnt
    acc = correct_sum.float() / float(train_loader.dataset.__len__())
    return acc, running_loss


def test(model, test_loader, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0
        acc = AccuraryLogger_top2(7)

        for image, labels, _ in tqdm(test_loader):
            image = image.to(device)
            labels = labels.to(device)


            outputs = model(image)


            loss = nn.CrossEntropyLoss()(outputs, labels).detach()

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)

            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num

            running_loss += loss.item()
            data_num += outputs.size(0)
            _, predicts = torch.max(outputs, 1)
            preds = predicts.detach().cpu().numpy() 
            targets = labels.detach().cpu().numpy()
            preds_top2 = outputs.detach().cpu().numpy()
            acc.update(preds, preds_top2, targets)
            
        running_loss = running_loss / iter_cnt
        test_acc = correct_sum.float() / float(data_num)
        
    return acc, test_acc, running_loss
        
        
        
def main():    
    setup_seed(args.seed)
    args.con = 0
    args.exp_name = args.exp_name
    args.output = os.path.join(args.output, args.dataset, args.exp_name)
    args.max_acc = 0
    args.max_acc_mean = 0
    args.save_cnt = 0
    createDirectory(args.output)
    hyper_setting = ['true', str(args.mu_x_t) ,str(args.mu_y_t), str(args.t_std_major), str(args.t_std_ratio), 
    'false',str(args.mu_y_t) ,str(args.mu_y_f), str(args.f_std_major), str(args.f_std_ratio), 'original']
    wandb.init(project='project1', name=args.exp_name+'_'+"_".join(hyper_setting))
    wandb_args = {
            "backbone":'ResNet18',
            "batch_size": args.batch_size,
            'num_gpu': 'A6000*1'
            }
    wandb.config.update(wandb_args)   

    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
        ])
    
    
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) 
    
    

    train_dataset = NLA_Affecnet(args, phase='train', transform=train_transforms)
    test_dataset = NLA_Affecnet(args, phase='test', transform=eval_transforms)
    

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        sampler= ImbalancedDatasetSampler(train_dataset),
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.workers,
                                        pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.workers,
                                        pin_memory=True)
    
    model = NLA_r18(args)
    device = torch.device('cuda:{}'.format(args.gpu))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)


    
    for idx, i in enumerate(range(1, args.epochs + 1)):
        train_acc, train_loss = train(args, idx, model, train_loader, optimizer, scheduler, device)
        wandb.log({"Train Acc@1": train_acc,
        "Train Loss" : train_loss,
        }, step=idx)
        
        print(f'Train: [{idx}/{args.epochs + 1}]\t'
        f'Train Acc@1 {train_acc:.4f}\t'
        f'Train Loss {train_loss:.3f}\t')

        
        acc_metric, test_acc, test_loss = test(model, test_loader, device)
        classwise_acc, total_acc, top2_acc = acc_metric.final_score()
        wandb.log({"Test Acc@1": total_acc,
        "Test Loss" : test_loss,
        "Test Mean Acc" : np.mean(classwise_acc),
        "Top 2 Acc" : top2_acc,
        'Neutral' : classwise_acc[6],
        'Happiness' : classwise_acc[3],
        'Sadness' : classwise_acc[4],
        'Surprise' : classwise_acc[0],
        'Fear' : classwise_acc[1],
        'Disgust' : classwise_acc[2],
        'Anger' : classwise_acc[5],
        }, step=idx)
        
        print(f'Test: [{idx}/{args.epochs + 1}]\t'
        f'Test Acc@1 {test_acc:.4f}\t'
        f'Test Mean Acc {np.mean(classwise_acc):.4f}\t'
        f'Test Loss {test_loss:.3f}\t')
        print(f'class acc : {classwise_acc}')
        
        if args.max_acc_mean < np.mean(classwise_acc):
            save_classifier(model, 'best_mean', args)
            args.max_acc_mean = np.mean(classwise_acc)
            
        if args.max_acc < test_acc:
            save_classifier(model, 'best', args)
            args.max_acc = test_acc
            
        # if  idx % 5 == 0:
        #     save_classifier(model, str(idx), args)
            
if __name__ == '__main__':
    main()
