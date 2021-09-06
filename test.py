import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict
import numpy as np
import yaml
from addict import Dict
import argparse
import cv2
# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid



from dataloaders import lane_detect
from dataloaders import utils
from dataloaders import augmentation as augment

from models.LDnet_network import LDNet_network
from utils import loss as losses
from utils import iou_eval
from utils.metrics import runningScore, averageMeter

from dataloaders.utils import *
#To make reproducible results  
torch.manual_seed(125)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(125)
CONFIG=Dict(yaml.load(open("./config/testing.yaml")))


ap = argparse.ArgumentParser()
ap.add_argument('--backbone_network', required=False,
                help = 'name of backbone network',default='LDNet')#resnet, mobilenet, and LDNet
ap.add_argument('--model_path_resume', required=False,
                help = 'path to a model to resume from',default= './experiments/lane_epoch-17.pth')

args = ap.parse_args()
backbone_network=args.backbone_network
model_path_resume=args.model_path_resume





# Setting parameters
nEpochs =1  # Number of epochs for training 150
resume_epoch = 0  # Default is 0, change if want to resume 0

p = OrderedDict()  # Parameters to include in report
p['trainBatch'] =4  # Training batch size
p['lr'] =1e-7# Learning rate  1e-8 for darknet and 1e-7 shufflenet and mobilenet
p['wd'] = 5e-4  # Weight decay
p['momentum'] = 0.9  # Momentum
p['epoch_size'] =5  # epochs to change learning rate

p['testBatch'] = 1 # Testing batch size
nValInterval = 2  # Run on test set every nTestInterval epochs
snapshot = 2  # Store a model every snapshot epochs


# save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

dataset_path=CONFIG.DATASET




# exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

#make a folder -with name of current time- for every experiment
experiment_id=datetime.now().strftime("%Y-%m-%d_%H_%M")
# save_path = os.path.join(save_dir_root, 'experiments', 'experiment_' + str(experiment_id))
# print(save_path)


# Network definition
net=LDNet_network.build(backbone_network,None,CONFIG)

if CONFIG.USING_GPU:
    torch.cuda.set_device(device=CONFIG.GPU_ID)
    net.cuda()



print("Using a weights from training coarse data from: {}...".format(CONFIG.model_path))
net.load_state_dict(torch.load(CONFIG.model_path)) 



running_metrics_test = runningScore(CONFIG.n_classes)    
modelName = 'LDNet-' + backbone_network + '-lane'
print(modelName)

criterion = losses.cross_entropy2d


if resume_epoch != nEpochs+1:
  

    composed_transforms_tr = transforms.Compose([
        augment.FixedResize((256,256)),
        augment.ToTensor()])
   
    
  

    lane_detect_test = lane_detect.Lane_detect(root=dataset_path,n_classes=CONFIG.n_classes,split='val',transform=composed_transforms_tr)
    testloader = DataLoader(lane_detect_test, batch_size=p['testBatch'], shuffle=True, num_workers=0)
  
    
    loaders=[ testloader ]
    

    num_img_test = len(testloader)
    running_loss_te = 0.0

    previous_miou = -1.0
    global_step = 0
    iev = iou_eval.Eval(CONFIG.n_classes,19)
    # from DET
    test_loss_meter = averageMeter()
   

    best_iou = -100.0
    i = 0
    flag = True

   
    test_rlt_f1=[]
    test_rlt_OA=[]
    test_rlt_iou=[]
    best_f1_till_now=0
    best_OA_till_now=0
    best_IOU_till_now=0

    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        total_miou = 0.0
        net.eval()
   
        for ii, sample_batched in enumerate(testloader):
                inputs, labels = sample_batched['image'], sample_batched['label']

    
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                if CONFIG.USING_GPU:
                    inputs, labels = inputs.cuda(), labels.cuda()

                with torch.no_grad():
                    outputs = net.forward(inputs)

                predictions = torch.max(outputs, 1)[1]
                off=predictions.detach().cpu().numpy()
                pred_color=decode_segmap_cv(off, 'lane')
                loss = criterion(outputs, labels,reduct='sum',weight=None)#sum elementwise_mean
                running_loss_te += loss.item()

                y = torch.ones(labels.size()[2], labels.size()[3]).mul(19).cuda()
                labels=labels.where(labels !=255, y)
                #  iev.addBatch(predictions.unsqueeze(1).data,labels.cpu())
                iev.addBatch(predictions.unsqueeze(1).data,labels.cpu())
                running_metrics_test.update(labels, predictions.unsqueeze(1).data.cpu())
               
                # Print stuff
                if ii % num_img_test == num_img_test - 1:
                    miou=iev.getIoU()[0]
                    running_loss_te = running_loss_te/ num_img_test
                    print('TEST:')
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['testBatch'] + inputs.data.shape[0]))
                    # writer.add_scalar('data/test_loss_epoch', running_loss_te, epoch)
                    # writer.add_scalar('data/test_miour', iev.getIoU()[0], epoch)
                    print('Loss: %f' % running_loss_te)
                    print("Predi iou",iev.getIoU())
                    running_loss_te = 0
                    iev.reset()
        score, class_iou = running_metrics_test.get_scores()

        # for k, v in score.items():
        #             print('score',k, v)
        # #             logger.info('{}: {}'.format(k, v))
        # #             # writer.add_scalar('val_metrics/{}'.format(k), v, i+1)

        # for k, v in class_iou.items():
        #     print('IOU',k,v)
        #             logger.info('{}: {}'.format(k, v))
        #             # writer.add_scalar('val_metrics/cls_{}'.format(k), v, i+1)

                # val_loss_meter.reset()
        running_metrics_test.reset()

                ### add by Sprit
        avg_f1 = score["Mean F1 : \t"]
        OA=score["Overall Acc: \t"]
        IOU=score["Mean IoU : \t"]
        test_rlt_f1.append(avg_f1)
        test_rlt_OA.append(score["Overall Acc: \t"])
        test_rlt_iou.append(score["Mean IoU : \t"])

        if avg_f1 >= best_f1_till_now:
            best_f1_till_now = avg_f1
            correspond_OA = score["Overall Acc: \t"]
            correspond_IOU = score["Mean IoU : \t"]
            best_f1_epoch_till_now = epoch+1
        print("\nBest F1 till now = ", best_f1_till_now)
        print("Correspond OA= ", correspond_OA)
        print("Correspond IOU= ", correspond_IOU)
        print("Best F1 Iter till now= ", best_f1_epoch_till_now)

        if IOU >= best_IOU_till_now:
            best_OA_till_now = IOU
            correspond_f1 = score["Mean F1 : \t"]
            correspond_iou = score["Mean IoU : \t"]
            correspond_acc=score["Overall Acc: \t"]
            best_IOU_epoch_till_now = i+1

            state = {
                "epoch": epoch + 1,
                # "model_state": model.state_dict(),
                # "optimizer_state": optimizer.state_dict(),
                # "scheduler_state": scheduler.state_dict(),
                "best_OA": best_OA_till_now,
            }

        print("Best IOU till now = ", best_IOU_till_now)
        print("Correspond F1= ", correspond_f1)
        print("Correspond OA= ",correspond_acc)
        print("Correspond IOU= ",correspond_iou)
        print("Best IOU Iter till now= ", best_IOU_epoch_till_now)
       
   
