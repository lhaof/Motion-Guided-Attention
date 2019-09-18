import argparse
from datetime import datetime
import time
import glob
import os
from PIL import Image
import numpy as np
import time

# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms 
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Custom includes
from model.mga_model import MGA_Network

# Dataloaders includes
from dataloaders import davis, fbms, visal
from dataloaders import custom_transforms as trforms

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu'            , type=str  , default='0')

    ## Model settings
    parser.add_argument('-model_name'     , type=str  , default= 'MGA')
    parser.add_argument('-num_classes'    , type=int  , default= 1)
    parser.add_argument('-input_size'     , type=int  , default=512)
    parser.add_argument('-output_stride'  , type=int  , default=16)

    ## Visualization settings
    parser.add_argument('-load_path'      , type=str  , default= 'MGA_trained.pth')
    parser.add_argument('-save_dir'       , type=str  , default= './results')

    parser.add_argument('-test_dataset'   , type=str  , default='DAVIS-valset', choices=['DAVIS-valset', 'FBMS', 'ViSal'])
    parser.add_argument('-test_fold'      , type=str  , default='/test')

    return parser.parse_args() 

def softmax_2d(x):
    return torch.exp(x) / torch.sum(torch.sum(torch.exp(x), dim=-1, keepdim=True), dim=-2, keepdim=True)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    net = MGA_Network(nInputChannels=3, n_classes=args.num_classes, os=args.output_stride, 
        img_backbone_type='resnet101', flow_backbone_type='resnet34')

    # load pre-trained weights
    pretrain_weights = torch.load(args.load_path)
    pretrain_keys = list(pretrain_weights.keys())
    pretrain_keys = [key for key in pretrain_keys if not key.endswith('num_batches_tracked')]
    net_keys = list(net.state_dict().keys())

    for key in net_keys:
        # key_ = 'module.' + key 
        key_ = key
        if key_ in pretrain_keys:
            assert(net.state_dict()[key].size() == pretrain_weights[key_].size())
            net.state_dict()[key].copy_(pretrain_weights[key_])
        else:
            print('missing key: ', key_)
    print('loaded pre-trained weights.')

    net.cuda()

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    if args.test_dataset == 'DAVIS-valset':
        test_data = davis.DAVIS(dataset='val', transform=composed_transforms_ts, return_size=True)
    elif args.test_dataset == 'FBMS':
        test_data = fbms.FBMS(dataset='test', transform=composed_transforms_ts, return_size=True)
    elif args.test_dataset == 'ViSal':
        test_data = visal.ViSal(dataset='test', transform=composed_transforms_ts, return_size=True)

    save_dir = args.save_dir + args.test_fold + '-' + args.model_name + '-' + args.test_dataset + '/saliency_map/'
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    num_iter_ts = len(testloader)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net.eval()

    cnt = 0
    accmu_t = 0
    with torch.no_grad():
        for i, sample_batched in enumerate(testloader):
            print("progress {}/{}\n".format(i, num_iter_ts))

            before_t = time.time()
            inputs, labels, label_name, size = sample_batched['image'], sample_batched['label'], sample_batched['label_name'], sample_batched['size']
            flows = sample_batched['flow']
            inputs = Variable(inputs, requires_grad=False)
            inputs = inputs.cuda()
            flows = Variable(flows, requires_grad=False)
            flows = flows.cuda()
            
            prob_pred, flow_map, before_attention_feat, enhanced_feat, after_attention_feat = net(inputs, flows)
            
            prob_pred = torch.nn.Sigmoid()(prob_pred)
            accmu_t += (time.time()-before_t)
            cnt += 1

            prob_pred = (prob_pred - torch.min(prob_pred) + 1e-8) / (torch.max(prob_pred) - torch.min(prob_pred) + 1e-8)

            shape = (size[0, 0], size[0, 1])
            # prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data 
            prob_pred = F.upsample(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data 
            save_data = prob_pred[0]
            save_png = save_data[0].numpy()
            save_png = np.round(save_png*255)
            save_png = save_png.astype(np.uint8)
            save_png = Image.fromarray(save_png)

            save_path = save_dir + label_name[0]
            if not os.path.exists(save_path[:save_path.rfind('/')]):
                os.makedirs(save_path[:save_path.rfind('/')])
            save_png.save(save_path)

if __name__=='__main__':
    args = get_arguments()
    main(args) 