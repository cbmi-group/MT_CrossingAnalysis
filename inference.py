import cv2
import numpy as np
import argparse
import torch
import os
import glob
import sys
from tqdm import tqdm

root = os.path.abspath('.')
sys.path.append(root)
from model import DGFNet as D_net
from pre_process import *


def segmentation(opts):
    gpus = opts["gpu_list"].split(',')
    gpu_list = []
    for str_id in gpus:
        id = int(str_id)
        gpu_list.append(id)
    os.environ['CUDA_VISIBLE_DEVICE'] = opts["gpu_list"]

    threshold = opts['threshold']
    img_type = opts['img_type']
    img_dir = opts['img_dir']
    save_dir = opts['save_dir']

    if not os.path.exists(img_dir):
        print('Image dictionary is not exist')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    pretrain_model = './model.pth'

    print('==> Image type: {}'.format(img_type))
    # define network
    print("==> Create network")
    model = D_net(1, 1)

    # load trained model
    if os.path.isfile(pretrain_model):
        c_checkpoint = torch.load(pretrain_model)
        model.load_state_dict(c_checkpoint["model_state_dict"])
        print("==> Loading model '{}'.".format(pretrain_model))

    else:
        print("==> No trained model.")
        return 0

    # set model to gpu mode
    print("==> Set to GPU mode")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_list)

    # enable evaluation mode
    with torch.no_grad():
        model.eval()

        img_list = glob.glob(os.path.join(img_dir, '*.tif'))
        print('==> Begin!')

        for img_path in tqdm(img_list):
            img_name = os.path.split(img_path)[-1]

            raw = cv2.imread(img_path, 0)
            h, w = raw.shape
            pre_raw = img_PreProc(raw, img_type)

            x = padding(pre_raw)
            x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).cuda().float()

            _,_,pred = model(x)

            seg = (pred > threshold).float()
            seg = seg * 255
            seg = seg.cpu().numpy()[0][0]
            seg = seg[:h, :w]

            if not os.path.exists(save_dir): os.mkdir(save_dir)
            cv2.imwrite(os.path.join(save_dir, img_name), seg.astype(np.uint8))
        print("==> Finish! ^-^ ")


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('-t', type=float, default=0.6, help='threshold')
    parse.add_argument('--save_dir', type=str, default='./segmentation/MTs-200619', help='save dir')
    parse.add_argument('--img_dir', type=str, default='./images/MTs-200619', help='img dir')
    parse.add_argument('--img_type', type=int, default=16, help='type of the image')
    args = parse.parse_args()

    opts = dict()
    opts["gpu_list"] = "0,1,2,3"
    opts['img_dir'] = args.img_dir
    opts['save_dir'] = args.save_dir
    opts['threshold'] = args.t
    opts['img_type'] = args.img_type

    segmentation(opts)

