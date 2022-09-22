# 单机单卡
# CUDA_VISIBLE_DEVICES=0 python SingleGPU.py

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#from  configs import get_args_parser

import argparse


from torch.utils.tensorboard import SummaryWriter




def get_args_parser():
    parser = argparse.ArgumentParser()

    # device
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')


    # model
    parser.add_argument('--weights', type=str, default='resNet34.pth',
                        help='initial weights path')
    parser.add_argument("--checkpoints", type=str, default='./weights',
                        help='the path to save model')
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--model-path", type=str, default="./weights/")

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)


    #data
    parser.add_argument('--data-path', type=str,default="./data")
    parser.add_argument('--num-workers', type=int, default=4)
   
    args = parser.parse_args()
    return args


def get_device_info(is_manual_set_gpus=False, gpus='2,3,4,5',is_clean_cache=False):

    if is_clean_cache:
        torch.cuda.empty_cache()
    
    if is_manual_set_gpus:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES",gpus)
    
    # if GPU available
    if not torch.cuda.is_available():
        print("cuda is not available ...")
        return
    
    # get GPU info
    availble_gpu = torch.cuda.device_count()
    name_gpu = torch.cuda.get_device_name()
    
    
    # get available GPU memory 
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
    memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
    os.system('rm tmp.txt')

    print("available gpu info: {}, GPU Version:{}".format(availble_gpu,name_gpu))
    print("all gpu memory: {}".format(memory_gpu))
	

    
def main():

    BATCH_SIZE = 256
    EPOCHS = 10

    args = get_args_parser()
    print("args:",args)
    print('Start TensorBoard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    tb_writer = SummaryWriter()

    if not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)


    # 2. define dataloader
    train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),])

    val_transforms = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),])

    trainset = torchvision.datasets.CIFAR10( 
        root="./data", 
        train=True, 
        download=False,
        transform=train_transforms
    )

    valset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=val_transforms
    )
    
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.number_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.number_workers,
        pin_memory=True,
    )


    # 1. define network
    # device = 'cuda' (default: master_gpu = 0), also can use device='cuda:0', 'cuda:1"...
    device = args.device
    net = torchvision.models.resnet18(num_classes=args.num_classes)
    net = net.to(device=device)

    if args.pretrained:
        if os.args.model_path:
            weights_dict = torch.load(args.model_path, map_location=device)
            load_weights_dict = { k:v for k,v in weights_dict.items()
                                if net.state_dict()[k].numel() == v.numel() }



    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    print("            =======  Training  ======= \n")

    # 4. start to train
    net.train()
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
                print(
                    "   == step: [{:3}/{}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                        idx + 1,
                        len(train_loader),
                        ep,
                        EPOCHS,
                        train_loss / (idx + 1),
                        100.0 * correct / total,
                    )
                )

    print("\n            =======  Training Finished  ======= \n")


if __name__ == "__main__":
    get_device_info(is_manual_set_gpus=True)
    main()

