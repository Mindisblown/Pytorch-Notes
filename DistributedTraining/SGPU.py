# 单机单卡


'''
CUDA_VISIBLE_DEVICES=1 python SGPU.py \
    --save-model \
    --save-path='./weights' \
    --batch-size=128 --epoch=5  --num-classes=10




CUDA_VISIBLE_DEVICES=1 python SGPU.py \
    --save-model \
    --save-path='./weights' \
    --is-pretrained \
    --checkpoint='./weights/SGPU-1.pth' \
    --batch-size=128 --epoch=5  --num-classes=10

CUDA_VISIBLE_DEVICES=1 python SGPU.py \
    --save-model \
    --save-path='./weights' \
    --is-pretrained \
    --checkpoint='./weights/MGPU-DP-1.pth' \
    --batch-size=128 --epoch=5  --num-classes=10


'''
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--gpus", type=str, default='0,1,2,3')

parser.add_argument('--save-model', action="store_true",  help='is it to save model')
parser.add_argument("--save-path", type=str, default='./weights',
                    help='the path to save model')

parser.add_argument("--is-pretrained", action='store_true')
parser.add_argument("--checkpoint", type=str, default="./weights/1.pth" )


parser.add_argument("--batch-size", type=int, default=128)

parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--num-classes', type=int, default=5)

    
args = parser.parse_args()
print(args)

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs




def get_device_info(is_manual_set_gpus=False, gpus= args.gpus, is_clean_cache=False):

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
    # 1. define network
    # device = 'cuda' (default: master_gpu = 0), also can use device='cuda:0', 'cuda:1"...
    # args.device
    device = "cuda"
    net = torchvision.models.resnet18(num_classes=args.num_classes)

    if args.is_pretrained:
        if not os.path.isfile(args.checkpoint):
            print(" -- please check model path ---")
        else:
            pretrained_state_dict = torch.load(args.checkpoint)
        
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrained_state_dict.items():
                key = k[7:] if k.startswith('module.') else k
                new_state_dict[key] = v
            net.load_state_dict(new_state_dict)
            
           # net.load_state_dict(pretrained_state_dict)

    net = net.to(device=device)


    # 2. define dataloader
    train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
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
    
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

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
    net.train()  # bn, dropout  train eval(inference)
    # net.eval()

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
        if args.save_model:
            if not os.path.isdir(args.save_path):
                os.mkdir(args.save_path)

            model_state_save_path = os.path.join(args.save_path, "SGPU-{}.pth".format(ep))
            torch.save(net.state_dict(),model_state_save_path)
            print(" === model is saved in {}  ===".format(model_state_save_path))

    print("\n            =======  Training Finished  ======= \n")


if __name__ == "__main__":
    get_device_info(is_manual_set_gpus=True)
    main()

