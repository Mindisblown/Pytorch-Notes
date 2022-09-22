# based on DistributedDataParallel



'''
CUDA_VISIBLE_DEVICES=6,7  python MGPU_MPDDP.py --nodes=2 --ngpus_per_node=2


CUDA_VISIBLE_DEVICES=6,7 python MGPU_MPDDP.py --nodes=2 --ngpus_per_node=2


'''

import argparse

from random import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp

import os

BATCH_SIZE = 16
EPOCHS = 50

parser = argparse.ArgumentParser()

parser.add_argument(
    "--nodes", default=1, type=int, help="number of nodes for distributed training"
)

parser.add_argument(
    "--ngpus_per_node",
    default=2,
    type=int,
    help="number of GPUs per node for distributed training",
)
parser.add_argument(
    "--dist-url",
    default="tcp://127.0.0.1:21306",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--node_rank", default=0, type=int, help="node rank for distributed training"
)



def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /=nprocs
    return rt


def get_device_info(is_manual_set_gpus=False, gpus='2,3,4,5',is_clean_cache=False,is_read_gpu_memory=False):

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
    print("available gpu info: {}, GPU Version:{}".format(availble_gpu,name_gpu))


    if is_read_gpu_memory:
        # get available GPU memory 
        os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
        memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
        os.system('rm tmp.txt')   
        print("all gpu memory: {}".format(memory_gpu))


def init_ddp_config(args,local_rank = 0, ngpus_per_node=1):

    args.global_rank = args.node_rank * ngpus_per_node + local_rank
    #set up distributed decice
    #rank = int(os.environ["RANK"])
    #print("rank:",rank)
    #local_rank = int(os.environ["LOCAL_RANK"])
    #torch.cuda.set_device(rank % torch.cuda.device_count())  # if there are several machines with multiple GPUs
    dist.init_process_group(
        backend="nccl",
        init_method = args.dist_url,
        world_size=args.global_world_size,
        rank = args.global_rank    
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    print(f"[init] == local rank: {local_rank}, global rank: {args.global_rank}, device: {device} ==")
    return args.global_rank, device




def train_worker(local_rank, ngpus_per_node,args):
    
    global_rank, device = init_ddp_config(args, local_rank, ngpus_per_node)
    
    # 1. define network
    net = torchvision.models.resnet18(num_classes=10)
    net = net.to(device=device)
    net = DDP(net,device_ids=[local_rank],output_device=local_rank) # using DataParallel with minminal modification


    # 2. define dataloader
    train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    trainset = torchvision.datasets.CIFAR10( 
        root="./data", 
        train=True, 
        download=False,
        transform=train_transforms
    )

    # single node with four GPUs
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        shuffle = True,  # 在epoch开始处调用sampler.set_epoch(epoch)
    )
    
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # should be False when distributedDataParallel
        num_workers=4,
        pin_memory=True,
        sampler = train_sampler,
    )

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.01*2,  # using two GPUs
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    if global_rank == 0:
        print("            =======  Training  ======= \n")

    # 4. start to train
    net.train()
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0

        # set sampler
        train_loader.sampler.set_epoch(ep)

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)



            loss = criterion(outputs, targets)

            dist.barrier()
            loss = reduce_mean(loss,args.global_world_size)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

            if global_rank==0 and (idx + 1) % 25 == 0 or (idx + 1) == len(train_loader):
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
    if global_rank == 0:
        print("\n            =======  Training Finished  ======= \n")


    
if __name__ == "__main__":

    args = parser.parse_args()
    get_device_info(is_manual_set_gpus=False)

    # init distributedDataParallel
    args.global_world_size = int(args.ngpus_per_node * args.nodes)
    mp.spawn(train_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node,args))
    


