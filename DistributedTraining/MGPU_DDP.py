# 单机多卡 based on DistributedDataParallel



'''

1个node，1个任务，4个GPU
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    MGPU_DDP.py


1 node, 2tasks, 4 GPUs per task (8GPUs)
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    MGPU_DDP.py

CUDA_VISIBLE_DEVICES=6,7 python3 -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=localhost \
    --master_port=22222 \
    MGPU_DDP.py


'''

from random import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


import os

BATCH_SIZE = 256
EPOCHS = 5


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
    
    if is_read_gpu_memory:
        # get available GPU memory 
        os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
        memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
        os.system('rm tmp.txt')

        print("available gpu info: {}, GPU Version:{}".format(availble_gpu,name_gpu))
        print("all gpu memory: {}".format(memory_gpu))
	



def init_ddp_config():
    #set up distributed decice
    world_size = int(os.environ["WORLD_SIZE"]) 

    rank = int(os.environ["RANK"]) 
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(rank % torch.cuda.device_count())  # if there are several machines with multiple GPUs
    dist.init_process_group(backend="nccl")
    
    
    device = torch.device("cuda", local_rank)

    print(f"[init] == world_size: {world_size} local rank: {local_rank}, global rank: {rank}, device: {device} ==")
    return local_rank, device, world_size



def main():
    
    # init ddp
    local_rank,device,world_size = init_ddp_config()
    
    # 1. define network
    net = torchvision.models.resnet18(num_classes=10)
    net = net.to(device=device)
    net = DDP(net,device_ids=[local_rank],output_device=local_rank) # using DataParallel with minminal modification


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

    # single node with four GPUs
    # 在epoch开始处调用sampler.set_epoch(epoch)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        shuffle = True,  
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
    #if local_rank == 0:
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

            '''\
            # 可同步多个进程的loss等值
            dist.barrier()
            reduce_loss = reduce_mean(loss,world_size)
            '''     


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

            if local_rank == 0 and (idx + 1) % 25 == 0 or (idx + 1) == len(train_loader):
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
    if local_rank == 0:
        print("\n            =======  Training Finished  ======= \n")


if __name__ == "__main__":
    get_device_info(is_manual_set_gpus=False)
    main()

