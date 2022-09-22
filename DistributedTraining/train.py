import torch
import torchvision
import argparse
# ddp训练所需模块
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn



def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--local_rank',
                        default=0,
                        type=int
                        )

    args = parser.parse_args()

    return args

def train(train_loader, net, criterion, optimizer, device):
    for idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # 同步多个进程loss
        dist.barrier()
        reduce_loss = reduce_tensor(loss)   


        optimizer.zero_grad()
        # 使用总loss BP
        reduce_loss.backward()
        optimizer.step()

def val():
    pass
    
def main():
    args = parse_args()
    # 外部接受，无需手动指定
    device = torch.device("cuda:{}".format(args.local_rank))
    print("device:{}".format(device))
    torch.cuda.set_device(device)
    # 网络算子优化，适用于固定尺寸
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True

    # 分布式的初始化
    dist.init_process_group(backend="nccl", init_method="env://")
    
    # 数据分发
    trainset = torchvision.datasets.CIFAR10(root="./data",train=True,download=False,transform=None)
    valset = torchvision.datasets.CIFAR10(root="./data",train=False,download=False,transform=None)

    # 从此处开始打乱
    train_datasampler = DistributedSampler(trainset, shuffle=True)
    val_datasampler = DistributedSampler(valset, shuffle=True)
    # 此处shuffle不可为True
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=False, num_workers=6, pin_memory=True, drop_last=True, sampler=train_datasampler)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=2, shuffle=False, num_workers=6, pin_memory=True, drop_last=True, sampler=val_datasampler)

    net = torchvision.models.resnet18(num_classes=10)
    net = net.to(device)
    # 模型分发
    net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=False, device_ids=[args.local_rank], output_device=args.local_rank)

    # 3. define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01*2,momentum=0.9, weight_decay=0.0001, nesterov=True)

    net.train()

    for ep in range(1, 30):

        # 仅在第0个进程打印信息， 无论打印什么调试信息尽量限制在第0个进程， 多个进程打印太乱
        if args.local_rank==0:
            print("Epoch is ", ep)
    
        # set sampler
        train_loader.sampler.set_epoch(ep)
        val_loader.sampler.set_epoch(ep)

        train()

        val()


        
        