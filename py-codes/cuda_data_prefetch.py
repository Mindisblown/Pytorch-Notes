import torch
import torchvision

class PrefetchedWrapper():

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    # 每次读取两个cuda流，通过yeild返回，第二次调用从yeild后位置开始
    def prefetched_loader(loader):
        # 创建一个cuda数据流
        # cuda流 - 特定设备的线性执行序列，独立于输入流
        stream = torch.cuda.Stream()
        first = True
        # loader - 经过DataLoader封装
        for next_input, next_target in loader:
            # 小写stream - 给定stream的context管理器
            with torch.cuda.stream(stream):
                next_input = next_input.cuda()
                next_target = next_target.cuda()
                next_input = next_input.float()

            if not first:
                # yield - return返回一个值，并记住返回位置，下次迭代从该位置后开始
                yield input, target
            else:
                first = False

            # 多个cuda流的同步
            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __iter__(self):
        if (self.dataloader.sampler is not None and isinstance(self.dataloader.sampler, torch.utils.data.distributed.DistributedSampler)):
            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)

def main():

    transforms = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root="./", train=True, download=True, transform=transforms)
    val_set = torchvision.datasets.MNIST(root="./", train=False, download=True, transform=transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=2)

    train_fetch = PrefetchedWrapper(train_loader)
    for i, (inputs, targets) in enumerate(train_fetch):
        print("Now {}th Iteration - Inputs Shape {}, Targets is {}".format(i, inputs.shape, targets))


if __name__ == "__main__":
    main()