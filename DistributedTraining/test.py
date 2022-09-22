import os
import torch
import torch.distributed as dist




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