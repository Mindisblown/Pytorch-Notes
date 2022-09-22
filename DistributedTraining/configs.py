import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()

    # device
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')


    # model
    parser.add_argument('--weights', type=str, default='resNet34.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)


    #data
    parser.add_argument('--data-path', type=str,default="./data")

    

    args = parser.parse_args()


    return args
