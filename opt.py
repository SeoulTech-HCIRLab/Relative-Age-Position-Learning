import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default="test", choices=['train', 'test'])
    parser.add_argument('--data_name', default='agedb',
                        choices=['agedb', 'cacd', 'afad'],
                        type=str, help='dataset option')
    parser.add_argument('--data_path', type=str, help='path to dataset images')
    parser.add_argument('--model_path', type=str, default='./trained_checkpoints/agedb.pt',
                        help='path to trained_checkpoints or to save newly trained model')
    parser.add_argument('--epoch', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--batch', default=64, type=int, help="batch size")
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float)
    parser.add_argument('--seed', '--random_seed', default=123, type=int, help='set a random seed')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    return args
