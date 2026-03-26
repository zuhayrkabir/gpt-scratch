import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    max_iters = args.max_iters
    n_layer = args.n_layer