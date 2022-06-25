import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, required=False, default=256)
    parser.add_argument("--batch_size", type=int, required=False, default=2)
    parser.add_argument("--lr", type=float, required=False, default=1e-6)
    parser.add_argument("--valid_batch_size", type=int, default=16, required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--folds", type=int, default=5, required=False)

    return parser.parse_args()
