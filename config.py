import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, required=False, default=256)
    parser.add_argument("--fold", type=int, required=False, default=0)
    parser.add_argument("--model", type=str, required=False, default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    parser.add_argument("--loss", type=str, required=False, default="Dice_BCE")
    parser.add_argument("--lr", type=float, required=False, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=2, required=False)
    parser.add_argument("--epochs", type=int, default=5, required=False)
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../input/hubmap-folds/train_256x256_5folds.csv",
        required=False,
    )
    return parser.parse_args()


args = parse_args()

FOLDS = args.fold
MODEL_PATH = args.model
LR = args.lr
TRAIN_BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = 2 * TRAIN_BATCH_SIZE
EPOCHS = args.epochs
LOSS = args.loss
DEVICE = "cuda"

CSV_PATH = args.csv_path

IMAGE_SIZE = args.image_size

MEAN = [0.78036435, 0.75635034, 0.77327976]
STD = [0.24925208, 0.26279064, 0.258655]
