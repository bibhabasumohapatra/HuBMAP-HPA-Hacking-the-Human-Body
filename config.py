import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, required=False, default=1024)
    parser.add_argument("--fold", type=int, required=False, default=0)
    parser.add_argument("--model", type=str, required=False, default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    parser.add_argument("--loss", type=str, required=False, default="Dice_BCE")
    parser.add_argument("--lr", type=float, required=False, default=8e-5)
    parser.add_argument("--batch_size", type=int, default=4, required=False)
    parser.add_argument("--epochs", type=int, default=21, required=False)
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../input/hubmap-folds/train_unsplit_data.csv",
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

MEAN =  [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
