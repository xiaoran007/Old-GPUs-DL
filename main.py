from benchmark.Bench import Bench
import argparse


def main():
    parser = argparse.ArgumentParser(description="Command line settings.")

    parser.add_argument("-m", "--manual", action="store_true", default=False,
                        help="Enable manual benchmark.")
    parser.add_argument("-a", "--auto", action="store_true", default=False,
                        help="Enable auto benchmark.")
    parser.add_argument("-s", "--size", type=int, required=False, default=1024,
                        help="Set the CUDA memory size in MB.")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=5,
                        help="Set the epochs.")
    parser.add_argument("-mt", "--model", type=str, required=False, default="cnn",
                        help="Set the model type.")
    parser.add_argument("-bs", "--batch", type=int, required=False, default=0,
                        help="Set the batch size.")

    args = parser.parse_args()

    model = "cnn"
    if args.model in ["resnet50", "ResNet-50"]:
        model = "resnet50"
    elif args.model in ["cnn", "CNN"]:
        model = "cnn"

    if args.auto:
        b = Bench(auto=True)
        b.start()
    elif args.manual:
        b = Bench(auto=False, size=args.size, epochs=args.epochs, method=model, batch_size=args.batch)
        b.start()


if __name__ == "__main__":
    main()

