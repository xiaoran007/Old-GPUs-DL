from benchmark.Bench import Bench
import argparse


def main():
    parser = argparse.ArgumentParser(description="Command line settings.")

    parser.add_argument("-m", "--manual", action="store_true", default=False,
                        help="Enable manual benchmark.")
    parser.add_argument("-a", "--auto", action="store_true", default=False,
                        help="Enable auto benchmark.")
    parser.add_argument("-s", "--size", type=int, required=True,
                        help="Set the CUDA memory size in MB.")
    parser.add_argument("-e", "--epochs", type=int, required=True,
                        help="Set the epochs.")

    args = parser.parse_args()

    if args.auto:
        b = Bench(auto=True)
        b.start()
    elif args.manual:
        b = Bench(auto=False, size=args.size, epochs=args.epochs)
        b.start()


if __name__ == "__main__":
    main()

