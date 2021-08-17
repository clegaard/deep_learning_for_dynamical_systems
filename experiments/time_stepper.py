from argparse import ArgumentParser


if __name__ == "main":

    parser = ArgumentParser()

    parser.add_argument("--type", choices=["direct", "resnet", "euler"])

    args = parser.parse_args()
