import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--square", help="display a square of a given number",
                    type=int, dest="whatever")
args = parser.parse_args()
print args.whatever**2
