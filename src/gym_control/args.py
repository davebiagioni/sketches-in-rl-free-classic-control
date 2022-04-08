import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-seeds", 
    default=10,
    type=int,
    help="Number of seeds to use in evaluation"
)
