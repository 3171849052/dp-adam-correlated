"""Run or reuse one exact JSON trial specification."""
import argparse
from exp3.spec import load_spec
from exp3.train import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    run(load_spec(parser.parse_args().spec))


if __name__ == '__main__':
    main()
