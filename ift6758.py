"""Initialization class"""
import getopt
import sys
from controller import controller


def get_opt(opts):
    params = {}
    for i in opts:
        params[i[0]] = i[1]
    return params


def main(argv):
    opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    params = get_opt(opts)
    controller.generate_files(params['-i'], params['-o'])


if __name__ == "__main__":
    main(sys.argv[1:])
