"""Initialization class"""
import getopt
import sys
from controller import controller
import time


def get_opt(opts):
    params = {}
    for i in opts:
        params[i[0]] = i[1]
    return params


def main(argv):
    start = time.time()
    print(start)
    opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    params = get_opt(opts)
    controller.generate_files(params['-i'], params['-o'])
    print(time.time() - start)


if __name__ == "__main__":
    main(sys.argv[1:])
