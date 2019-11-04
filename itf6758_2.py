"""Initialization class"""
import getopt
import sys
import pandas as pd


def generate_files(input_path, output_path, limit):
    """Generates files with the predictions"""

    data_train = pd.read_csv("{0}Profile/Profile.csv".format(input_path))
    count = 0
    for i in range(len(data_train)):
        count = count + 1
        file = open("${0}${0}.xml".format(data_train[1]), "w")
        file.write("<user id=${0} ".format(data_train[1]))
        file.write("age_group=xx-24 ")
        file.write("gender=female ")
        file.write("extrovert=3.487 ")
        file.write("neurotic=2.732 ")
        file.write("agreeable=3.584 ")
        file.write("open=3.909 ")
        file.write("conscientious=3.446 />")
        file.close()
        if count >= limit:
            break


def get_opt(opts):
    params = {}
    for i in opts:
        params[i[0]] = i[1]
    return params


def main(argv):
    opts, args = getopt.getopt(argv, "hi:o:l:", ["ifile=", "ofile=", "llimit="])
    print(opts)
    params = get_opt(opts)
    generate_files(params['-i'], params['-o'], params.get('-l', 0))


if __name__ == "__main__":
    main(sys.argv[1:])
