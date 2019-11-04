"""Initialization class"""
import getopt
import sys
import pandas as pd
import os


def get_opt(opts):
    params = {}
    for i in opts:
        params[i[0]] = i[1]
    return params


def main(argv):
    opts, args = getopt.getopt(argv, "hi:p:", ["ifile=", "pfile="])
    params = get_opt(opts)

    profiles = pd.read_csv(f'{params["-i"]}/Profile/Profile.csv')
    # r=root, d=directories, f = files
    xml = {}
    for r, d, f in os.walk(params["-p"]):
        for file in f:
            xml[file.split('.')[0]] = open(os.path.join(r, file), "r").read()
    right = 0
    count = 0
    for k, y in xml.items():
        gender = profiles.loc[profiles['userid'] == k]['gender'].values[0]
        if f'gender="{get_gender(gender)}"' in y:
            right += 1
        if f'gender="null"' in y:
            count += 1
    print(f'Accuracy is {right/(len(xml.items()) - count)}, total profiles is {len(xml.items()) - count}')


def get_gender(gender):
    return 'female' if gender == 1 else 'male'

if __name__ == "__main__":
    main(sys.argv[1:])
