import os
import pandas
import yaml


TIME_FORMAT = '%Y:%m:%d %H:%M:%S'

LANGUAGE = {
    code: lang 
    for _, [code, lang] in pandas.read_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', 'data', 'ISO639-1.txt'
        ),
        sep='\t', index_col=None, header=None
    ).iterrows()
}

CREDENTIAL = {}
with open(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data', 'credentials.yaml'), 'r'
) as creds:
    CREDENTIAL = yaml.load(creds, Loader=yaml.Loader)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data', 'dataset')
# DATASET = pandas.read_csv(os.path.join(DATASET_PATH, 'data.csv'), escapechar='\\')

def access(dictionary, address):
    tokens = address.split('/')
    data = dictionary
    for token in tokens:
        try:
            data = data[token]
        except:
            return None
    return data

def version(v):
    return tuple(map(int, (v.split("."))))
