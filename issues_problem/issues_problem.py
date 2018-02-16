from os import path
import gc
import pickle
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd
from tensor2tensor.data_generators import problem, generator_utils, text_encoder
from tensor2tensor.utils import registry
from sklearn.preprocessing import LabelEncoder
import numpy as np

split_token = ' <$issue_body$> '
EOS = text_encoder.EOS_ID
ISSUES_FILE = path.expanduser('~/Code/dl/datasets/github_issues.csv')
VOCAB_FILE = path.join(data_dir, 'vocab_encoder.pkl')

encoder = None
try:
    encoder = pickle.load(open(VOCAB_FILE, 'rb'))
    print('Loaded saved vocab file from', VOCAB_FILE)
except Exception:
    encoder = LabelEncoder()
    print('Couldn\'t find saved vocab file, starting from scratch')

def get_data():
    print('Reading csv...')
    issues = pd.read_csv(ISSUES_FILE)
    return list(issues.issue_title + split_token + issues.body)

def encode(text):
    encoded = encoder.transform(list(text)) + 2 # 0 and 1 are reserved by t2t
    encoded = encoded.tolist()
    encoded.append(EOS)
    return encoded
        
def encode_row(row):
    title, body = row.split(split_token)
    return {"inputs": encode(body), "targets": encode(title)}

@registry.register_problem
class IssueToTitle(problem.Text2TextProblem):
    @property
    def is_character_level(self):
        return True

    @property
    def vocab_name(self):
        return "vocab.gh_issue"

    @property
    def num_shards(self):
        return 10

    def train_encoder(self, data):
        print('Running label encoder...')
        concat_text = ' '.join(data)

        uniq_chars = set()
        for char in concat_text: uniq_chars.add(char)

        del concat_text
        for i in range(3): gc.collect()

        encoder.fit(list(uniq_chars))
        pickle.dump(encoder, open(VOCAB_FILE, 'wb'))

    def generator(self, data_dir, tmp_dir, is_training):
        data = get_data()

        if not hasattr(encoder, 'classes_'):
            train_encoder(data)

        with Pool() as p:
            return tqdm(p.imap(encode_row, data), total=len(data))

