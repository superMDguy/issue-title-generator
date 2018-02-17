from os import path
import gc
import pickle
import multiprocessing
from multiprocessing import Pool

import pandas as pd
from tensor2tensor.data_generators import problem, text_encoder
from tensor2tensor.utils import registry
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

EOS = text_encoder.EOS_ID
ISSUES_FILE = path.expanduser('~/github_issues.csv')
VOCAB_FILE = path.expanduser('~/Code/dl/datasets/vocab_encoder.pkl')
LINES_PER_CHUNK = 10000

encoder = None

def clean():
    for i in range(10): gc.collect()

def get_n_rows():
    df = pd.read_csv(ISSUES_FILE)
    nrows = len(df)
    del df
    clean()
    return nrows

def train_encoder():
    print('Running label encoder...')
    concat_text = open(ISSUES_FILE).read()

    uniq_chars = set()
    for char in concat_text: uniq_chars.add(char)
    encoder.fit(list(uniq_chars))

    pickle.dump(encoder, open(VOCAB_FILE, 'wb'))

def encode(text):
    encoded = encoder.transform(list(text)) + 2 # 0 and 1 are reserved by t2t
    encoded = encoded.tolist()
    encoded.append(EOS)
    return encoded
        
def encode_chunk(df):
    encoded = []
    for title, body in zip(df.issue_title, df.body):
        encoded.append({"inputs": encode(body), "targets": encode(title)})
    return encoded


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
        return 100

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_CHR

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_CHR

    def generator(self, data_dir, tmp_dir, is_training):
        global encoder

        if path.isfile(VOCAB_FILE):
            encoder = pickle.load(open(VOCAB_FILE, 'rb'))
            print('Loaded saved vocab file from', VOCAB_FILE)
        else:
            print('Couldn\'t find saved vocab file, starting from scratch')
            encoder = LabelEncoder()
            train_encoder()

        clean()

        cpu_count = multiprocessing.cpu_count()
        estimated_chunks = get_n_rows() // LINES_PER_CHUNK
        datagen = pd.read_csv(ISSUES_FILE, chunksize=LINES_PER_CHUNK)

        with Pool(cpu_count) as p:
            processed = p.imap(encode_chunk, tqdm(datagen, total=estimated_chunks))

            for chunk in processed:
                for row in chunk:
                    yield row

