from os import path
import gc

import pandas as pd
from tensor2tensor.data_generators import problem, text_problems
from tensor2tensor.utils import registry
from tqdm import tqdm

ISSUES_FILE = path.expanduser('~/github_issues.csv')
LINES_PER_CHUNK = 10000

def get_n_rows():
    df = pd.read_csv(ISSUES_FILE)
    return len(df)

def clean():
    for i in range(3): gc.collect()

@registry.register_problem
class IssueToTitle(text_problems.Text2TextProblem):
    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
    # 5% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 95,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 5,
        }]

    def generate_samples(self, data_dir, tmp_dir, is_training):
        estimated_chunks = get_n_rows() // LINES_PER_CHUNK
        clean()
        datagen = pd.read_csv(ISSUES_FILE, chunksize=LINES_PER_CHUNK)

        for chunk in tqdm(datagen, total=estimated_chunks):
            for title, body in zip(chunk.issue_title, chunk.body):
                yield {"inputs": body, "targets": title}

