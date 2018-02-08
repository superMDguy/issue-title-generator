from os import path

from tqdm import tqdm
import pandas as pd
from tensor2tensor.data_generators import problem, generator_utils, text_encoder
from tensor2tensor.utils import registry

EOS = text_encoder.EOS_ID

split_token = ' <$issue_body$> '


def issue_title_generator():
    print('Reading csv...')
    issues = pd.read_csv(path.expanduser('~/Code/dl/datasets/github_issues.csv'))
    concatenated = list(issues.issue_title + split_token + issues.body)
    return (i for i in concatenated[:10000])


@registry.register_problem
class IssueToTitle(problem.Text2TextProblem):
    @property
    def is_character_level(self):
        return False

    @property
    def targeted_vocab_size(self):
        return 20000

    @property
    def vocab_name(self):
        return "vocab.gh_issue"

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def num_shards(self):
        return 100

    @property
    def use_subword_tokenizer(self):
        return True

    def generator(self, data_dir, tmp_dir, is_training):
        encoder = generator_utils.get_or_generate_vocab_inner(
            data_dir, self.vocab_file, self.targeted_vocab_size,
            issue_title_generator())

        print('done encoding')
        for issue in tqdm(issue_title_generator()):
            title, body = issue.split(split_token)
            encoded_title = encoder.encode(title) + [EOS]
            encoded_body = encoder.encode(body) + [EOS]
            yield {"inputs": encoded_body, "targets": encoded_title}

