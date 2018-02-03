from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

@registry.register_problem
class IssueTitle(problem.Text2TextProblem):
    @property
    def is_character_level(self):
        return False

    @property
    def vocab_name(self):
        return "vocab.issue_title.en"

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



