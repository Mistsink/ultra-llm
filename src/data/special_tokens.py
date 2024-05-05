from enum import Enum
from transformers import AutoTokenizer


class SpecialToken(Enum):
    G_BEGIN = "<graph-begin>"
    G_END = "<graph-end>"
    INFO_NODE = "<info-node>"
    INFO_GRAPH = "<info-graph>"

    @staticmethod
    def add_tokens(tokenizer: AutoTokenizer) -> int:
        """
        :return: number of new tokens ( num <= tokens added)
        """
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        return tokenizer.add_tokens(
            [token.value for token in SpecialToken], special_tokens=True
        )

    @staticmethod
    def get_token_ids(tokenizer: AutoTokenizer) -> dict[str, int]:
        return {token.value: tokenizer.convert_tokens_to_ids(token.value) for token in SpecialToken}