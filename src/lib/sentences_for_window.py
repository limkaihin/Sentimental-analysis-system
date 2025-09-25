from typing import List
from src.lib.preprocessing import split_sentences

def sentences_for_window(text: str, start: int, end: int) -> List[str]:
    sent_tokens = split_sentences(text)
    return [" ".join(toks) for toks in sent_tokens[start:end + 1]]
