from typing import List


def patch2file(patch: str) -> List[str]:
    lines = patch.splitlines()
    ret = []
    for line in lines:
        if line.startswith("--- a/"):
            ret.append(line.removeprefix("--- a/").strip())
        elif line.startswith("+++ b/"):
            ret.append(line.removeprefix("+++ b/").strip())
    return list(set(ret))

def num_lines_of_file(file_path: str) -> int:
    """
    Count the number of lines in a file.
    :param file_path: Path to the file.
    :return: Number of lines in the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return len(f.readlines())

import logging
import time
import os
class LazyLogger:
    def __init__(self):
        self.logger = None

    def _setup_logger(self):
        if not os.path.exists("logs"):
            os.makedirs("logs", exist_ok=True)
        logging.basicConfig(format='%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s', 
                            level=logging.INFO, 
                            filemode='w', 
                            filename=f"logs/{time.strftime('%Y-%m-%d-%H-%M-%S')}.log")
        self.logger = logging.getLogger(__name__)

    def __getattr__(self, name):
        if self.logger is None:
            self._setup_logger()
        return getattr(self.logger, name)

logger = LazyLogger()

import nltk
from nltk.corpus import wordnet as wn
import random

# Download necessary NLTK data files
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# Define function to generate a random sentence
class RandomSentenceGenerator:
    def __init__(self) -> None:
        self.adj_list = list(wn.all_synsets(wn.ADJ))
        self.noun_list = list(wn.all_synsets(wn.NOUN))
        self.verb_list = list(wn.all_synsets(wn.VERB))
        self.adv_list = list(wn.all_synsets(wn.ADV))
        self.det_list = ["a", "the", "every", "some"]
        self.prep_list = ["in", "on", "with", "over", "under"]
        self.sentence_structure = [
            "The {adj} {noun} {verb} the {noun}.",
            "A {noun} {verb} {adv} {noun}.",
            "{noun} {verb} {det} {adj} {noun}.",
            "Every {noun} {verb} {prep} a {noun}.",
        ]
    
    def generate_random_sentence(self, num_sent=1):
        # Choose a random sentence structure
        sentences = []
        for _ in range(num_sent):
            structure = random.choice(self.sentence_structure)
            
            # Word categories
            adj = random.choice(self.adj_list)
            noun = random.choice(self.noun_list)
            verb = random.choice(self.verb_list)
            adv = random.choice(self.adv_list)
            det = random.choice(self.det_list)
            prep = random.choice(self.prep_list)
            
            # Get random words from the categories
            adj_word = random.choice(adj.lemmas()).name()
            noun_word = random.choice(noun.lemmas()).name()
            verb_word = random.choice(verb.lemmas()).name()
            adv_word = random.choice(adv.lemmas()).name()
            
            # Fill the structure with words
            sentence = structure.format(
                adj=adj_word,
                noun=noun_word,
                verb=verb_word,
                adv=adv_word,
                det=det,
                prep=prep
            )
            
            sentences.append(sentence)
        return " ".join(sentences)