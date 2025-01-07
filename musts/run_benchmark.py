from datasets import Dataset
from datasets import load_dataset

import pandas as pd

from musts.evaluate import pearson_corr, spearman_corr, rmse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def capitalise_words(word):
    parts = word.split('_')
    capitalised_parts = [part.capitalize() for part in parts]
    return ' '.join(capitalised_parts)


def train(train_method):
    languages = ["arabic", "brazilian_portuguese", "czech", "english", "french", "japanese", "korean", "portuguese",
                 "romanian", "serbian", "sinhala", "spanish", "tamil"]


    if train_method is not None:
        train_sets = []
        for language in languages:
            dataset_name = 'musts' + '/' + language
            train_set = Dataset.to_pandas(load_dataset(dataset_name, split='train'))
            train_sets.append(train_set)

        combined_train_set = pd.concat(train_sets, ignore_index=True)
        logging.info("=============================================")
        logging.info("Start training")
        train_method(combined_train_set)


def test(test_method):
    languages = ["arabic", "brazilian_portuguese", "czech", "english", "french", "japanese", "korean", "portuguese",
                 "romanian", "serbian", "sinhala", "spanish", "tamil"]

    for language in languages:

        language_name = capitalise_words(language)
        dataset_name = 'musts' + '/' + language

        logging.info(language_name)
        test_set = Dataset.to_pandas(load_dataset(dataset_name, split='test'))

        to_predit = []
        sims = []

        for index, row in test_set.iterrows():
            to_predit.append([row['sentence_1'], row['sentence_2']])
            sims.append(row['similarity'])

        predicted_sims = test_method(to_predit)

        logging.info("Pearson Correlation %f", pearson_corr(predicted_sims, sims))
        logging.info("Spearman Correlation %f", spearman_corr(predicted_sims, sims))
        logging.info("RMSE %f", rmse(predicted_sims, sims))

        logging.info("=============================================")




