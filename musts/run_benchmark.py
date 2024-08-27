from datasets import Dataset
from datasets import load_dataset

from musts.evaluate import pearson_corr, spearman_corr, rmse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def capitalise_words(word):
    parts = word.split('_')
    capitalised_parts = [part.capitalize() for part in parts]
    return ''.join(capitalised_parts)



def test(test_method, train_method=None):
    languages = ["arabic", "brazilian_portuguese", "czech", "english", "french", "korean", "portuguese",
                 "romanian", "serbian", "sinhala", "tamil", "spanish"]

    for language in languages:

        language_name = capitalise_words(language)
        dataset_name = 'musts' + '/' + language

        logging.info(language_name)
        train_set = Dataset.to_pandas(load_dataset(dataset_name, split='train'))
        test_set = Dataset.to_pandas(load_dataset(dataset_name, split='test'))

        to_predit = []
        sims = []

        for index, row in test_set.iterrows():
            to_predit.append([row['sentence_1'], row['sentence_2']])
            sims.append(row['similarity'])

        predicted_sims = test_method(to_predit)
        logging.info("Pearson Correlation ", pearson_corr(predicted_sims, sims))
        logging.info("Spearman Correlation ", spearman_corr(predicted_sims, sims))
        logging.info("RMSE ", rmse(predicted_sims, sims))




