from datasets import Dataset
from datasets import load_dataset

from musts.evaluate import pearson_corr, spearman_corr, rmse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test(test_method, train_method):

    logging.info("English")
    english_train = Dataset.to_pandas(load_dataset('musts/english', split='train'))
    english_test = Dataset.to_pandas(load_dataset('musts/english', split='test'))

    to_predit = []
    sims = []

    for index, row in english_test.iterrows():
        to_predit.append([row['sentence_1'], row['sentence_2']])
        sims.append(row['similarity'])

    predicted_sims = test_method(to_predit)
    logging.info("Pearson Correlation ", pearson_corr(predicted_sims, sims))
    logging.info("Spearman Correlation ", spearman_corr(predicted_sims, sims))
    logging.info("RMSE ", rmse(predicted_sims, sims))




