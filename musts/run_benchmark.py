from datasets import Dataset
from datasets import load_dataset

from musts.evaluate import pearson_corr, spearman_corr, rmse


def test(test_method, train_method):

    print("English")
    english_train = Dataset.to_pandas(load_dataset('musts/english', split='train'))
    english_test = Dataset.to_pandas(load_dataset('musts/english', split='test'))

    to_predit = []
    sims = []

    for index, row in english_test.iterrows():
        to_predit.append([row['sentence_1'], row['sentence_2']])
        sims.append(row['similarity'])

    predicted_sims = test_method(to_predit)
    print("Pearson Correlation ", pearson_corr(predicted_sims, sims))
    print("Spearman Correlation ", spearman_corr(predicted_sims, sims))
    print("RMSE ", rmse(predicted_sims, sims))




