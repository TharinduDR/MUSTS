# To run this you need to do pip install simplests
from simplests.model_args import WordEmbeddingSTSArgs
from simplests.algo.word_avg import WordEmbeddingAverageSTSMethod

from musts.run_benchmark import test


def predict(to_predict):
    model_args = WordEmbeddingSTSArgs()
    model_args.embedding_models = [["transformer", "google/rembert"]]
    model = WordEmbeddingAverageSTSMethod(model_args=model_args)
    pred_sims = model.predict(to_predict)

    return pred_sims

test(predict)