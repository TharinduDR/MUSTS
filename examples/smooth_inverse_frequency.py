# To run this you need to do pip install simplests
from simplests.model_args import WordEmbeddingSTSArgs
from simplests.algo.sif import WordEmbeddingSIFSTSMethod

from musts.run_benchmark import test


def predict(to_predict):
    model_args = WordEmbeddingSTSArgs()
    model_args.embedding_models = [["transformer", "FacebookAI/xlm-roberta-large"]]
    model = WordEmbeddingSIFSTSMethod(model_args=model_args)
    pred_sims = model.predict(to_predict)

    return pred_sims

test(predict)