# To run this you need to do pip install sentence_transformers

from sentence_transformers import SentenceTransformer
import tqdm
import numpy as np
from numpy.linalg import norm
from musts.run_benchmark import test

def predict(to_predict):
    query_prompt_name = "s2s_query"
    model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()
    sentences_1 = list(zip(*to_predict))[0]
    sentences_2 = list(zip(*to_predict))[1]

    embeddings_1 = model.encode(sentences_1, batch_size=32, show_progress_bar=True, prompt_name=query_prompt_name)
    embeddings_2 = model.encode(sentences_2, batch_size=32, show_progress_bar=True, prompt_name=query_prompt_name)

    sims = model.util.pairwise_cos_sim(embeddings_1, embeddings_2)

    return sims

test(predict)


