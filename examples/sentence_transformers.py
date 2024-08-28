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

    sims = []
    for embedding_1, embedding_2 in tqdm(zip(embeddings_1, embeddings_2), total=len(embeddings_1),
                                         desc="Calculating similarity "):
        cos_sim = np.dot(embedding_1, embedding_2) / (
                norm(embedding_1) * norm(embedding_2))
        sims.append(cos_sim)

    return sims

test(predict)


