# To run this you need to do pip install -U FlagEmbedding

from sentence_transformers import util
from FlagEmbedding import FlagModel

from musts.run_benchmark import test

model = FlagModel('BAAI/bge-multilingual-gemma2',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

def predict(to_predict):
    sentences_1 = list(zip(*to_predict))[0]
    sentences_2 = list(zip(*to_predict))[1]

    embeddings_1 = model.encode(sentences_1, batch_size=32)
    embeddings_2 = model.encode(sentences_2, batch_size=32)

    cosine_similarity_matrix = util.cos_sim(embeddings_1, embeddings_2)
    sims = cosine_similarity_matrix.diagonal().tolist()

    return sims

test(predict)