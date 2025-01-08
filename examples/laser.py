# To run this you need to do pip install sentence_transformers

from sentence_transformers import util
from laser_encoders import LaserEncoderPipeline

from musts.run_benchmark import test

model = LaserEncoderPipeline('ces_Latn')

def predict(to_predict):
    # query_prompt_name = "s2s_query"
    sentences_1 = list(zip(*to_predict))[0]
    sentences_2 = list(zip(*to_predict))[1]

    embeddings_1 = model.encode_sentences(sentences_1)
    embeddings_2 = model.encode_sentences(sentences_2)
    cosine_similarity_matrix = util.cos_sim(embeddings_1, embeddings_2)
    sims = cosine_similarity_matrix.diagonal().tolist()

    return sims

test(predict)


