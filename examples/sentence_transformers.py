# To run this you need to do pip install sentence_transformers

from sentence_transformers import SentenceTransformer, util
from musts.run_benchmark import test

model = SentenceTransformer("Lajavaness/bilingual-embedding-large", trust_remote_code=True).cuda()

def predict(to_predict):
    query_prompt_name = "s2s_query"
    sentences_1 = list(zip(*to_predict))[0]
    sentences_2 = list(zip(*to_predict))[1]

    embeddings_1 = model.encode(sentences_1, batch_size=32, show_progress_bar=True, prompt_name=query_prompt_name)
    embeddings_2 = model.encode(sentences_2, batch_size=32, show_progress_bar=True, prompt_name=query_prompt_name)

    cosine_similarity_matrix = util.cos_sim(embeddings_1, embeddings_2)
    sims = cosine_similarity_matrix.diagonal().tolist()

    return sims

test(predict)


