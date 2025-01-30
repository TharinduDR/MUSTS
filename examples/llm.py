import logging
import os
from pprint import pprint

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import pipeline, set_seed
import torch

from musts.run_benchmark import test, capitalise_words

os.environ['HF_HOME'] = '/mnt/data/hettiar1/hf_cache/'
set_seed(777)

OUTPUT_FOLDER = "outputs/llama"
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

QUERY_TYPE="zero-shot"

# load pipeline
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "meta-llama/Llama-3.1-8B-Instruct"

pipe_lm = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    do_sample=False,
    top_p=1.0,
)

def format_chat(row):
    match QUERY_TYPE:
        case "zero-shot":
            # return [
            #     {"role": "user", "content": f"Determine the similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. S1: {row['sentence1']} S2: {row['sentence2']} Score:"}]
            return [
                {"role": "user",
                 "content": f"Determine the similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. Return the label only following the prefix 'Score:' without any other text. S1: {row['sentence1']} S2: {row['sentence2']}"}]

        case _:
            return [
                {"role": "user",
                 "content": f"Determine the similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. S1: {row['sentence1']} S2: {row['sentence2']} Score:"}]


def query(pipe, inputs):
  """
  :param pipe: text-generation pipeline
  :param model_folder_path: list of messages
  :return: list
  """
  assistant_outputs = []

  for out in tqdm(pipe(
      inputs,
      max_new_tokens=200,
      pad_token_id = pipe.model.config.eos_token_id,
  )):
    assistant_outputs.append(out[0]["generated_text"][-1]['content'].strip())

  return assistant_outputs


def test(test_method):
    languages = ["english"]

    for language in languages:

        language_name = capitalise_words(language)
        dataset_name = 'musts' + '/' + language

        logging.info(language_name)
        test_set = Dataset.to_pandas(load_dataset(dataset_name, split='test'))

        to_predit = []
        sims = []

        for index, row in test_set.iterrows():
            to_predit.append([row['sentence_1'], row['sentence_2']])
            sims.append(row['similarity'])

        predicted_sims = test_method(to_predit)

def predict(to_predict):
    # sentences_1 = list(zip(*to_predict))[0]
    # sentences_2 = list(zip(*to_predict))[1]

    df = pd.DataFrame(to_predict, columns=['sentence1', 'sentence2'])
    df = df.head(10)
    print(df.shape)

    # format chats
    df.loc[:, 'chat'] = df.apply(format_chat, axis=1)
    pprint(df.loc[:2, 'chat'].tolist(), sort_dicts=False)

    # generate responses

    responses = query(pipe_lm, df['chat'].tolist())
    df['responses'] = responses

    df.to_csv(os.path.join(OUTPUT_FOLDER, "sample.csv"), header=True, index=False, encoding='utf-8')
    print()


test(predict)