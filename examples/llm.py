import logging
import os
import re
from pprint import pprint

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import pipeline, set_seed

from musts.run_benchmark import capitalise_words

os.environ['HF_HOME'] = '/mnt/data/hettiar1/hf_cache/'
set_seed(777)

OUTPUT_FOLDER = "outputs/llama"
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

QUERY_TYPE = "zero-shot"

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
            # return [
            #     {"role": "user",
            #      "content": f"Determine the similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. Return the label only following the prefix 'Score:' without any other text. S1: {row['sentence1']} S2: {row['sentence2']}"}]
            # return [
            #     {"role": "user",
            #      "content": f"Determine the similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. Return the score only following the prefix 'Score:' without any other text or explanations. S1: {row['sentence1']} S2: {row['sentence2']}"}]
            # return [
            #     {"role": "user",
            #      "content": f"Determine the semantic textual similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 (for no similarity) to 5.0 (for identical meaning), and can be a decimal. Return the score only following the prefix 'Score:' without any other text or explanations. S1: {row['sentence1']} S2: {row['sentence2']}"}]
            return [
                {"role": "user",
                 "content": f"Determine the semantic textual similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. Return the score only following the prefix 'Score:' without any other text or explanations. S1: {row['sentence1']} S2: {row['sentence2']}"}]

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

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    for out in tqdm(pipe(
            inputs,
            max_new_tokens=200,
            # pad_token_id=pipe.model.config.eos_token_id,
            eos_token_id=terminators,
            pad_token_id=pipe.tokenizer.eos_token_id
    )):
        assistant_outputs.append(out[0]["generated_text"][-1]['content'].strip())

    return assistant_outputs


def extract_score(response):
    try:
        score = float(re.findall('Score:\s(\d+\.\d+|\d+)', response)[0])
    except IndexError:
        score = 0.0
    if score > 5:
        score = 5.0
    elif score < 0:
        score = 0.0
    return score


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
    df = pd.DataFrame(to_predict, columns=['sentence1', 'sentence2'])
    df = df.head(10)
    # print(df.shape)

    # format chats
    df.loc[:, 'chat'] = df.apply(format_chat, axis=1)
    pprint(df.loc[:2, 'chat'].tolist(), sort_dicts=False)

    # generate responses
    responses = query(pipe_lm, df['chat'].tolist())
    df['responses'] = responses

    # extract scores
    df['preds'] = df.apply(lambda row: extract_score(row['responses']), axis=1)

    df.to_csv(os.path.join(OUTPUT_FOLDER, "sample.csv"), header=True, index=False, encoding='utf-8')


test(predict)
