# 2784246

import os
import os
import re

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import pipeline, set_seed

from musts.run_benchmark import test

os.environ['HF_HOME'] = '/mnt/data/hettiar1/hf_cache/'
set_seed(777)

QUERY_TYPE = "few-shot-en"

# load pipeline
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_FOLDER = os.path.join("outputs", model_id.split('/')[-1])
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

pipe_lm = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    do_sample=False,
    top_p=1.0,
)


def get_few_shots(language, n=5):
    dataset_name = 'musts' + '/' + language
    train_set = Dataset.to_pandas(load_dataset(dataset_name, split='train'))
    few_shot_df = train_set.sample(n, random_state=777)
    return few_shot_df


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

        case "zero-shot-sys":
            return [
                {"role": "system",
                 "content": f"Determine the semantic textual similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. Return the score only following the prefix 'Score:' without any other text or explanations."},
                {"role": "user",
                 "content": f"S1: {row['sentence1']} S2: {row['sentence2']}"}]

        case "few-shot-en":
            few_shot_df = get_few_shots("English")
            few_shot_prompt = ("Five demonstration examples\n\n")
            for idx, (index, row) in enumerate(few_shot_df.iterrows()):
                fewshot_prompt = fewshot_prompt + f"Example {idx + 1}:\n S1: {row['sentence_1']} S2: {row['sentence_2']} Score: {row['similarity']}\n\n"

            return [
                {"role": "user",
                 "content": few_shot_prompt + f"Determine the semantic textual similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. Return the score only following the prefix 'Score:' without any other text or explanations. S1: {row['sentence1']} S2: {row['sentence2']}"}]
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


# def test(test_method):
#     languages = ["english"]
#
#     for language in languages:
#
#         language_name = capitalise_words(language)
#         dataset_name = 'musts' + '/' + language
#
#         logging.info(language_name)
#         test_set = Dataset.to_pandas(load_dataset(dataset_name, split='test'))
#
#         to_predit = []
#         sims = []
#
#         for index, row in test_set.iterrows():
#             to_predit.append([row['sentence_1'], row['sentence_2']])
#             sims.append(row['similarity'])
#
#         predicted_sims = test_method(to_predit)


def predict(to_predict, language):
    df = pd.DataFrame(to_predict, columns=['sentence1', 'sentence2'])
    df = df.head(10)
    # print(df.shape)

    # format chats
    df.loc[:, 'chat'] = df.apply(format_chat, axis=1)
    # pprint(df.loc[:2, 'chat'].tolist(), sort_dicts=False)

    # generate responses
    responses = query(pipe_lm, df['chat'].tolist())
    df['responses'] = responses

    # extract scores
    df['preds'] = df.apply(lambda row: extract_score(row['responses']), axis=1)

    df.to_csv(os.path.join(OUTPUT_FOLDER, f"{language}.csv"), header=True, index=False,
              encoding='utf-8')

    return df['preds'].tolist()


test(predict)