from musts.run_benchmark import train
from sentence_transformers import SentenceTransformer, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import logging
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

final_output_dir = ""

def train_musts(train_df):
    model_name = "google/mt5-base"
    train_batch_size = 8
    num_epochs = 5
    output_dir = (
        "output/training_musts_" + model_name.replace("/", "-")
    )

    model = SentenceTransformer(model_name, trust_remote_code=True)
    # model.max_seq_length = 80
    train_df = train_df[["sentence_1", "sentence_2", "similarity"]]
    train_df['similarity'] = train_df['similarity'] / 5
    train_df = train_df.rename(columns={'sentence_1': 'sentence1', 'sentence_2': 'sentence2', 'similarity': 'score'})

    train_df, eval_df = train_test_split(train_df, test_size=0.2)

    train_df = train_df.dropna(how='any', axis=0)
    eval_df = eval_df.dropna(how='any', axis=0)

    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    train_loss = losses.CosineSimilarityLoss(model=model)

    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset["sentence1"],
        sentences2=eval_dataset["sentence2"],
        scores=eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="musts-dev",
    )

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        learning_rate=1e-6,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=500,
        run_name="musts-" + model_name.replace("/", "-"),  # Will be used in W&B if `wandb` is installed
    )

    # 6. Create the trainer & start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    trainer.train()
    final_output_dir = f"{output_dir}/final"
    model.save(final_output_dir)


# def predict(to_predict):
#     model = SentenceTransformer(final_output_dir, trust_remote_code=True).cuda()
#     # query_prompt_name = "s2s_query"
#     sentences_1 = list(zip(*to_predict))[0]
#     sentences_2 = list(zip(*to_predict))[1]
#
#     embeddings_1 = model.encode(sentences_1, batch_size=32, show_progress_bar=True)
#     embeddings_2 = model.encode(sentences_2, batch_size=32, show_progress_bar=True)
#
#     cosine_similarity_matrix = util.cos_sim(embeddings_1, embeddings_2)
#     sims = cosine_similarity_matrix.diagonal().tolist()
#
#     del(model)
#
#     return sims

train(train_musts)

