from musts.run_benchmark import test
from sentence_transformers import SentenceTransformer, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import logging
from sklearn.model_selection import train_test_split

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

final_output_dir = ""
def train(train_df):
    model_name = "dunzhang/stella_en_400M_v5"
    train_batch_size = 4
    num_epochs = 5
    output_dir = (
        "output/training_musts_" + model_name.replace("/", "-")
    )

    # 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
    # create one with "mean" pooling.
    model = SentenceTransformer(model_name, trust_remote_code=True)
    train_df = train_df[["sentence_1", "sentence_2", "similarity"]]
    train_df = train_df.rename(columns={'similarity': 'score'})

    train_dataset, eval_dataset = train_test_split(train_df, test_size=0.2)

    train_loss = losses.CosineSimilarityLoss(model=model)
    # train_loss = losses.CoSENTLoss(model=model)

    # 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval["sentence_1"],
        sentences2=eval["sentence_2"],
        scores=eval["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )

    # 5. Define the training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="musts-" + model_name,  # Will be used in W&B if `wandb` is installed
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



def predict(to_predict):
    model = SentenceTransformer(final_output_dir, trust_remote_code=True).cuda()
    # query_prompt_name = "s2s_query"
    sentences_1 = list(zip(*to_predict))[0]
    sentences_2 = list(zip(*to_predict))[1]

    embeddings_1 = model.encode(sentences_1, batch_size=32, show_progress_bar=True)
    embeddings_2 = model.encode(sentences_2, batch_size=32, show_progress_bar=True)

    cosine_similarity_matrix = util.cos_sim(embeddings_1, embeddings_2)
    sims = cosine_similarity_matrix.diagonal().tolist()

    del(model)

    return sims

test(predict,train)
