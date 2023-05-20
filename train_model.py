import gc

import torch
from torch import nn
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from transformers import AutoConfig
import mlflow
from transformers import AutoTokenizer

from config import get_config
from rudoduo.model import Model
from train import Trainer
import dataset


def main():
    config = get_config()
    device = torch.device(config.device)
    pretrained_model_name = config.pretrained_model_name


    allowed_labels = pd.read_csv("./rudoduo/labels.csv")["0"].to_list()

    labels_encoder = preprocessing.LabelEncoder()
    labels_encoder.fit(allowed_labels)

    tokenizer = AutoTokenizer.from_pretrained("rudoduo_tokenizer")

    paths = pd.read_csv("./data/filteredData3/files_list.csv")["paths"].to_list()
    train_paths, val_test_paths = train_test_split(
        paths, test_size=0.30, random_state=509
    )
    test_paths, val_paths = train_test_split(
        val_test_paths, test_size=0.3, random_state=42
    )

    mlflow.set_experiment(experiment_name=config.mlflow.experiment_name)
    mlflow.start_run(run_name=config.mlflow.run_name)
    torch.manual_seed(509)

    mlflow.set_tag(
        "mlflow.note.content",
        config.mlflow.tag,
    )

    try:
        # dataset tokenizer
        max_tokens_per_column = config.tokenizer.max_tokens_per_column
        max_columns = config.tokenizer.max_columns
        max_tokens = config.tokenizer.max_tokens

        # trainer
        epochs = config.trainer.epochs
        grad_acum_step = config.trainer.grad_acum_step
        loss_fn = nn.CrossEntropyLoss()

        # optimizer
        lr = config.optimizer.lr
        weight_decay = config.optimizer.weight_decay
        eps = config.optimizer.eps

        # scheduler
        step_size = config.scheduler.step_size
        gamma = config.scheduler.gamma

        # model
        hidden_dropout_prob = config.model.hidden_dropout_prob
        last_layer_dropout = config.model.last_layer_dropout

        mlflow.log_param("grad_acum_step", grad_acum_step)
        mlflow.log_param("MAX_TOKENS_PER_COLUMN", max_tokens_per_column)
        mlflow.log_param("MAX_COLUMNS", max_columns)
        mlflow.log_param("MAX_TOKENS", max_tokens)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("eps", eps)
        mlflow.log_param("Epochs", epochs)
        mlflow.log_param("sched", f"StepLR step_size: {step_size}, gamma: {gamma}")
        mlflow.log_param("loss_fn", repr(loss_fn))
        mlflow.log_param("hidden_dropout_prob", hidden_dropout_prob)
        mlflow.log_param("last_layer_dropout", last_layer_dropout)

        gc.collect()
        torch.cuda.empty_cache()

        train_dataloader = dataset.Tables(
            train_paths,
            tokenizer,
            labels_encoder,
            use_rand=True,
            max_tokens=max_tokens,
            max_columns=max_columns,
            max_tokens_per_column=max_tokens_per_column,
        ).create_dataloader(batch_size=80, shuffle=True, num_workers=1)

        val_dataloader = dataset.Tables(
            val_paths,
            tokenizer,
            labels_encoder,
            max_tokens=max_tokens,
            max_columns=max_columns,
            max_tokens_per_column=max_tokens_per_column,
        ).create_dataloader(batch_size=100, num_workers=1)

        test_dataloader = dataset.Tables(
            test_paths,
            tokenizer,
            labels_encoder,
            max_tokens=max_tokens,
            max_columns=max_columns,
            max_tokens_per_column=max_tokens_per_column,
        ).create_dataloader(batch_size=100, num_workers=1)

        bert_config = AutoConfig.from_pretrained(pretrained_model_name)
        bert_config.update(
            {"hidden_dropout_prob": hidden_dropout_prob, "layer_norm_eps": 1e-7}
        )
        model = Model(
            bert_config,
            pretrained_model_name=pretrained_model_name,
            tokenizer=tokenizer,
            last_layer_dropout=last_layer_dropout,
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            pct_start=0.1,
            max_lr=lr,
            steps_per_epoch=len(train_dataloader),
            epochs=epochs,
        )
        trainer = Trainer(
            device,
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            optimizer,
            scheduler,
            grad_acum_step=grad_acum_step,
        )

        trainer.train_loop(epochs)

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
