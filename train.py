import gc
from collections import defaultdict
from datetime import datetime

import torch
from torch import nn
from sklearn.metrics import f1_score
import numpy as np
from tqdm.notebook import tqdm
import mlflow


class Trainer:
    def __init__(
        self,
        device,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        grad_acum_step=1,
        loss_fn=nn.CrossEntropyLoss(),
    ):
        """

        grad_acum_step -- defines the frequency of an optimizer step() call (e.g. every 1 step, every 2, etc.)
        """
        assert grad_acum_step > 0, "grad_acum_step cannot be negative"

        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.current_epoch = 0
        self.step = 0
        self.best_f1_macro = 0
        self.grad_acum_step = grad_acum_step
        self.loss_fn = loss_fn

    def train_epoch(self):
        def check_grads_accumulated():
            return self.step % self.grad_acum_step == 0

        self.model.train()
        losses = []
        for batch, idx in zip(
            tqdm(self.train_dataloader), range(len(self.train_dataloader))
        ):
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["labels"]
            outputs = self.model(input_ids=input_ids)
            loss = self.loss_fn(outputs, targets)
            losses.append(loss.item())
            loss /= self.grad_acum_step
            loss.backward()

            if check_grads_accumulated():
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.scheduler.step()
            mlflow.log_metric("train loss", loss.item(), self.step)
            self.step += 1

        return np.mean(losses)

    @torch.no_grad()
    def eval_model(self, dataloader):
        self.model.eval()

        true_labels = []
        predicted_labels = []

        losses = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["labels"]
                outputs = self.model(input_ids=input_ids)
                loss = self.loss_fn(outputs, targets)
                losses.append(loss.item())

                targets = targets.cpu()
                true_labels += list(targets[targets != -1])
                predicted_labels += (
                    nn.Softmax(dim=1)(outputs.cpu()).argmax(dim=1).tolist()
                )

        f1_macro = f1_score(true_labels, predicted_labels, average="macro")
        f1_micro = f1_score(true_labels, predicted_labels, average="micro")
        f1_weighted = f1_score(true_labels, predicted_labels, average="weighted")

        return np.mean(losses), f1_macro, f1_micro, f1_weighted

    def test_model(self):
        model_state_path = mlflow.artifacts.download_artifacts(
            mlflow.get_artifact_uri("checkpoint.pt")
        )
        self.model.load_state_dict(torch.load(model_state_path))
        _, test_f1_macro, test_f1_micro, test_f1_weighted = self.eval_model(
            self.test_dataloader
        )
        mlflow.log_metric("test F1 macro", test_f1_macro, self.current_epoch)
        mlflow.log_metric("test F1 micro", test_f1_micro, self.current_epoch)
        mlflow.log_metric("test F1 wghtd", test_f1_weighted, self.current_epoch)

    def save_checkpoint(self):
        torch.save(
            self.model.state_dict(),
            "./checkpoints/checkpoint.pt",
        )
        mlflow.log_artifact("./checkpoints/checkpoint.pt")

    def train_loop(self, n_epochs):
        gc.collect()
        torch.cuda.empty_cache()

        for _ in range(n_epochs):
            print(f"Epoch: {self.current_epoch}")
            print("-" * 10)

            train_loss = self.train_epoch()
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Train loss: {round(train_loss, 4)}\n")

            val_loss, f1_macro, f1_micro, f1_weighted = self.eval_model(
                self.val_dataloader
            )
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Val loss: {round(val_loss, 4)}")
            print(f"F1 macro: {round(f1_macro, 4)}")
            print(f"F1 micro: {round(f1_micro, 4)}")
            print(f"F1 wghtd: {round(f1_weighted, 4)}\n")

            mlflow.log_metric("mean train loss", train_loss, self.current_epoch)
            mlflow.log_metric("val loss", val_loss, self.current_epoch)
            mlflow.log_metric("F1 macro", f1_macro, self.current_epoch)
            mlflow.log_metric("F1 micro", f1_micro, self.current_epoch)
            mlflow.log_metric("F1 wghtd", f1_weighted, self.current_epoch)

            if f1_macro >= self.best_f1_macro:
                mlflow.log_metric("Best F1 macro", f1_macro, self.current_epoch)
                self.best_f1_macro = f1_macro
                self.save_checkpoint()

            self.current_epoch += 1

        self.test_model()
