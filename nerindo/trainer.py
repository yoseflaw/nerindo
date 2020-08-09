import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from spacy.lang.id import Indonesian
from sklearn.metrics import f1_score, classification_report


class Trainer(object):

    def __init__(self, model, data, optimizer, device, checkpoint_path=None):
        self.device = device
        self.model = model.to(self.device)
        self.data = data
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def f1_positive(self, preds, y, full_report=False):
        index_o = self.data.tag_field.vocab.stoi["O"]
        # take all labels except padding and "O"
        positive_labels = [i for i in range(len(self.data.tag_field.vocab.itos))
                           if i not in (self.data.tag_pad_idx, index_o)]
        # make the prediction one dimensional to follow sklearn f1 score input param
        flatten_preds = [pred for sent_pred in preds for pred in sent_pred]
        # remove prediction for padding and "O"
        positive_preds = [pred for pred in flatten_preds
                          if pred not in (self.data.tag_pad_idx, index_o)]
        # make the true tags one dimensional to follow sklearn f1 score input param
        flatten_y = [tag for sent_tag in y for tag in sent_tag]
        if full_report:
            # take all names except padding and "O"
            positive_names = [self.data.tag_field.vocab.itos[i]
                              for i in range(len(self.data.tag_field.vocab.itos))
                              if i not in (self.data.tag_pad_idx, index_o)]
            print(classification_report(
                y_true=flatten_y,
                y_pred=flatten_preds,
                labels=positive_labels,
                target_names=positive_names
            ))
        # average "micro" means we take weighted average of the class f1 score
        # weighted based on the number of support
        return f1_score(
            y_true=flatten_y,
            y_pred=flatten_preds,
            labels=positive_labels,
            average="micro"
        ) if len(positive_preds) > 0 else 0

    def epoch(self):
        epoch_loss = 0
        true_tags_epoch = []
        pred_tags_epoch = []
        self.model.train()
        for batch in self.data.train_iter:
            # words = [sent len, batch size]
            words = batch.word.to(self.device)
            # chars = [batch size, sent len, char len]
            chars = batch.char.to(self.device)
            # tags = [sent len, batch size]
            true_tags = batch.tag.to(self.device)
            self.optimizer.zero_grad()
            pred_tags_list, batch_loss = self.model(words, chars, true_tags)
            pred_tags_epoch += pred_tags_list
            # to calculate the loss and f1, we flatten true tags
            true_tags_epoch += [
                [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                for sent_tag in true_tags.permute(1, 0).tolist()
            ]
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        epoch_score = self.f1_positive(pred_tags_epoch, true_tags_epoch)
        return epoch_loss / len(self.data.train_iter), epoch_score

    def evaluate(self, iterator, full_report=False):
        epoch_loss = 0
        true_tags_epoch = []
        pred_tags_epoch = []
        self.model.eval()
        with torch.no_grad():
            # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                words = batch.word.to(self.device)
                chars = batch.char.to(self.device)
                true_tags = batch.tag.to(self.device)
                pred_tags, batch_loss = self.model(words, chars, true_tags)
                pred_tags_epoch += pred_tags
                true_tags_epoch += [
                    [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                    for sent_tag in true_tags.permute(1, 0).tolist()
                ]
                epoch_loss += batch_loss.item()
        epoch_score = self.f1_positive(pred_tags_epoch, true_tags_epoch, full_report)
        return epoch_loss / len(iterator), epoch_score

    def train(self, max_epochs, no_improvement=None):
        history = {
            "num_params": self.model.count_parameters(),
            "train_loss": [],
            "train_f1": [],
            "val_loss": [],
            "val_f1": [],
        }
        elapsed_train_time = 0
        best_val_f1 = 0
        best_epoch = None
        lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            patience=2,
            factor=0.3,
            verbose=True
        )
        epoch = 1
        n_stagnant = 0
        stop = False
        while not stop:
            start_time = time.time()
            train_loss, train_f1 = self.epoch()
            end_time = time.time()
            elapsed_train_time += end_time - start_time
            history["train_loss"].append(train_loss)
            history["train_f1"].append(train_f1)
            val_loss, val_f1 = self.evaluate(self.data.val_iter)
            lr_scheduler.step(val_loss)
            # take the current model if it it at least 1% better than the previous best F1
            if self.checkpoint_path and val_f1 > (best_val_f1 + 0.01 * best_val_f1):
                print(f"Epoch-{epoch}: found better Val F1: {val_f1:.4f}, saving model...")
                self.model.save_state(self.checkpoint_path)
                best_val_f1 = val_f1
                best_epoch = epoch
                n_stagnant = 0
            else:
                n_stagnant += 1
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            if epoch >= max_epochs:
                print(f"Reach maximum number of epoch: {epoch}, stop training.")
                stop = True
            elif no_improvement is not None and n_stagnant >= no_improvement:
                print(f"No improvement after {n_stagnant} epochs, stop training.")
                stop = True
            else:
                epoch += 1
        if self.checkpoint_path and best_val_f1 > 0:
            self.model.load_state(self.checkpoint_path)
        test_loss, test_f1 = self.evaluate(self.data.test_iter)
        history["best_val_f1"] = best_val_f1
        history["best_epoch"] = best_epoch
        history["test_loss"] = test_loss
        history["test_f1"] = test_f1
        history["elapsed_train_time"] = elapsed_train_time
        return history

    # @staticmethod
    # def visualize_attn(tokens, weights):
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     im = ax.imshow(weights, cmap=plt.get_cmap("gray"))
    #     ax.set_xticks(list(range(len(tokens))))
    #     ax.set_yticks(list(range(len(tokens))))
    #     ax.set_xticklabels(tokens)
    #     ax.set_yticklabels(tokens)
    #     # Create colorbar
    #     _ = ax.figure.colorbar(im, ax=ax)
    #     # Rotate the tick labels and set their alignment.
    #     plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
    #              rotation_mode="anchor")
    #     plt.tight_layout()
    #     plt.show()

    def infer(self, sentence, true_tags=None):
        self.model.eval()
        # tokenize sentence
        nlp = Indonesian()
        tokens = [token.text for token in nlp(sentence)]
        max_word_len = max([len(token) for token in tokens])
        # transform to indices based on corpus vocab
        numericalized_tokens = [self.data.word_field.vocab.stoi[token.lower()] for token in tokens]
        numericalized_chars = []
        char_pad_id = self.data.char_pad_idx
        for token in tokens:
            numericalized_chars.append(
                [self.data.char_field.vocab.stoi[char] for char in token]
                + [char_pad_id for _ in range(max_word_len - len(token))]
            )
        # find unknown words
        unk_idx = self.data.word_field.vocab.stoi[self.data.word_field.unk_token]
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
        # begin prediction
        token_tensor = torch.as_tensor(numericalized_tokens)
        token_tensor = token_tensor.unsqueeze(-1).to(self.device)
        char_tensor = torch.as_tensor(numericalized_chars)
        char_tensor = char_tensor.unsqueeze(0).to(self.device)
        predictions, _ = self.model(token_tensor, char_tensor)
        # convert results to tags
        predicted_tags = [self.data.tag_field.vocab.itos[t] for t in predictions[0]]
        # print inferred tags
        max_len_token = max([len(token) for token in tokens] + [len('word')])
        max_len_tag = max([len(tag) for tag in predicted_tags] + [len('pred')])
        print(
            f"{'word'.ljust(max_len_token)}\t{'unk'.ljust(max_len_token)}\t{'pred tag'.ljust(max_len_tag)}"
            + ("\ttrue tag" if true_tags else "")
        )
        for i, token in enumerate(tokens):
            is_unk = "✓" if token in unks else ""
            print(
                f"{token.ljust(max_len_token)}\t{is_unk.ljust(max_len_token)}\t{predicted_tags[i].ljust(max_len_tag)}"
                + (f"\t{true_tags[i]}" if true_tags else "")
            )
        return tokens, predicted_tags, unks
