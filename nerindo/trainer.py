import regex
import time

import torch
from spacy.lang.id import Indonesian


class Trainer(object):

    def __init__(self, model, data, optimizer_cls, loss_fn_cls):
        self.model = model
        self.data = data
        self.optimizer = optimizer_cls(model.parameters())
        self.loss_fn = loss_fn_cls(ignore_index=self.data.tag_pad_idx)

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def accuracy(self, preds, y):
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        non_pad_elements = (y != self.data.tag_pad_idx).nonzero()  # prepare masking for paddings
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

    def epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for batch in self.data.train_iter:
            # text = [sent len, batch size]
            text = batch.word
            # tags = [sent len, batch size]
            true_tags = batch.tag
            self.optimizer.zero_grad()
            pred_tags = self.model(text)
            # to calculate the loss and accuracy, we flatten both prediction and true tags
            # flatten pred_tags to [sent len, batch size, output dim]
            pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
            # flatten true_tags to [sent len * batch size]
            true_tags = true_tags.view(-1)
            batch_loss = self.loss_fn(pred_tags, true_tags)
            batch_acc = self.accuracy(pred_tags, true_tags)
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
        return epoch_loss / len(self.data.train_iter), epoch_acc / len(self.data.train_iter)

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        with torch.no_grad():
            # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                text = batch.word
                true_tags = batch.tag
                pred_tags = self.model(text)
                pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
                true_tags = true_tags.view(-1)
                batch_loss = self.loss_fn(pred_tags, true_tags)
                batch_acc = self.accuracy(pred_tags, true_tags)
                epoch_loss += batch_loss.item()
                epoch_acc += batch_acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc = self.epoch()
            end_time = time.time()
            epoch_mins, epoch_secs = Trainer.epoch_time(start_time, end_time)
            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrn Loss: {train_loss:.3f} | Trn Acc: {train_acc * 100:.2f}%")
            val_loss, val_acc = self.evaluate(self.data.val_iter)
            print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%")
        test_loss, test_acc = self.evaluate(self.data.test_iter)
        print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%")

    def infer(self, sentence):
        self.model.eval()
        # tokenize sentence
        nlp = Indonesian()
        tokens = [token.text.lower() for token in nlp(sentence)]
        # transform to indices based on corpus vocab
        numericalized_tokens = [self.data.word_field.vocab.stoi[t] for t in tokens]
        # find unknown words
        unk_idx = self.data.word_field.vocab.stoi[self.data.word_field.unk_token]
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
        # begin prediction
        token_tensor = torch.LongTensor(numericalized_tokens)
        token_tensor = token_tensor.unsqueeze(-1)
        predictions = self.model(token_tensor)
        # convert results to tags
        top_predictions = predictions.argmax(-1)
        predicted_tags = [self.data.tag_field.vocab.itos[t.item()] for t in top_predictions]
        # print inferred tags
        max_len = max([len(token) for token in tokens])
        print(f"{'word'.ljust(max_len)}\t{'unk'.ljust(max_len)}\ttag")
        for token, tag in zip(tokens, predicted_tags):
            is_unk = "✓" if token in unks else ""
            print(f"{token.ljust(max_len)}\t{is_unk.ljust(max_len)}\t{tag}")
        return tokens, predicted_tags, unks