import time
import torch
from torch import nn
from torch.optim import Adam
from torchtext.data import Field, BucketIterator
from torchtext.datasets import SequenceTaggingDataset
from spacy.lang.id import Indonesian


class Corpus(object):

    def __init__(self, input_folder, min_word_freq, batch_size):
        # list all the fields
        self.word_field = Field(lower=True)
        self.tag_field = Field(unk_token=None)
        # create dataset using built-in parser from torchtext
        self.train_dataset, self.val_dataset, self.test_dataset = SequenceTaggingDataset.splits(
            path=input_folder,
            train="train.tsv",
            validation="val.tsv",
            test="test.tsv",
            fields=(("word", self.word_field), ("tag", self.tag_field))
        )
        # convert fields to vocabulary list
        self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
        self.tag_field.build_vocab(self.train_dataset.tag)
        # create iterator for batch input
        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            datasets=(self.train_dataset, self.val_dataset, self.test_dataset),
            batch_size=batch_size
        )
        # prepare padding index to be ignored during model training/evaluation
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]


class BiLSTM(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, lstm_layers,
               emb_dropout, lstm_dropout, fc_dropout, word_pad_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        # LAYER 1: Embedding
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=word_pad_idx
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        # LAYER 2: BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )
        # LAYER 3: Fully-connected
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # times 2 for bidirectional

    def forward(self, sentence):
        # sentence = [sentence length, batch size]
        # embedding_out = [sentence length, batch size, embedding dim]
        embedding_out = self.emb_dropout(self.embedding(sentence))
        # lstm_out = [sentence length, batch size, hidden dim * 2]
        lstm_out, _ = self.lstm(embedding_out)
        # ner_out = [sentence length, batch size, output dim]
        ner_out = self.fc(self.fc_dropout(lstm_out))
        return ner_out

    def init_weights(self):
        # to initialize all parameters from normal distribution
        # helps with converging during training
        for name, param in self.named_parameters():
          nn.init.normal_(param.data, mean=0, std=0.1)

    def init_embeddings(self, tag_pad_idx):
        # initialize embedding for padding as zero
        self.embedding.weight.data[tag_pad_idx] = torch.zeros(self.embedding_dim)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class NER(object):

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
            epoch_mins, epoch_secs = NER.epoch_time(start_time, end_time)
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


if __name__ == "__main__":
    corpus = Corpus(
        input_folder="../input",
        min_word_freq=3,  # any words occurring less than 3 times will be ignored from vocab
        batch_size=64
    )
    print(f"Train set: {len(corpus.train_dataset)} sentences")
    print(f"Val set: {len(corpus.val_dataset)} sentences")
    print(f"Test set: {len(corpus.test_dataset)} sentences")
    bilstm = BiLSTM(
        input_dim=len(corpus.word_field.vocab),
        embedding_dim=300,
        hidden_dim=64,
        output_dim=len(corpus.tag_field.vocab),
        lstm_layers=2,
        emb_dropout=0.5,
        lstm_dropout=0.1,
        fc_dropout=0.25,
        word_pad_idx=corpus.word_pad_idx
    )
    bilstm.init_weights()
    bilstm.init_embeddings(tag_pad_idx=corpus.tag_pad_idx)
    print(f"The model has {bilstm.count_parameters():,} trainable parameters.")
    ner = NER(
        model=bilstm,
        data=corpus,
        optimizer_cls=Adam,
        loss_fn_cls=nn.CrossEntropyLoss
    )
    ner.train(11)
    sentence = "\"Menjatuhkan sanksi pemberhentian tetap kepada teradu Sophia Marlinda Djami selaku Ketua KPU Kabupaten Sumba Barat, sejak dibacakannya putusan ini\", ucap Alfitra dalam sidang putusan, Rabu (8/7/2020)."
    words, infer_tags, unknown_tokens = ner.infer(sentence=sentence)
