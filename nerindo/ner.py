import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torchtext.data import Field, BucketIterator
from torchtext.datasets import SequenceTaggingDataset
import time
from spacy.lang.id import Indonesian


class BiLSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        # pass text through embedding layer
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)

        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]

        return predictions


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


def train(model, iterator, optimizer, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        text = batch.word
        tags = batch.tag
        optimizer.zero_grad()
        # text = [sent len, batch size]
        predictions = model(text)
        # predictions = [sent len, batch size, output dim]
        # tags = [sent len, batch size]
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        # predictions = [sent len * batch size, output dim]
        # tags = [sent len * batch size]
        loss = criterion(predictions, tags)
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text = batch.word
            tags = batch.tag
            predictions = model(text)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            loss = criterion(predictions, tags)
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def tag_sentence(model, sentence, text_field, tag_field):
    model.eval()
    if isinstance(sentence, str):
        nlp = Indonesian()
        tokens = [token.text for token in nlp(sentence)]
    else:
        tokens = [token for token in sentence]
    if text_field.lower:
        tokens = [t.lower() for t in tokens]
    numericalized_tokens = [text_field.vocab.stoi[t] for t in tokens]
    unk_idx = text_field.vocab.stoi[text_field.unk_token]
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
    token_tensor = torch.LongTensor(numericalized_tokens)
    token_tensor = token_tensor.unsqueeze(-1)
    predictions = model(token_tensor)
    top_predictions = predictions.argmax(-1)
    predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions]
    return tokens, predicted_tags, unks


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    WORD = Field(lower=True)
    TAG = Field(unk_token=None)
    train_dataset, test_dataset = SequenceTaggingDataset.splits(
        path="../input",
        train="train.tsv",
        test="test.tsv",
        fields=(("word", WORD), ("tag", TAG))
    )
    WORD.build_vocab(train_dataset.word, min_freq=3)
    TAG.build_vocab(train_dataset.tag)
    train_iter, test_iter = BucketIterator.splits((train_dataset, test_dataset), batch_size=64, device=device)
    PAD_IDX = WORD.vocab.stoi[WORD.pad_token]
    TAG_PAD_IDX = TAG.vocab.stoi[TAG.pad_token]
    EMBEDDING_DIM = 100
    bilstm = BiLSTM(
        input_dim=len(WORD.vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=64,
        output_dim=len(TAG.vocab),
        n_layers=2,
        bidirectional=True,
        dropout=0.25,
        pad_idx=WORD.vocab.stoi[WORD.pad_token]
    )
    bilstm.apply(init_weights)
    print(f'The model has {count_parameters(bilstm):,} trainable parameters')
    pretrained_embeddings = WORD.vocab.vectors
    bilstm.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    adam = Adam(bilstm.parameters())
    ce_loss = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)
    N_EPOCHS = 15
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(bilstm, train_iter, adam, ce_loss, TAG_PAD_IDX)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")
    test_loss, test_acc = evaluate(bilstm, test_iter, ce_loss, TAG_PAD_IDX)
    print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%")
    torch.save(bilstm.state_dict(), "../pretrain/model/bilstm/simple.pt")
    bilstm.load_state_dict(torch.load("../pretrain/model/bilstm/simple.pt"))
    sentence = "\"Menjatuhkan sanksi pemberhentian tetap kepada teradu Sophia Marlinda Djami selaku Ketua KPU Kabupaten Sumba Barat, sejak dibacakannya putusan ini\", ucap Alfitra dalam sidang putusan, Rabu (8/7/2020)."
    tokens, pred_tags, unks = tag_sentence(bilstm, sentence.strip(), WORD, TAG)
    print(tokens)
    print(pred_tags)
    print(unks)
    for token, tag in zip(tokens, pred_tags):
        print(f"{token}\t\t{tag}")
