import torch
from torch import nn


class BiLSTM(nn.Module):

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 lstm_layers,
                 emb_dropout,
                 lstm_dropout,
                 fc_dropout,
                 word_pad_idx):
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

    def init_embeddings(self, word_pad_idx, pretrained=None, freeze=True):
        # initialize embedding for padding as zero
        self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)
        if pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.as_tensor(pretrained),
                padding_idx=word_pad_idx,
                freeze=freeze
            )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
