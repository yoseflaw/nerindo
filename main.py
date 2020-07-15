from torch import nn
from torch.optim import Adam
from nerindo.trainer import Trainer
from nerindo.corpus import Corpus
from nerindo.models import BiLSTM

if __name__ == "__main__":
    corpus = Corpus(
        input_folder="input",
        min_word_freq=3,
        batch_size=64,
        wv_file="pretrain/embeddings/id_ft.bin"
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
    bilstm.init_embeddings(
        word_pad_idx=corpus.word_pad_idx,
        pretrained=corpus.word_field.vocab.vectors if corpus.wv_model else None,
        freeze=True
    )
    print(f"The model has {bilstm.count_parameters():,} trainable parameters.")
    trainer = Trainer(
        model=bilstm,
        data=corpus,
        optimizer_cls=Adam,
        loss_fn_cls=nn.CrossEntropyLoss
    )
    trainer.train(20)
    sentence = "\"Menjatuhkan sanksi pemberhentian tetap kepada teradu Sophia Marlinda Djami selaku Ketua KPU Kabupaten Sumba Barat, sejak dibacakannya putusan ini\", ucap Alfitra dalam sidang putusan, Rabu (8/7/2020)."
    words, infer_tags, unknown_tokens = trainer.infer(sentence=sentence)
