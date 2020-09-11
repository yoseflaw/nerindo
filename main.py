import torch
from torch.optim import Adam
from nerindo.corpus import Corpus
from nerindo.models import NERModel
from nerindo.lr_finder import LRFinder
from nerindo.trainer import Trainer
from pprint import pprint

if __name__ == "__main__":
    use_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = Corpus(
        input_folder="input",
        min_word_freq=3,
        batch_size=64,
        wv_file="pretrain/embeddings/id_ft.bin"
    )
    print(f"Train set: {len(corpus.train_dataset)} sentences")
    print(f"Val set: {len(corpus.val_dataset)} sentences")
    print(f"Test set: {len(corpus.test_dataset)} sentences")
    # configurations building block
    base = {
        "word_input_dim": len(corpus.word_field.vocab),
        "char_pad_idx": corpus.char_pad_idx,
        "word_pad_idx": corpus.word_pad_idx,
        "tag_names": corpus.tag_field.vocab.itos,
        "device": use_device
    }
    w2v = {
        "word_emb_pretrained": corpus.word_field.vocab.vectors if corpus.wv_model else None
    }
    cnn = {
        "use_char_emb": True,
        "char_input_dim": len(corpus.char_field.vocab),
        "char_emb_dim": 37,
        "char_emb_dropout": 0.25,
        "char_cnn_filter_num": 4,
        "char_cnn_kernel_size": 3,
        "char_cnn_dropout": 0.25
    }
    attn = {
        "attn_heads": 16,
        "attn_dropout": 0.25
    }
    transformer = {
        "model_arch": "transformer",
        "trf_layers": 1,
        "fc_hidden": 256,
    }
    configs = {
        "bilstm": base,
        "bilstm+w2v": {**base, **w2v},
        "bilstm+w2v+cnn": {**base, **w2v, **cnn},
        "bilstm+w2v+cnn+attn": {**base, **w2v, **cnn, **attn},
        "transformer+w2v+cnn": {**base, **transformer, **w2v, **cnn, **attn}
    }
    suggested_lrs = {}
    for model_name in configs:
        model = NERModel(**configs[model_name])
        lr_finder = LRFinder(model, Adam(model.parameters(), lr=1e-4, weight_decay=1e-2), device=use_device)
        lr_finder.range_test(corpus.train_iter, corpus.val_iter, end_lr=10, num_iter=55, step_mode="exp")
        _, suggested_lrs[model_name] = lr_finder.plot(skip_start=10, skip_end=0)
    pprint(suggested_lrs)
    max_epochs = 50
    histories = {}
    for model_name in configs:
        print(f"Start Training: {model_name}")
        model = NERModel(**configs[model_name])
        trainer = Trainer(
            model=model,
            data=corpus,
            optimizer=Adam(model.parameters(), lr=suggested_lrs[model_name], weight_decay=1e-2),
            device=use_device,
            checkpoint_path=f"saved_states/{model_name}.pt"
        )
        histories[model_name] = trainer.train(max_epochs=max_epochs, no_improvement=3)
        print(f"Done Training: {model_name}")
        print()
        trainer.model.load_state(f"saved_states/{model_name}.pt")
        sentence = "\"Menjatuhkan sanksi pemberhentian tetap kepada teradu Sophia Marlinda Djami selaku Ketua KPU Kabupaten Sumba Barat, sejak dibacakannya putusan ini\", ucap Alfitra dalam sidang putusan, Rabu (8/7/2020)."
        words, infer_tags, unknown_tokens = trainer.infer(sentence=sentence)
    print()
    pprint(histories)
