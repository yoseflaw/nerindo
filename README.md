## Nerindo

Named Entity Recognition for Bahasa Indonesia NER with PyTorch.

Corpus for NER:
* https://github.com/yohanesgultom/nlp-experiments
* https://github.com/yusufsyaifudin/indonesia-ner

The step-by-step implementation in Google Colab is indexed [here](https://medium.com/@yoseflaw/step-by-step-ner-model-for-bahasa-indonesia-with-pytorch-and-torchtext-6f94fca08406?source=friends_link&sk=c15c89082c00c8785577e1cebb77c9c2).

The Fine-tuned Indonesian word embeddings `id_ft.bin` is available [here](https://drive.google.com/file/d/1BGWnSHGZXdPfVCCkvx3_ZbjNnKh2t9pF/view?usp=sharing), based on word embeddings trained in [indonesian-word-embedding](https://github.com/galuhsahid/indonesian-word-embedding).

### Included configurations
1. BiLSTM
2. BiLSTM + Word Embeddings
3. BiLSTM + Word Embeddings + Char Embeddings (CNN)
4. BiLSTM + Word Embeddings + Char Embeddings (CNN) + Attention Layer
5. Transformer (simplified BERT) + Word Embeddings + Char Embeddings (CNN)

### Learning rate finder
Automatic learning rate finder based on [pytorch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder).

Note: since the learning rates are determined automatically from the same range for all models, it may not be the best learning rate. To see the best learning rate, check the google colab version.

Example output:

<img src="https://github.com/yoseflaw/nerindo/blob/master/images/lr_finder.png" alt="LR Finder Example Output"/>

### Final result

<img src="https://github.com/yoseflaw/nerindo/blob/master/images/final_result.png" alt="LR Finder Example Output"/>

### Main reference

Gunawan, W., Suhartono, D., Purnomo, F., & Ongko, A. (2018). Named-entity recognition for indonesian language using bidirectional lstm-cnns. Procedia Computer Science, 135, 425-432.
