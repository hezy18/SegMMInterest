import torch

def delete_noisy_char(s):
    s = (
        s.replace(",", " ")
        .replace("/", " ")
        .replace('"', " ")
        .replace("-", " ")
        .replace(";", " ")
        .replace(".", " ")
        .replace("&", " ")
        .replace("?", " ")
        .replace("!", " ")
        .replace("(", " ")
        .replace(")", " ")
    )
    s = s.strip()
    return s

def extend_vocab(pretrained_vocab, token, vector):
    pretrained_vocab.itos.extend([token])
    pretrained_vocab.stoi[token] = pretrained_vocab.vectors.shape[0]
    pretrained_vocab.vectors = torch.cat([pretrained_vocab.vectors, vector], dim=0)


