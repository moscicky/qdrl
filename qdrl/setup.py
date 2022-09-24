from omegaconf import DictConfig
from torch import nn

from qdrl.batch_softmax_cross_entropy_loss import BatchSoftmaxCrossEntropyLossComputer
from qdrl.loss_computer import LossComputer
from qdrl.models import SimpleTextEncoder, TwoTower
from qdrl.preprocess import TextVectorizer, DictionaryLoaderTextVectorizer
from qdrl.triplet_loss import BatchTripletLossComputer


def setup_vectorizer(config: DictConfig) -> TextVectorizer:
    if config.text_vectorizer.type == "dictionary":
        c = config.text_vectorizer
        return DictionaryLoaderTextVectorizer(
            dictionary_path=c.dictionary_path,
            word_unigrams_limit=c.word_unigrams_limit,
            word_bigrams_limit=c.word_bigrams_limit,
            char_trigrams_limit=c.char_trigrams_limit,
            num_oov_tokens=c.num_oov_tokens
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {config.text.vectorizer.type}")


def setup_model(config: DictConfig) -> nn.Module:
    if config.model.type == "SimpleTextEncoder":
        c = config.model
        model = SimpleTextEncoder(
            num_embeddings=c.text_embedding.num_embeddings,
            embedding_dim=c.text_embedding.embedding_dim,
            fc_dim=c.fc_dim,
            output_dim=c.output_dim)
        return model
    if config.model.type == "TwoTower":
        c = config.model
        model = TwoTower(
            num_embeddings=c.text_embedding.num_embeddings,
            text_embedding_dim=c.text_embedding.embedding_dim,
            output_dim=c.output_dim
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")


def setup_loss(config: DictConfig) -> LossComputer:
    if config.loss.type == "triplet":
        return BatchTripletLossComputer(
            batch_size=config.loss.batch_size,
            negatives_count=config.loss.num_negatives,
            loss_margin=config.loss.margin)
    elif config.loss.type == "batch_softmax":
        return BatchSoftmaxCrossEntropyLossComputer(
            batch_size=config.loss.batch_size
        )
    raise ValueError(f"Unknown loss type: {config.loss.type}")