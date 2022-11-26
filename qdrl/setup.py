from typing import List

from omegaconf import DictConfig
from torch import nn

from qdrl.batch_softmax_cross_entropy_loss import BatchSoftmaxCrossEntropyLossComputer
from qdrl.configs import Features, CategoricalFeature
from qdrl.loss_computer import LossComputer
from qdrl.models import SimpleTextEncoder, TwoTower, MultiModalTwoTower
from qdrl.preprocess import TextVectorizer, DictionaryLoaderTextVectorizer, CategoricalFeatureMapper
from qdrl.triplet_loss import BatchTripletLossComputer


def parse_features(config: DictConfig) -> Features:
    query_features = []
    document_features = []
    text_features = []
    categorical_features = []

    def parse(config: List[DictConfig], features: List[str], text_features: List[str],
              categorical_features: List[CategoricalFeature]):
        for feature in config:
            features.append(feature.name)
            if feature.type == "text":
                text_features.append(feature.name)
            elif feature.type == "categorical":
                mapper = CategoricalFeatureMapper(feature.dictionary_path, feature.num_oov_categories)
                cf = CategoricalFeature(
                    name=feature.name,
                    mapper=mapper
                )
                categorical_features.append(cf)
            else:
                raise ValueError(f"Unknown feature type: {feature.type}")

    parse(config.dataset.query_features, query_features, text_features, categorical_features)
    parse(config.dataset.document_features, document_features, text_features, categorical_features)

    return Features(
        document_features=document_features,
        query_features=query_features,
        text_features=text_features,
        categorical_features=categorical_features,
    )


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
    c = config.model
    if c.type == "SimpleTextEncoder":
        model = SimpleTextEncoder(
            num_embeddings=c.text_embedding.num_embeddings,
            embedding_dim=c.text_embedding.embedding_dim,
            fc_dim=c.fc_dim,
            output_dim=c.output_dim,
            query_text_feature=c.query.text_feature,
            document_text_feature=c.document.text_feature
        )
        return model
    elif c.type == "TwoTower":
        model = TwoTower(
            num_embeddings=c.text_embedding.num_embeddings,
            text_embedding_dim=c.text_embedding.embedding_dim,
            hidden_layers=c.hidden_layers,
            output_dim=c.output_dim,
            query_text_feature=c.query.text_feature,
            document_text_feature=c.document.text_feature,
            last_linear=c.last_linear
        )
        return model
    elif c.type == "MultiModalTwoTower":
        model = MultiModalTwoTower(
            num_embeddings=c.text_embedding.num_embeddings,
            text_embedding_dim=c.text_embedding.embedding_dim,
            category_embedding_dim=c.category_feature.embedding_dim,
            category_num_embeddings=c.category_feature.num_embeddings,
            hidden_layers=c.hidden_layers,
            output_dim=c.output_dim,
            query_text_feature=c.query.text_feature,
            document_text_feature=c.document.text_feature,
            document_categorical_feature=c.document.categorical_feature,
            last_linear=c.last_linear
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
            batch_size=config.loss.batch_size,
            temperature=config.loss.temperature
        )
    raise ValueError(f"Unknown loss type: {config.loss.type}")
