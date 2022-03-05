
from typing import List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F



def transform_training_dataset(df: DataFrame, query_cols: List[str], document_cols: List[str],
                               doc_id_col: str) -> DataFrame:
    return (
        df
            .withColumnRenamed(doc_id_col, "id")
            .select(
            F.struct(
                query_cols
            ).alias("query"),
            F.struct(
                ["id"] + document_cols
            ).alias("document")
        )
    )


def transform_documents_set(df: DataFrame, document_cols: List[str], doc_id_col: str) -> DataFrame:
    return (
        df
            .withColumnRenamed(doc_id_col, "id")
            .select(["id"] + document_cols)
    )
