from pyspark.sql import SparkSession
from datasets.transform import transform_training_dataset, transform_documents_set
from datasets.helpers import clean_phrase_udf

from pyspark.sql import functions as F

def create_dataset(spark: SparkSession, raw_dataset_path: str, output_dir: str) -> None:
    df = spark.read.csv(raw_dataset_path, header=True)
    df = (
        df.drop("id")
            .withColumn("product_title", clean_phrase_udf(F.col("product_title")))
            .withColumn("search_term", clean_phrase_udf(F.col("search_term")))
    )

    df.cache()

    documents_ds = transform_documents_set(df.dropDuplicates(subset=["product_uid"]), document_cols=["product_title"],
                                           doc_id_col="product_uid")

    relevant_pairs = df.filter(F.col("relevance") > 2.0)

    training_ds = transform_training_dataset(relevant_pairs, query_cols=["search_term"],
                                             document_cols=["product_title"], doc_id_col="product_uid")
    training_ds.repartition(1).write.json(f"{output_dir}/training_ds", mode="overwrite")
    documents_ds.repartition(1).write.json(f"{output_dir}/document_ds", mode="overwrite")


def create_spark_context() -> SparkSession:
    spark = SparkSession.builder.master("local[*]").config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()
    spark.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    return spark


if __name__ == '__main__':
    spark = create_spark_context()
    create_dataset(
        spark, raw_dataset_path="resources/raw_datasets/home_depot.csv", output_dir="resources/datasets/home_depot"
    )
