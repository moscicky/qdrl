from pyspark.sql import SparkSession
from datasets.helpers import clean_phrase_udf

from pyspark.sql import functions as F


def create_training_dataset(spark: SparkSession, raw_dataset_path: str, output_dir: str) -> None:
    df = spark.read.csv(f"{raw_dataset_path}/train.csv", header=True)
    df = (
        df.drop("id")
            .withColumn("document_product_title", clean_phrase_udf(F.col("product_title")))
            .withColumn("query_search_term", clean_phrase_udf(F.col("search_term")))
            .withColumnRenamed("product_uid", "document_id")
    )

    df.cache()

    documents_ds = df.dropDuplicates(subset=["document_id"]).select(F.col("document_id"),
                                                                    F.col("document_product_title"))

    relevant_pairs = df.filter(F.col("relevance") > 2.0).select(F.col("document_id"),
                                                                    F.col("document_product_title"),
                                                                    F.col("query_search_term"))

    relevant_pairs.repartition(1).write.csv(f"{output_dir}/training/qd_pairs", mode="overwrite", header=True)
    documents_ds.repartition(1).write.csv(f"{output_dir}/training/documents", mode="overwrite", header=True)


def create_evaluation_dataset(spark: SparkSession, raw_dataset_path: str, output_dir: str):
    solution = spark.read.csv(f"{raw_dataset_path}/solution.csv", header=True).limit(10000)
    test = spark.read.csv(f"{raw_dataset_path}/test.csv", header=True).limit(10000)

    df = test.join(solution, on="id", how="inner")
    df = (
        df.drop("id")
            .withColumn("document_product_title", clean_phrase_udf(F.col("product_title")))
            .withColumn("query_search_term", clean_phrase_udf(F.col("search_term")))
            .withColumnRenamed("product_uid", "document_id")
    )

    df.cache()

    documents_ds = df.dropDuplicates(subset=["document_id"]).select(F.col("document_id"),
                                                                    F.col("document_product_title"))

    queries = (
        df.filter(F.col("relevance") > 2.0)
            .groupBy("query_search_term")
            .agg(F.collect_set("document_id").alias("document_ids"))
            .withColumn("document_ids", F.col("document_ids").cast("string"))

    )

    documents_ds.repartition(1).write.csv(f"{output_dir}/evaluation/documents", mode="overwrite", header=True)
    queries.repartition(1).write.csv(f"{output_dir}/evaluation/queries", mode="overwrite", header=True)


def create_spark_context() -> SparkSession:
    spark = SparkSession.builder.master("local[*]").config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()
    spark.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    return spark


if __name__ == '__main__':
    spark = create_spark_context()
    create_training_dataset(
        spark, raw_dataset_path="resources/raw_datasets/home_depot", output_dir="resources/datasets/home_depot"
    )

    create_evaluation_dataset(spark, raw_dataset_path="resources/raw_datasets/home_depot",
                              output_dir="resources/datasets/home_depot")
