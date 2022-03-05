import re
import unicodedata
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

clean_pattern = re.compile(r"[^a-z0-9 ]")


def clean_phrase(text: str) -> str:
    decoded = unicodedata.normalize("NFKD", text)

    return clean_pattern.sub("", decoded.lower()).strip()


clean_phrase_udf = F.udf(clean_phrase, returnType=StringType())
