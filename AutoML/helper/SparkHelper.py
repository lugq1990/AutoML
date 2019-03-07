# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np

def create_sparksession(config=None, enable_hive=True):
    """
    This function is used to create spark session for access to spark
    :param config: What config to be configured
    :param enable_hive: Whether or not to enable Hive Support
    :return: Instant spark object, name with spark
    """
    if config is not None:
        spark = SparkSession.builder.config(config)
    if enable_hive:
        spark = spark.enableHiveSupport()
    return spark.getOrCreate()

def create_dataframe(X, y, columns_name=None, label_name=None):
    if columns_name is None:
        columns_name = np.arange(X.shape[1]).astype(str).tolist()
    if label_name is None:
        label_name = 'label'

    spark = create_sparksession()

    # Before I create spark DataFrame, Here I will first combination data and label,
    # So here will check data and label row should be same.




def spark_dataframe_to_pandas(data):
    if not isinstance(data, pd.DataFrame):
        raise AttributeError('Provided data should be DataFrame type.')
    return data.toPandas()