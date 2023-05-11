from airflow import Dataset
from airflow.decorators import dag, task
import pendulum


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preparation import filter
from data_preparation import filter_by_labels
from data_preparation import change_labels
from data_preparation import generate_columns_dataset as gen_columns


raw_wikitables = Dataset("raw_wikitables")
rough_filtered_wikitables = Dataset("rough_filtered_wikitables")
label_filtered_wikitables = Dataset("label_filtered_wikitables")
replaced_labels_wikitables = Dataset("replaced_labels_wikitables")
columns_dataset = Dataset("columns_dataset")


@dag(
    catchup=False,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    tags=["filtering"],
    schedule=[raw_wikitables]
)
def filter_dirty_tables():
    """
    """
    @task(outlets=[rough_filtered_wikitables])
    def filter_dirty_tables():
        """
        """
        return filter.main()

    filter_dirty_tables()


@dag(
    catchup=False,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    tags=["filtering"],
    schedule=[rough_filtered_wikitables]
)
def generate_columns_dataset():
    """
    """
    @task(outlets=[columns_dataset])
    def generate_columns_dataset():
        """
        """
        return gen_columns.main()

    generate_columns_dataset()


@dag(
    catchup=False,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    tags=["filtering"],
    schedule=[rough_filtered_wikitables]
)
def filter_using_labels():
    """
    """
    @task(outlets=[label_filtered_wikitables])
    def filter_using_labels():
        """
        """

        return filter_by_labels.main()

    filter_using_labels()


@dag(
    catchup=False,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    tags=["filtering"],
    schedule=[label_filtered_wikitables]
)
def replace_labels():
    """
    """
    @task(outlets=[replaced_labels_wikitables])
    def replace_labels():
        """
        """

        return change_labels.main()

    replace_labels()


filter_dirty_tables()
generate_columns_dataset()
filter_using_labels()
replace_labels()
