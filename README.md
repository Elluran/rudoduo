# Instruction
1. Clone repo and create .env file using the next example as a template.
    ##### example of .env
    ```BASH
    AIRFLOW_UID=1000
    ```
2. Init airflow
   ```BASH
   docker-compose up airflow-init
   ```
3. To start airflow run
   ```
   docker-compose up --force-recreate
   ```
4. Put wikitables data in rudoduo/data folder.
5. Run filter_dirty_tables in airflow. This will generate all necessary files for learning.
