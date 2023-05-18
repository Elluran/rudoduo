# Use model

## Load model

```Python
   from transformers import AutoTokenizer
   from model import Model


   tokenizer = AutoTokenizer.from_pretrained("Elluran/rudoduo")
   model = Model.from_pretrained(
      "Elluran/rudoduo",
      tokenizer=tokenizer
   )
   model.eval()
```

## Get predictions

```Python
   import pandas as pd
   from tokens_extractor import extract_tokens 


   df = pd.read_csv("file.csv")

   tokens = extract_tokens(
      df,
      tokenizer,
      num_of_labels=len(df.columns),
      max_tokens=200,
      max_columns=20,
      max_tokens_per_column=200
   )

   preds = model.predict(torch.tensor([tokens]))
```

# Deploy data
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
