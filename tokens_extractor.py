def extract_tokens(
    table_df, tokenizer, num_of_labels, max_tokens, max_columns, max_tokens_per_column
):
    tokens_left = max_tokens

    def available_tokens_for_column(num_of_labels_left):
        return min(tokens_left // num_of_labels_left, max_tokens_per_column)

    table_tokens = []
    for label, idx in zip(table_df.columns, range(max_columns)):
        col_str_repr = table_df[label].astype(str).str.cat(sep=" ")
        columns_left = num_of_labels - idx
        col_tokens = tokenizer(
            col_str_repr,
            truncation=True,
            max_length=available_tokens_for_column(columns_left),
        ).input_ids

        tokens_left -= len(col_tokens)
        table_tokens += col_tokens

    return table_tokens
