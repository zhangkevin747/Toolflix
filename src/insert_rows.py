import pandas as pd

# Load the Excel file
file_path = 'data/fixtures/sales_data.xlsx'
df = pd.read_excel(file_path)

# Create 3 empty rows (or placeholder rows)
new_rows = pd.DataFrame([
    [None] * len(df.columns),
    [None] * len(df.columns),
    [None] * len(df.columns)
], columns=df.columns)

# Insert the rows at index 4 (which corresponds to row 5 in 1-based indexing)
# We split the dataframe and concatenate
df_top = df.iloc[:4]
df_bottom = df.iloc[4:]
df_final = pd.concat([df_top, new_rows, df_bottom]).reset_index(drop=True)

# Save the file
df_final.to_excel(file_path, index=False)
