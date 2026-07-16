import kagglehub
import shutil
import os
import pandas as pd

# path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

# data_folder = "data"
# os.makedirs(data_folder, exist_ok=True)

# for item in os.listdir(path):
#     source = os.path.join(path, item)
#     destination = os.path.join(data_folder, item)

#     if os.path.isfile(source):
#         shutil.copy2(source, destination)
#         print(f"Copied: {item}")
# print(f"\n All files saved to '{data_folder}' folder")

customers_df = pd.read_csv('data/olist_customers_dataset.csv')
print(f"Shape: {customers_df.shape}")
print(customers_df.head())