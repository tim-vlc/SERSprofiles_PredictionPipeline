import os
import pandas as pd

# Specify the directory path where your CSV files are located
directory_path = '../../CSVs'
save_path = '../../CSVs/diabetes'

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Create an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Iterate through each CSV file
for csv_file in csv_files:
    # Construct the full file path
    file_path = os.path.join(directory_path, csv_file)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Concatenate the current DataFrame with the merged_data DataFrame
    merged_data = pd.concat([merged_data, df], ignore_index=True)
    
merged_data.to_csv(save_path + ".csv", index=False)

