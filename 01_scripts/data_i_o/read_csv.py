# Data ETL: Read CSV
# this script reads a CSV file and returns a DataFrame.


# requirements: pandas (for DataFrame)
# Install the required packages using the following command:
# pip install pandas

# Importing the required libraries

import pandas as pd

# Function to read a CSV file and return a DataFrame

def read_csv(file_path):
    
    '''
    Function to read a CSV file and return a DataFrame
    
    Parameters:
    file_path : str
    
    Returns:
    DataFrame : DataFrame
    '''
    
    # Reading the CSV file
    data = pd.read_csv(file_path)
    
    return data   

# Example Usage
if __name__ == "__main__":
    file_path = 'sample_data.csv'
    data = read_csv(file_path)
    print('Sample Dataset:')
    print(data)

### End of file ******************************************************************************************************