# Data ETL: Write CSV
# this script writes a DataFrame to a CSV file

# requirements: pandas (for DataFrame)
# Install the required packages using the following command:
# pip install pandas

# Importing the required libraries

import pandas as pd

# Function to write a DataFrame to a CSV file
def write_csv(df, file_name):
    """
    Function to write a DataFrame to a CSV file

    Parameters
    ----------
    df : DataFrame
        DataFrame to be written to a CSV file
    file_name : str
        Name of the CSV file

    Returns
    -------
    None
    """
    df.to_csv(file_name, index=False)
    print("DataFrame written to CSV file")
    
# Example
if __name__ == "__main__":
    # Create a DataFrame
    data = {'Name': ['Tom', 'Jerry', 'Mickey', 'Donald'],
            'Age': [25, 30, 35,
                    40]}
    df = pd.DataFrame(data)
    # Write the DataFrame to a CSV file
    write_csv(df, 'data.csv')

### End of file ******************************************************************************************************