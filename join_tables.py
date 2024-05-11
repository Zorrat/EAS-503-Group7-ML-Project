import sqlite3
import pandas as pd
import numpy as np

# Connect to SQLite database
conn = sqlite3.connect('videogamessales.db')

# SQL query to fetch data using JOINs
sql_query = '''
    SELECT g.Name, g.Platform, g.Year_of_Release, g.Genre, g.Publisher, g.Rating,
       s.NA_Sales, s.EU_Sales, s.JP_Sales, s.Other_Sales, s.Global_Sales,
       u.User_Score, u.User_Count
FROM Games g
LEFT JOIN Sales s ON g.Name = s.Name
LEFT JOIN User_Reviews u ON g.Name = u.Name
'''
# Execute the SQL query and fetch data into a Pandas DataFrame
df = pd.read_sql_query(sql_query, conn)

# Read the original CSV file into a DataFrame
#original_csv_df = pd.read_csv('datasets/VideoGamesSalesCleaned.csv', usecols=lambda col: col != 'index')

#compare = df.equals(original_csv_df)
#print(compare)

# Reset index of both DataFrames if needed
#df1_reset = df.reset_index(drop=True)
#df2_reset = original_csv_df.reset_index(drop=True)

# Compare the values of the two DataFrames
#are_equal = (df1_reset.values == df2_reset.values).all()

#print(are_equal)

# Find the differing values element-wise
#differing_values = np.where(df1_reset.values != df2_reset.values)

# Count the total number of differing values
#total_differences = len(differing_values[0])

#print(f"Total differing values: {total_differences}")

# Close the connection
conn.close()

# Display the DataFrame
print(df)
#print(original_csv_df)