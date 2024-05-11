import csv
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('videogamessales.db')
cursor = conn.cursor()

# Create tables
# Create Games table
conn.execute('''CREATE TABLE IF NOT EXISTS Games (
                    Name TEXT PRIMARY KEY,
                    Platform TEXT,
                    Year_of_Release REAL,
                    Genre TEXT,
                    Publisher TEXT,
                    Rating TEXT
                    )''')

    # Create Sales table
conn.execute('''CREATE TABLE IF NOT EXISTS Sales (
                    Name TEXT,
                    NA_Sales REAL,
                    EU_Sales REAL,
                    JP_Sales REAL,
                    Other_Sales REAL,
                    Global_Sales REAL,
                    PRIMARY KEY (Name, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales),
                    FOREIGN KEY (Name) REFERENCES Games(Name)
                    )''')

    # Create User_Reviews table
conn.execute('''CREATE TABLE IF NOT EXISTS User_Reviews (
                    Name TEXT PRIMARY KEY,
                    User_Score REAL,
                    User_Count REAL,
                    FOREIGN KEY (Name) REFERENCES Games(Name)
                    )''')

with open('datasets/VideoGamesSalesCleaned.csv', 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    games_data = []
    sales_data = []
    user_reviews_data = []
    for row in reader:
        games_data.append((row['Name'], row['Platform'], row['Year_of_Release'], row['Genre'], row['Publisher'], row['Rating']))
        sales_data.append((row['Name'], row['NA_Sales'], row['EU_Sales'], row['JP_Sales'], row['Other_Sales'], row['Global_Sales']))
        user_reviews_data.append((row['Name'], row['User_Score'], row['User_Count']))

# Insert Games data
cursor.executemany('''INSERT INTO Games (Name, Platform, Year_of_Release, Genre, Publisher, Rating) 
                    VALUES (?, ?, ?, ?, ?, ?)''', games_data)

# Insert Sales data
cursor.executemany('''INSERT INTO Sales (Name, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales) VALUES (?, ?, ?, ?, ?, ?)''', sales_data)

# Insert User Reviews data
cursor.executemany('''INSERT INTO User_Reviews (Name, User_Score, User_Count) VALUES (?, ?, ?)''',
                    user_reviews_data)

# Commit changes and close connection
conn.commit()
conn.close()

print("Database and tables created successfully.")