import csv
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('videogamessales.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''CREATE TABLE IF NOT EXISTS Platform (
                    PFID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Platform TEXT
                )''')

cursor.execute('''CREATE TABLE IF NOT EXISTS Genre (
                    GID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Genre TEXT
                )''')

cursor.execute('''CREATE TABLE IF NOT EXISTS Score (
                    Name TEXT PRIMARY KEY,
                    User_Score REAL,
                    FOREIGN KEY (Name) REFERENCES SalesData(Name)
                )''')

cursor.execute('''CREATE TABLE IF NOT EXISTS SalesData (
                    Name TEXT PRIMARY KEY,
                    Platform TEXT,
                    Year_of_Release INTEGER,
                    Genre TEXT,
                    Publisher TEXT,
                    NA_Sales REAL,
                    EU_Sales REAL,
                    JP_Sales REAL,
                    Other_Sales REAL,
                    Global_Sales REAL,
                    Developer TEXT,
                    Rating TEXT,
                    FOREIGN KEY (Platform) REFERENCES Platform(Platform)
                )''')

# Read data from CSV and insert into tables
with open('datasets/VideoGamesSalesCleaned.csv', 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    platform_set = set()
    score_data = []
    sales_data = []
    for row in reader:
        platform_set.add(row['Platform'])
        score_data.append((row['Name'], row['User_Score']))
        sales_data.append((row['Name'], row['Platform'], row['Year_of_Release'], row['Genre'], row['Publisher'],
                           row['NA_Sales'], row['EU_Sales'], row['JP_Sales'], row['Other_Sales'], row['Global_Sales'],
                           row['Developer'], row['Rating']))

# Insert Platform data
platform_values = [(platform,) for platform in platform_set]
cursor.executemany("INSERT INTO Platform (Platform) VALUES (?)", platform_values)

# Insert Score data
cursor.executemany("INSERT INTO Score (Name, User_Score) VALUES (?, ?)", score_data)

# Insert SalesData
cursor.executemany('''INSERT INTO SalesData (Name, Platform, Year_of_Release, Genre, Publisher, NA_Sales, EU_Sales,
                    JP_Sales, Other_Sales, Global_Sales, Developer, Rating) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    sales_data)

# Commit changes and close connection
conn.commit()
conn.close()

print("Database and tables created successfully.")