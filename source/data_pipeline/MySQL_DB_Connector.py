import pandas as pd
import pymysql

df = pd.read_csv('D:/AIjourney/projects/Pride Ads Project/CSV/pride_ads_cleaned_1.csv', encoding='utf-8-sig')

# ------------------------ Creating connection to MysQL and creating DB -------------------------------
def create_db_connection(host_name, user_name, user_password, db_name=None):
    try:
        connection = pymysql.connect(
            host=host_name,
            user=user_name,
            password=user_password,
            database=db_name,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.Cursor
        )
        print("MySQL Database connection successful")
        return connection
    except Exception as e:  
        print(f"Error: '{e}'")
        return None

def create_database(connection, db_name):
    cursor = connection.cursor()
    try:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_persian_ci")
        print(f"Database '{db_name}' created successfully or already exists")
    except Exception as e:  # Changed from Error to Exception
        print(f"Error creating database: '{e}'")

HOST = "localhost"
USER = "root"  
PASSWORD = ""  # Local db password..
DB_NAME = "iranian_cars_db"

connection = create_db_connection(HOST, USER, PASSWORD)

create_database(connection, DB_NAME)
connection.close()
print('Connection has been closed..')

# ------------------------------- Creating 'Pride_Cars' table in database -------------------------------
def create_table(connection, db_name):

    cursor = connection.cursor()
    cursor.execute(f"USE {db_name}")
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS pride_cars (
        id INT AUTO_INCREMENT PRIMARY KEY,
        brand VARCHAR(50),
        name VARCHAR(100),
        model VARCHAR(100),
        trim VARCHAR(50),
        year INT,
        mileage INT,
        fuel VARCHAR(50),
        transmission VARCHAR(50),
        body_status VARCHAR(100),
        price BIGINT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_year (year),
        INDEX idx_price (price),
        INDEX idx_trim (trim)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_persian_ci;
    """
    
    try:
        cursor.execute(create_table_query)
        connection.commit()  # Important: commit the changes
        print("Table 'pride_cars' created successfully")
    except Exception as e:
        print(f"Error creating table: '{e}'")
        connection.rollback()

connection = create_db_connection(HOST, USER, PASSWORD, DB_NAME)

if connection:
    create_table(connection, DB_NAME)
else:
    print("Failed to connect to database")

# ------------------------------- Inserting dataset in database table -------------------------------
def insert_data(connection, df, batch_size=100):
    
    cursor = connection.cursor()

    insert_query = """
    INSERT INTO pride_cars 
    (brand, name, model, trim, year, mileage, fuel, transmission, body_status, price)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    data_tuples = [
        tuple(x) for x in df[['Brand', 'Name', 'Model', 'Trim', 'Year', 
                              'Mileage', 'Fuel', 'Transmission', 'Body status', 'Price']].values
    ]
    
    total_rows = len(data_tuples)
    inserted_rows = 0
    
    try:
        for i in range(0, total_rows, batch_size):
            batch = data_tuples[i:i+batch_size]
            cursor.executemany(insert_query, batch)
            connection.commit()
            inserted_rows += len(batch)
            print(f"Inserted batch {i//batch_size + 1}: {len(batch)} rows")
        
        print(f"\nSuccessfully inserted {inserted_rows} out of {total_rows} total rows")
        
    except Exception as e:
        print(f"Error inserting data: '{e}'")
        connection.rollback()
        return False
    
    return True

print(f"Loaded {len(df)} rows from cleaned dataset CSV file..")


success = insert_data(connection, df, batch_size=50)
if success:
    print("Data insertion completed!")
else:
    print("Data insertion failed")

# ------------------------------- OPTIONAL: Some analysis queries -------------------------------
def run_test_queries(connection):
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    
    queries = {
        "1. Total rows": "SELECT COUNT(*) as count FROM pride_cars",
        "2. Average price": "SELECT ROUND(AVG(price)) as avg_price FROM pride_cars",
        "3. Price range": """
            SELECT 
                ROUND(MIN(price)) as min_price, 
                ROUND(MAX(price)) as max_price,
                ROUND(AVG(price)) as avg_price
            FROM pride_cars
        """,
        "4. Year distribution": """
            SELECT 
                year, 
                COUNT(*) as count,
                ROUND(AVG(price)) as avg_price
            FROM pride_cars 
            GROUP BY year 
            ORDER BY year DESC
            LIMIT 10
        """,
        "5. Trim distribution": """
            SELECT 
                trim, 
                COUNT(*) as count,
                ROUND(AVG(price)) as avg_price
            FROM pride_cars 
            GROUP BY trim 
            ORDER BY count DESC
        """,
        "6. Sample data": "SELECT * FROM pride_cars LIMIT 3"
    }
    
    print("=" * 60)
    print("DATABASE VERIFICATION RESULTS")
    print("=" * 60)
    
    for name, query in queries.items():
        print(f"\n{name}:")
        print("-" * 40)
        cursor.execute(query)
        results = cursor.fetchall()
        
        if results:
            for row in results:
                # Format output nicely
                formatted_row = []
                for key, value in row.items():
                    if isinstance(value, (int, float)) and key != 'id':
                        # Format large numbers with commas
                        if value > 1000:
                            formatted_value = f"{value:,}"
                        else:
                            formatted_value = str(value)
                    else:
                        formatted_value = str(value)
                    formatted_row.append(f"{key}: {formatted_value}")
                
                print(" | ".join(formatted_row))
        else:
            print("No results")


run_test_queries(connection)

connection.close() 

print("\n" + "=" * 60)
print("Connection closed. Database setup complete!")
print("=" * 60)

