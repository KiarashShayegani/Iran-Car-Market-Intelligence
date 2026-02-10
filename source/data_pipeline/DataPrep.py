import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt

# ------------------------------- Loading dataset from MySQL db -------------------------------
connection = pymysql.connect(
    host = 'localhost',
    user='root',
    password='',      # Password censored +_+
    database='iranian_cars_db',
    charset='utf8mb4'
)

query = 'SELECT * FROM pride_cars'
df = pd.read_sql(query, connection)

connection.close()

print('Shape of loaded dataset<db>:', df.shape)

df_eng = df_eng.drop(['id', 'created_at'], axis=1)

# ------------------------------- Adding 'age' column -------------------------------
current_year = 1404
df_eng['age'] = 1404 - df_eng['year']

# ------------------------------- Encoding 'trim' column -------------------------------
print('\nEncoding trim column')
print('-' * 50)
trim_avg_price = df_eng.groupby('trim')['price'].mean().sort_values()
print("Trim average prices (sorted):")
for trim, avg_price in trim_avg_price.items():
    print(f"  {trim}: {avg_price:,.0f} Toman")

trim_mapping = {}
for i, (trim, _) in enumerate(trim_avg_price.items(), 1):
    trim_mapping[trim] = i

print(f"\ntrim column encoding mapping:")
for trim, code in sorted(trim_mapping.items(), key=lambda x: x[1]):
    print(f"  {code:2} → {trim}")

df_eng['trim'] = df_eng['trim'].map(trim_mapping)

# ------------------------------- Encoding 'name' column -------------------------------
print('\nEncoding name column')
print('-' * 50)
name_avg_price = df_eng.groupby('name')['price'].mean().sort_values()
print("Name average prices:")
for name, avg_price in list(name_avg_price.items()):
    print(f"  {name}: {avg_price:,.0f} Toman")

name_mapping = {}
for i, (name, _) in enumerate(name_avg_price.items(), 1):
    name_mapping[name] = i

print(f"\nname column encoding mapping:")
for name, code in sorted(name_mapping.items(), key=lambda x: x[1]):
    print(f"  {code:2} → {name}")

df_eng['name'] = df_eng['name'].map(name_mapping)

# ------------------------------- Encoding 'body_status' column -------------------------------
body_status_hierarchy = {
    # Lowest quality - major repairs
    'اتاق تعویض': 1,          # Body replaced
    
    # Lower quality - part replacements
    'درب تعویض': 2,          # Door replaced
    'گلگیر تعویض': 3,        # Fender replaced
    'کاپوت تعویض': 4,        # Hood replaced
    
    # Medium quality - significant paint work
    'کامل رنگ': 5,           # Full paint job
    'صافکاری بدون رنگ': 6,   # Bodywork without paint
    'دور رنگ': 7,            # All-around paint
    'گلگیر رنگ': 8,          # Fender painted
    
    # Good quality - minor paint work
    'کاپوت رنگ': 9,          # Hood painted
    'دو درب رنگ': 10,        # Two doors painted
    'یک درب رنگ': 11,        # One door painted
    'چند لکه رنگ': 12,       # Several spots painted
    'دو لکه رنگ': 13,        # Two spots painted
    'یک لکه رنگ': 14,        # One spot painted
    
    # Highest quality - perfect/complete paint
    'بدون رنگ': 15           # No paint needed (original paint in good condition)
}

for status in df_eng['body_status'].unique():
    if status not in body_status_hierarchy:
        body_status_hierarchy[status] = 0

df_eng['body_status'] = df_eng['body_status'].map(body_status_hierarchy)

# ------------------------------- Encoding 'fuel' column -------------------------------
fuel_mapping = {'دوگانه سوز': 2, 'بنزینی': 1}

df_eng['fuel'] = df_eng['fuel'].map(fuel_mapping)
print(f"Fuel encoded: {df_eng['fuel'].unique()}")

# ------------------------------- Encoding 'transmission' column -------------------------------
transmission_mapping = {'اتوماتیک': 2, 'دنده ای': 1}
df_eng['transmission'] = df_eng['transmission'].map(transmission_mapping)
print(f"Transmission encoded: {df_eng['transmission'].unique()}")

# ------------------------------- Saving Engineered dataset as csv -------------------------------
df_eng.to_csv('D:/AIjourney/projects/Pride Ads Project/CSV/pride_ads_engineered_1.csv', encoding='utf-8-sig')
print('Engineered dataset has been successfully saved.')