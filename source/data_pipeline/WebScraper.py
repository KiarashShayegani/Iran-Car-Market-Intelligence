import json
import pandas as pd
import numpy as np
import requests

total_pages = 21   
car_counter = 0
page_counter = 0

columns = [
    'Brand',
    'Name',
    'Model',
    'Trim',
    'Year',
    'Mileage',
    'Fuel',
    'Transmission',
    'Body status',
    'Price'
]

df = pd.DataFrame(columns=columns)          

for page in range(total_pages):
    page_counter += 1
    print(f'Receiving page No.{page_counter}')
    pageReq = requests.get(f'https://bama.ir/cad/api/search?vehicle=pride&pageIndex={page}') 
    print('Responded HTTP Code:', pageReq)                                                     
    
    pageReqTxt = pageReq.text 
    pageReqJson = json.loads(pageReqTxt)      
    
    container = pageReqJson.get('data').get('ads') 
    for item in container:
        item_type = item.get('type')
        if item_type == 'ad':
            car_counter +=1

            car_brand = item.get('detail').get('brand') 
            
            car_title_list = item.get('detail').get('title').split('،')          
            if len(car_title_list) == 2:    
                car_name = car_title_list[1].strip()
                car_model = car_title_list[0].strip()                    
            # else:
            #     car_name = str(car_title_list[0])
            #     car_model = str(car_title_list[0])
            
            car_trim = item.get('detail').get('trim')
            car_year = int(item.get('detail').get('year'))
            
            car_mileage = item.get('detail').get('mileage')  
            if car_mileage == 'صفر کیلومتر':
                car_mileage = 0
            elif car_mileage == 'کارکرده':
                pass
            else:
                car_mileage = int(car_mileage.replace('km','').replace(',',''))
            
            
            car_fuel = item.get('detail').get('fuel')
            car_trans = item.get('detail').get('transmission') 
            car_status = item.get('detail').get('body_status')
            
            car_price_str = item.get('price').get('price')                                
            car_price = int(car_price_str.replace(',',''))

            data_row = {
                    'Brand':car_brand,
                    'Name':car_name,
                    'Model':car_model,
                    'Trim':car_trim,
                    'Year':car_year,
                    'Mileage':car_mileage,
                    'Fuel':car_fuel,
                    'Transmission':car_trans,
                    'Body status':car_status,
                    'Price':car_price
            }

            data_row = pd.DataFrame([data_row])
            df = pd.concat([df, data_row], ignore_index=True)
            
    print('All car ads were fetched from this page.')
    print(f'{car_counter} cars info extracted so far.')
    print('-'*50)

print('\nAll pages related to Pride car brand has successfully been fetched from bama.ir !')
print('Total observed pages:', page_counter)
print('Total car ads fetched:', car_counter)

# Handling different Brands issue:
df = df[df['Brand'] == 'pride']

# Handling dtypes issues(Except for Mileage which will be done in next scripts)
str_cols = ['Brand', 'Name', 'Model', 'Trim', 'Mileage', 'Fuel', 'Transmission', 'Body status']
for col in str_cols:
    df[col] = df[col].astype(str).str.strip()
print('Converted string columns to <str>')

num_cols = ['Year', 'Price']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # coerce invalid to NaN and converts others to integer
print('Converted integer columns to <int>')

print(df.dtypes)

# Checking DataFram dtypes
print('Checking DataFram dtypes:\n')

print(df['Brand'].apply(type).value_counts())
print('-'*30)
print(df['Name'].apply(type).value_counts())
print('-'*30)
print(df['Model'].apply(type).value_counts())
print('-'*30)
print(df['Trim'].apply(type).value_counts())
print('-'*30)
print(df['Year'].apply(type).value_counts())
print('-'*30)
print(df['Mileage'].apply(type).value_counts())
print('-'*30)
print(df['Fuel'].apply(type).value_counts())
print('-'*30)
print(df['Transmission'].apply(type).value_counts())
print('-'*30)
print(df['Body status'].apply(type).value_counts())
print('-'*30)
print(df['Price'].apply(type).value_counts()) 

# Saving dataset as csv file
df.to_csv('D:/AIjourney/projects/Pride Ads Project/CSV/pride_ads_1.csv', index=False, encoding='utf-8-sig')
print('Successfully saved as csv.')
