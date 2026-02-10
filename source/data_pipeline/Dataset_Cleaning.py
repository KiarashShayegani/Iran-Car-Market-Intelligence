import pandas as pd
import numpy as np

df = pd.read_csv('D:/AIjourney/projects/Pride Ads Project/CSV/pride_ads_1.csv')

# Step 1 ----------------------------------------------------------------------
print('Step 1:')
print('-' * 50)
df = df[df['Price'] != 0]
print('New shape:', df.shape)
print(df['Price'].value_counts()) 

# Step 2 ----------------------------------------------------------------------
print('\nStep 2:')
print('-' * 50)
df.loc[df['Trim'] == 'دنده ای', 'Trim'] = 'ساده'
print('Shape:', df.shape)

# Step 3 ----------------------------------------------------------------------
print('\nStep 3:')
print('-' * 50)

karkarde_mask = df['Mileage'] == 'کارکرده'
df_karkarde = df[karkarde_mask].copy()
df_normal = df[~karkarde_mask].copy()

df_normal['Mileage'] = pd.to_numeric(df_normal['Mileage'], errors='coerce')

def get_year_group(year):
    if 1370 <= year <= 1380:
        return 'group_1370_1380'
    elif 1381 <= year <= 1390:
        return 'group_1381_1390'
    elif 1391 <= year <= 1400:
        return 'group_1391_1400'
    elif 1401 <= year <= 1404:
        return 'group_1401_1404'
    else:
        return 'other'

df_normal['Year_Group'] = df_normal['Year'].apply(get_year_group)

mileage_by_group = df_normal.groupby('Year_Group')['Mileage'].median()

print("Median mileage by year group:")
print(mileage_by_group)
print(f"\nOverall median mileage: {df_normal['Mileage'].median():.0f}")


def impute_mileage(row):
    year = row['Year']

    if 1370 <= year <= 1380:
        return mileage_by_group.get('group_1370_1380', df_normal['Mileage'].median())
    elif 1381 <= year <= 1390:
        return mileage_by_group.get('group_1381_1390', df_normal['Mileage'].median())
    elif 1391 <= year <= 1400:
        return mileage_by_group.get('group_1391_1400', df_normal['Mileage'].median())
    elif 1401 <= year <= 1404:
        return mileage_by_group.get('group_1401_1404', df_normal['Mileage'].median())
    else:
        return df_normal['Mileage'].median()       

df_karkarde['Mileage'] = df_karkarde.apply(impute_mileage, axis=1)

df_cleaned = pd.concat([df_normal.drop('Year_Group', axis=1), df_karkarde], ignore_index=True)
df_cleaned['Mileage'] = pd.to_numeric(df_cleaned['Mileage'])

print(f"\n=== Cleaning Summary ===")
print(f"Original dataset rows: {len(df)}")
print(f"Rows with 'کارکرده' mileage: {len(df_karkarde)}")
print(f"Rows with normal mileage: {len(df_normal)}")
print(f"Cleaned dataset rows: {len(df_cleaned)}")

unique_mileages = df_cleaned['Mileage'].apply(lambda x: isinstance(x, (int, float)))
print(f"\nAll mileage values are numeric: {unique_mileages.all()}")

# Saving cleand dataset ----------------------------------------------------------------------
df_cleaned.to_csv('D:/AIjourney/projects/Pride Ads Project/CSV/pride_ads_cleaned_1.csv',
 index=False,
 encoding='utf-8-sig')
print('Cleaned dataset has been successfully saved.')
