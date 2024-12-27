import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv(r'C:\Users\shalini.annam\OneDrive - Qentelli\Desktop\AIMLprojects\Marketing_Project2\1689942193_marketing_data\marketing_data.csv');

data.drop_duplicates(inplace=True)

# Clean the 'income' column by removing '$' and ',' and convert to numeric
data[' Income '] = data[' Income '].replace({'\$': '', ',': ''}, regex=True)

# Convert 'income' column to numeric (float)
data[' Income '] = pd.to_numeric(data[' Income '], errors='coerce')  # 'coerce' will set invalid values to NaN
grouped = data.groupby(['Education', 'Marital_Status'])[' Income '].mean()

# Function to fill missing income based on group mean
def impute_income(row, grouped):
    if pd.isnull(row[' Income ']):
        # Get the mean income for the corresponding education and marital status group
        return grouped.get((row['Education'], row['Marital_Status']), row[' Income '])
    else:
        return row[' Income ']

# Apply the function to the DataFrame
data[' Income '] = data.apply(impute_income, axis=1, grouped=grouped)

# Check if all missing values are filled
# print(data.isnull().sum())

sum_of_children = data["Kidhome"]
this_year = datetime.date.today().year
birth_year = data['Year_Birth']
age = this_year - birth_year
mnt_columns = data.filter(like='Mnt')
df = pd.DataFrame(mnt_columns)
total_spent = df.sum(axis=1)
data["Children"] = sum_of_children
data["Age"] = age
df["Total_Spent"] = total_spent
final_data = pd.concat((data["Children"], data["Age"], df["Total_Spent"]))
# print((data["Children"], data["Age"], df["Total_Spent"]))

###########
# Create the DataFrame
df = pd.DataFrame(data)
df["Total_Spent"] = total_spent
print(df)
# Plot box plots to detect outliers
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.boxplot(x=df[' Income '])
plt.title('Box Plot for Income')

plt.subplot(2, 2, 2)
sns.boxplot(x=df['Age'])
plt.title('Box Plot for Age')

plt.subplot(2, 2, 3)
sns.boxplot(x=df['Total_Spent'])
plt.title('Box Plot for Total Spending')

# Plot histograms
plt.subplot(2, 2, 4)
df[[' Income ', 'Age', 'Total_Spent']].hist(bins=15, figsize=(12, 8), layout=(2, 2))
plt.suptitle('Histograms for Income, Age, and Total Spending')

# Show plots
plt.tight_layout()
plt.show()

# Calculate Q1 (25th percentile) and Q3 (75th percentile) for each column
Q1 = df[[' Income ', 'Age', 'Total_Spent']].quantile(0.25)
Q3 = df[[' Income ', 'Age', 'Total_Spent']].quantile(0.75)

# Calculate IQR (Interquartile Range)
IQR = Q3 - Q1

# Define outlier thresholds (lower and upper bounds)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers by checking which values fall outside the bounds
outliers = (df[[' Income ', 'Age', 'Total_Spent']] < lower_bound) | (df[[' Income ', 'Age', 'Total_Spent']] > upper_bound)

# Print the outliers
print("Outliers detected:")
print(df[outliers.any(axis=1)])

# Remove outliers from the dataset
df_cleaned = df[~((df[[' Income ', 'Age', 'Total_Spent']] < lower_bound) | (df[[' Income ', 'Age', 'Total_Spent']] > upper_bound)).any(axis=1)]

print("Data after removing outliers:")
print(df_cleaned)

# #  Ordinal Encoding for 'Education'
# education_order = ['2n Cycle','Graduation\'s', 'Master\'s', 'PhD\'s']  # Define the order
# ordinal_encoder = OrdinalEncoder(categories=[education_order])

# # Apply Ordinal Encoding to the 'education' column
# df['education_encoded'] = ordinal_encoder.fit_transform(df[['Education']])
# Correlation matrix (only for numeric columns)
# corr = df[[' Income ', 'Age', 'Education']].corr()

# # Create the heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()

######### products are performing the best, and which are performing the least in terms 
product_columns = data.filter(like='Mnt')
df = pd.DataFrame(product_columns)
# Use the melt function to reshape the data
df = df.melt(var_name='ProductCategory', value_name='TotalRevenue')
# Group by ProductCategory and sum the TotalRevenue
grouped = df.groupby('ProductCategory')['TotalRevenue'].sum().reset_index()
# Sort the DataFrame by TotalRevenue
grouped = grouped.sort_values(by='TotalRevenue', ascending=False)
# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(grouped['ProductCategory'], grouped['TotalRevenue'], color='blue')
plt.xlabel('Product Category')
plt.ylabel('Total Revenue')
plt.title('Revenue by Product Category')
plt.xticks(rotation=45)
# Display the chart
plt.tight_layout()
plt.show()

##### Total Spend by Number of Children at Home
# Calculate the sum of all columns that start with 'Mnt'
data1 = pd.read_csv(r'C:\Users\shalini.annam\OneDrive - Qentelli\Desktop\AIMLprojects\Marketing_Project2\1689942193_marketing_data\marketing_data.csv');
df = pd.DataFrame(data1)
total_mnt = df.filter(like='Mnt').sum(axis=1)
# Create a new DataFrame with 'Kidhome' and 'TotalMnt' columns
new_df = pd.DataFrame({'Kidhome': df['Kidhome'], 'TotalMnt': total_mnt})
plt.figure(figsize=(8, 6))
sns.barplot(x='Kidhome', y='TotalMnt', data=new_df, palette='viridis')
plt.title('Total Spend by Number of Children at Home')
plt.xlabel('Number of Children at Home (Kidhome)')
plt.ylabel('Total Spend (TotalMnt)')
plt.show()

########## Education background of customers who complained in the last 2 years
# Create DataFrame
df = pd.DataFrame(data)

# Filter customers who complained in the last 2 years
complained_customers = df[df['Complain'] == 1]

# Get the education background of customers who complained
education_complaints = complained_customers['Education'].value_counts()

print("Education background of customers who complained in the last 2 years:")
print(education_complaints)

####### country with the greatest number of customers who accepted the last campaign
df = pd.DataFrame(data)

# Count the number of accepted campaigns by country
accepted_campaign_by_country = df[df['Response'] == 1].groupby('Country')['ID'].count()

# Find the country with the highest number of accepted campaigns
max_country = accepted_campaign_by_country.idxmax()
max_country_count = accepted_campaign_by_country.max()

print(f"The country with the greatest number of customers who accepted the last campaign is {max_country} with {max_country_count} customers.")