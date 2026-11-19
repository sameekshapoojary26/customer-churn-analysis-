import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download customer churn dataset
url = "https://raw.githubusercontent.com/IBM/Customer-Churn-Data/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(url)

# Clean data
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'], errors='coerce'
)
df.dropna(inplace=True)

# Basic analysis
total_customers = len(df)
churned = df[df['Churn'] == 'Yes']
retained = df[df['Churn'] == 'No']

# Calculate churn rate
churn_rate = (len(churned) / total_customers) * 100

# Calculate Customer Lifetime Value
df['CLV'] = df['MonthlyCharges'] * df['tenure']
average_clv = df['CLV'].mean()

# Calculate MRR loss
mrr_loss = churned['MonthlyCharges'].sum()

print("CUSTOMER CHURN ANALYSIS")
print("-----------------------")
print("Total Customers:", total_customers)
print("Churned Customers:", len(churned))
print("Retained Customers:", len(retained))
print("Churn Rate:", round(churn_rate, 2), "%")
print("Average Customer Lifetime Value:", round(average_clv, 2))
print("Monthly Recurring Revenue Loss:", round(mrr_loss, 2))

# Churn by Contract Type
print("\nChurn by Contract Type:")
print(df.groupby('Contract')['Churn'].value_counts())

# Churn by Payment Method
print("\nChurn by Payment Method:")
print(df.groupby('PaymentMethod')['Churn'].value_counts())

# Churn by Tenure
df['TenureGroup'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=['0-12 Months', '13-24 Months',
            '25-48 Months', '49-72 Months']
)

print("\nChurn by Tenure:")
print(df.groupby('TenureGroup', observed=False)['Churn'].value_counts())

# Visualization 1: Churn Count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='Churn')
plt.title('Customer Churn Count')
plt.savefig('churn_count.png')
plt.show()

# Visualization 2: Churn by Contract
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn by Contract Type')
plt.xticks(rotation=15)
plt.savefig('churn_contract.png')
plt.show()

# Visualization 3: Monthly Charges Boxplot
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges vs Churn')
plt.savefig('churn_boxplot.png')
plt.show()

# Visualization 4: Correlation Heatmap
numeric_df = df.select_dtypes(include='number')

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Customer Data Correlation Heatmap')
plt.savefig('churn_heatmap.png')
plt.show()

# Business Recommendations
print("\nBUSINESS RECOMMENDATIONS")
print("------------------------")
print("1. Provide discounts to customers with month-to-month contracts.")
print("2. Offer loyalty rewards to long-term customers.")
print("3. Improve customer support and service quality.")
print("4. Provide special offers for high-risk customers.")
print("5. Encourage automatic payment methods.")

# Save analysis report
df.to_csv('customer_churn_analysis.csv', index=False)

print("\nAnalysis completed successfully!")
print("Charts and analysis report generated.")
