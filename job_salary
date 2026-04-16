import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("job_salary_prediction_dataset.csv")
df
print(df.shape)
print(df.columns)
df.info()
print(df.head())
print(df.describe())
print(df)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df
df['salary_per_exp'] =df['salary'] / (df['experience_years'] + 1)
df
df['exp_level']=pd.cut(df['experience_years'],
                       bins=[0,2,5,10,20],
                       labels=['fresher','junior','mid','senior'])
df

sns.histplot(df["salary"], kde=True)
plt.show()
sns.scatterplot(x='experience_years',y='salary', data=df)
plt.show()
sns.heatmap(df.corr(numeric_only=True), annot=True,)
plt.show()
sns.boxplot(x='exp_level', y='salary', data=df)
plt.show()
top_jobs =df.groupby("job_title")['salary'].mean().sort_values(ascending=False)
print(top_jobs.head(10))
avg_salary=df.groupby('job_title')['salary'].transform('mean')
df['salary_gap']=df['salary'] - avg_salary
df
plt.figure(figsize=(8,5))

sns.scatterplot(
    x='experience_years',
    y='salary',
    data=df,
    hue='exp_level',
    size='experience_years',   
    style='exp_level'    
)

plt.title("Experience vs Salary Analysis")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend(title="Experience Level")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


plt.grid(True)
plt.show()
plt.figure(figsize=(10,6))

avg_salary = df.groupby('exp_level')['salary'].mean()

avg_salary.plot(kind='bar')

plt.title("Average Salary by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Average Salary")

plt.xticks(rotation=0)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(18,12))

plt.subplot(3,2,1)
sns.histplot(df['salary'], kde=True)
plt.title("Salary Distribution")

plt.subplot(3,2,2)
sns.scatterplot(x='experience_years', y='salary', data=df)
plt.title("Experience vs Salary")

plt.subplot(3,2,3)
sns.boxplot(x='exp_level', y='salary', data=df)
plt.title("Salary by Experience Level")

plt.subplot(3,2,4)
avg_salary = df.groupby('exp_level')['salary'].mean()
avg_salary.plot(kind='bar')
plt.title("Avg Salary by Experience Level")
plt.xticks(rotation=0)

plt.subplot(3,2,5)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")

plt.subplot(3,2,6)
top_jobs = df.groupby('job_title')['salary'].mean().sort_values(ascending=False).head(10)
top_jobs.plot(kind='barh')
plt.title("Top 10 Highest Paying Jobs")

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")

plt.figure(figsize=(20,18))

palette = sns.color_palette("Set2")

plt.subplot(4,3,1)
sns.histplot(df['salary'], kde=True, color=palette[0])
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")

plt.subplot(4,3,2)
sns.scatterplot(x='experience_years', y='salary', data=df, color=palette[1])
plt.title("Experience vs Salary")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")

plt.subplot(4,3,3)
sns.boxplot(x='exp_level', y='salary', data=df, palette="Set3")
plt.title("Salary by Level")
plt.xlabel("Experience Level")
plt.ylabel("Salary")

plt.subplot(4,3,4)
df.groupby('exp_level')['salary'].mean().plot(kind='bar', color='skyblue')
plt.title("Average Salary by Level")
plt.xlabel("Experience Level")
plt.ylabel("Average Salary")

plt.subplot(4,3,5)
sns.countplot(x='exp_level', data=df, palette="pastel")
plt.title("Employee Count by Level")
plt.xlabel("Experience Level")
plt.ylabel("Count")

plt.subplot(4,3,6)
sns.violinplot(x='exp_level', y='salary', data=df, palette="coolwarm")
plt.title("Salary Distribution (Violin)")
plt.xlabel("Experience Level")
plt.ylabel("Salary")

plt.subplot(4,3,7)
sns.regplot(x='experience_years', y='salary', data=df, color='green')
plt.title("Regression Analysis")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.subplot(4,3,8)
sns.kdeplot(df['salary'], fill=True, color='purple')
plt.title("Salary Density")
plt.xlabel("Salary")

plt.subplot(4,3,9)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")

plt.subplot(4,3,10)
sns.lineplot(x='experience_years', y='salary', data=df, color='orange')
plt.title("Salary Trend")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.subplot(4,3,11)
df.groupby('experience_years')['salary'].mean().plot(kind='area', color='lightgreen')
plt.title("Average Salary Growth")
plt.xlabel("Experience")
plt.ylabel("Salary")

# 1️⃣2️⃣ Top Jobs
plt.subplot(4,3,12)
df.groupby('job_title')['salary'].mean().sort_values().tail(10).plot(kind='barh', color='teal')
plt.title("Top Paying Jobs")
plt.xlabel("Salary")
plt.ylabel("Job Title")

# Layout fix
plt.tight_layout()
plt.show()
