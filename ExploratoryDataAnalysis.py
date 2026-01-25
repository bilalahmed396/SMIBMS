import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
data_path = '/kaggle/input/twitter-airline-sentiment/'

# Try to find CSV files in the directory
import os
print("Files in dataset directory:")
for root, dirs, files in os.walk(data_path):
    for file in files:
        print(os.path.join(root, file))
print("\n" + "="*80 + "\n")

# Load the main CSV file (usually Tweets.csv)
df = pd.read_csv(data_path + 'Tweets.csv')

print("DATASET STRUCTURE ANALYSIS")
print("="*80)

# 1. Basic Information
print("\n1. BASIC DATASET INFO:")
print(f"   - Total rows: {len(df)}")
print(f"   - Total columns: {len(df.columns)}")
print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2. Column Names and Types
print("\n2. COLUMN INFORMATION:")
print(df.dtypes)

# 3. First few rows
print("\n3. FIRST 5 ROWS:")
print(df.head())

# 4. Statistical Summary
print("\n4. STATISTICAL SUMMARY:")
print(df.describe(include='all'))

# 5. Missing Values
print("\n5. MISSING VALUES:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Percentage': missing_pct.values
})
print(missing_df[missing_df['Missing_Count'] > 0])

# 6. Target Variable Distribution (sentiment)
print("\n6. SENTIMENT DISTRIBUTION:")
if 'airline_sentiment' in df.columns:
    sentiment_counts = df['airline_sentiment'].value_counts()
    print(sentiment_counts)
    print(f"\nPercentages:")
    print(df['airline_sentiment'].value_counts(normalize=True) * 100)

# 7. Text Length Analysis
print("\n7. TEXT LENGTH ANALYSIS:")
if 'text' in df.columns:
    df['text_length'] = df['text'].astype(str).apply(len)
    df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

    print(f"   - Average text length: {df['text_length'].mean():.2f} characters")
    print(f"   - Average word count: {df['word_count'].mean():.2f} words")
    print(f"   - Max text length: {df['text_length'].max()} characters")
    print(f"   - Min text length: {df['text_length'].min()} characters")

# 8. Sample Texts from Each Sentiment
print("\n8. SAMPLE TEXTS FROM EACH SENTIMENT CLASS:")
if 'airline_sentiment' in df.columns and 'text' in df.columns:
    for sentiment in df['airline_sentiment'].unique():
        print(f"\n   {sentiment.upper()}:")
        samples = df[df['airline_sentiment'] == sentiment]['text'].head(2)
        for i, text in enumerate(samples, 1):
            print(f"   {i}. {text[:100]}...")

# 9. Airline Distribution
print("\n9. AIRLINE DISTRIBUTION:")
if 'airline' in df.columns:
    print(df['airline'].value_counts())

# 10. Visualizations
print("\n10. GENERATING VISUALIZATIONS...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Sentiment Distribution
if 'airline_sentiment' in df.columns:
    df['airline_sentiment'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    axes[0, 0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Sentiment')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Text Length Distribution
if 'text_length' in df.columns:
    axes[0, 1].hist(df['text_length'], bins=50, color='#95e1d3', edgecolor='black')
    axes[0, 1].set_title('Text Length Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Text Length (characters)')
    axes[0, 1].set_ylabel('Frequency')

# Plot 3: Airline vs Sentiment
if 'airline' in df.columns and 'airline_sentiment' in df.columns:
    pd.crosstab(df['airline'], df['airline_sentiment']).plot(kind='bar', stacked=True, ax=axes[1, 0])
    axes[1, 0].set_title('Sentiment by Airline', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Airline')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend(title='Sentiment')

# Plot 4: Word Count Distribution
if 'word_count' in df.columns:
    axes[1, 1].hist(df['word_count'], bins=30, color='#f38181', edgecolor='black')
    axes[1, 1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Word Count')
    axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('dataset_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("EXPLORATION COMPLETE!")
print("="*80)
print("\nKey Insights to Consider:")
print("1. Check class imbalance in sentiment distribution")
print("2. Note the text length range for model input")
print("3. Consider data cleaning needed (URLs, mentions, etc.)")
print("4. Plan for handling missing values if any")
print("\nNext Steps: Share the output, and I'll create the preprocessing and model code!")
