# cord19_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
import numpy as np
import os

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data(file_path):
    """Load the CORD-19 metadata dataset"""
    try:
        print("Loading dataset...")
        df = pd.read_csv(file_path, low_memory=False)
        print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def explore_data(df):
    """Explore the dataset structure"""
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    
    # Display basic information
    print("\n1. First 5 rows:")
    print(df.head())
    
    print(f"\n2. Dataset shape: {df.shape}")
    print(f"\n3. Columns: {list(df.columns)}")
    
    print("\n4. Data types:")
    print(df.dtypes)
    
    print("\n5. Missing values summary:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    return missing_values

def clean_data(df):
    """Clean and prepare the data for analysis"""
    print("\n" + "="*60)
    print("DATA CLEANING")
    print("="*60)
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Handle missing values in key columns
    print("Handling missing values...")
    
    # Fill missing abstracts with empty string
    df_clean['abstract'] = df_clean['abstract'].fillna('')
    
    # Extract year from publish_time
    print("Extracting publication year...")
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['publication_year'] = df_clean['publish_time'].dt.year
    
    # Fill missing years with 2020 (most common year for COVID research)
    df_clean['publication_year'] = df_clean['publication_year'].fillna(2020).astype(int)
    
    # Create abstract word count
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()))
    
    # Filter only relevant years for COVID research
    df_clean = df_clean[df_clean['publication_year'] >= 2019]
    
    print(f"✅ Data cleaning completed! New shape: {df_clean.shape}")
    return df_clean

def analyze_data(df):
    """Perform data analysis"""
    print("\n" + "="*60)
    print("DATA ANALYSIS")
    print("="*60)
    
    # 1. Papers by publication year
    yearly_counts = df['publication_year'].value_counts().sort_index()
    print("\n1. Publications by year:")
    print(yearly_counts)
    
    # 2. Top journals
    top_journals = df['journal'].value_counts().head(10)
    print("\n2. Top 10 journals:")
    print(top_journals)
    
    # 3. Basic statistics for abstract word count
    print("\n3. Abstract word count statistics:")
    print(df['abstract_word_count'].describe())
    
    return yearly_counts, top_journals

def create_visualizations(df, yearly_counts, top_journals):
    """Create all required visualizations"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create visuals directory if it doesn't exist
    os.makedirs('visuals', exist_ok=True)
    
    # 1. Publications over time
    plt.figure(figsize=(12, 6))
    yearly_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of COVID-19 Publications by Year', fontsize=16, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Number of Publications')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visuals/publications_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top journals bar chart
    plt.figure(figsize=(12, 6))
    top_journals.plot(kind='bar', color='lightgreen')
    plt.title('Top 10 Journals Publishing COVID-19 Research', fontsize=16, fontweight='bold')
    plt.xlabel('Journal')
    plt.ylabel('Number of Publications')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visuals/top_journals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Word cloud of titles
    print("Generating word cloud...")
    titles_text = ' '.join(df['title'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(titles_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud of Research Paper Titles', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('visuals/title_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Distribution of abstract word count
    plt.figure(figsize=(12, 6))
    plt.hist(df['abstract_word_count'], bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Abstract Word Count', fontsize=16, fontweight='bold')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.xlim(0, 500)  # Limit to reasonable range
    plt.tight_layout()
    plt.savefig('visuals/abstract_wordcount_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ All visualizations saved to 'visuals' folder!")

def main_analysis():
    """Main function to run the analysis"""
    print("Starting CORD-19 Dataset Analysis...")
    
    # Load data (replace with your actual file path)
    file_path = "data/metadata.csv"  # Update this path
    
    df = load_data(file_path)
    if df is None:
        print("Please download the metadata.csv file from Kaggle and place it in the data/ folder")
        print("Download from: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge")
        return
    
    # Explore data
    explore_data(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Analyze data
    yearly_counts, top_journals = analyze_data(df_clean)
    
    # Create visualizations
    create_visualizations(df_clean, yearly_counts, top_journals)
    
    # Save cleaned data for Streamlit app
    df_clean.to_csv('data/cleaned_metadata.csv', index=False)
    print("✅ Cleaned data saved to 'data/cleaned_metadata.csv'")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Key findings:")
    print(f"- Total papers analyzed: {len(df_clean)}")
    print(f"- Time range: {df_clean['publication_year'].min()} to {df_clean['publication_year'].max()}")
    print(f"- Most productive year: {yearly_counts.idxmax()} with {yearly_counts.max()} papers")
    print(f"- Top journal: {top_journals.index[0]} with {top_journals.iloc[0]} papers")

if __name__ == "__main__":
    main_analysis()
