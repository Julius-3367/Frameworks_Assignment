# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center;}
    .section-header {font-size: 2rem; color: #2ca02c; margin-top: 2rem;}
    .info-text {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    try:
        df = pd.read_csv('data/cleaned_metadata.csv')
        return df
    except:
        st.error("Please run the analysis script first to generate cleaned data")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">CORD-19 Research Dataset Explorer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-text">
    This application explores the COVID-19 Open Research Dataset (CORD-19), containing metadata 
    about scientific papers related to COVID-19 and coronavirus research.
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    year_range = st.sidebar.slider(
        "Select Publication Year Range",
        min_value=int(df['publication_year'].min()),
        max_value=int(df['publication_year'].max()),
        value=(2020, 2021)
    )
    
    min_word_count = st.sidebar.slider(
        "Minimum Abstract Word Count",
        min_value=0,
        max_value=500,
        value=50
    )
    
    # Filter data based on selections
    filtered_df = df[
        (df['publication_year'] >= year_range[0]) & 
        (df['publication_year'] <= year_range[1]) &
        (df['abstract_word_count'] >= min_word_count)
    ]
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Papers", len(filtered_df))
    
    with col2:
        st.metric("Average Abstract Length", f"{filtered_df['abstract_word_count'].mean():.1f} words")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Publications", "Content Analysis", "Sample Data"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Publications by year
            yearly_counts = filtered_df['publication_year'].value_counts().sort_index()
            st.subheader("Publications by Year")
            fig, ax = plt.figure(figsize=(10, 6))
            yearly_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Number of Publications by Year')
            ax.set_xlabel('Year')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            # Top journals
            top_journals = filtered_df['journal'].value_counts().head(10)
            st.subheader("Top 10 Journals")
            fig, ax = plt.figure(figsize=(10, 6))
            top_journals.plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_title('Top Publishing Journals')
            ax.set_xlabel('Journal')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
    
    with tab2:
        st.markdown('<h2 class="section-header">Temporal Analysis</h2>', unsafe_allow_html=True)
        
        # Monthly publications (if publish_time is available)
        if 'publish_time' in filtered_df.columns:
            filtered_df['publish_time'] = pd.to_datetime(filtered_df['publish_time'])
            monthly_data = filtered_df.resample('M', on='publish_time').size()
            
            st.subheader("Monthly Publication Trend")
            fig, ax = plt.figure(figsize=(12, 6))
            monthly_data.plot(ax=ax, color='orange', linewidth=2)
            ax.set_title('Monthly Publication Trend')
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Publications')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with tab3:
        st.markdown('<h2 class="section-header">Content Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Word cloud
            st.subheader("Title Word Cloud")
            titles_text = ' '.join(filtered_df['title'].dropna().astype(str))
            wordcloud = WordCloud(width=600, height=400, background_color='white').generate(titles_text)
            
            fig, ax = plt.figure(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Common Words in Paper Titles')
            st.pyplot(fig)
        
        with col2:
            # Abstract length distribution
            st.subheader("Abstract Length Distribution")
            fig, ax = plt.figure(figsize=(10, 6))
            ax.hist(filtered_df['abstract_word_count'], bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
            ax.set_title('Distribution of Abstract Word Count')
            ax.set_xlabel('Word Count')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
    
    with tab4:
        st.markdown('<h2 class="section-header">Sample Data</h2>', unsafe_allow_html=True)
        
        # Show sample data
        st.subheader("Sample Research Papers")
        sample_data = filtered_df[['title', 'journal', 'publication_year', 'abstract_word_count']].head(10)
        st.dataframe(sample_data)
        
        # Data statistics
        st.subheader("Dataset Statistics")
        st.write(f"**Total papers in current selection:** {len(filtered_df)}")
        st.write(f"**Time range:** {year_range[0]} - {year_range[1]}")
        st.write(f"**Columns available:** {len(filtered_df.columns)}")
        
        # Show raw data option
        if st.checkbox("Show raw data"):
            st.write(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** This analysis uses the CORD-19 metadata from Kaggle. 
    The dataset contains metadata for COVID-19 and coronavirus-related research papers.
    """)

if __name__ == "__main__":
    main()
