import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# GitHub raw CSV URL
DATA_URL = 'clean_df.csv'

@st.cache
def load_data():
    """Fetch data from GitHub and cache it."""
    return pd.read_csv(DATA_URL)

# Main app
def main():
    st.title("Automobile Data Analysis")
    st.markdown("This app analyzes automobile data from a GitHub repository and visualizes relationships in the dataset.")

    # Load data
    df = load_data()

    # Display data
    st.subheader("Dataset Overview")
    st.dataframe(df)

    # Data types
    st.subheader("Data Types")
    st.write(df.dtypes)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Scatterplot (engine-size vs price)
    st.subheader("Scatterplot: Engine Size vs Price")
    fig, ax = plt.subplots()
    sns.regplot(x="engine-size", y="price", data=df, ax=ax)
    st.pyplot(fig)

    # Boxplot for categorical variables
    st.subheader("Boxplot: Drive Wheels vs Price")
    fig, ax = plt.subplots()
    sns.boxplot(x="drive-wheels", y="price", data=df, ax=ax)
    st.pyplot(fig)

    # Grouping example
    st.subheader("Average Price by Body Style")
    avg_price_by_body = df.groupby('body-style', as_index=False)['price'].mean()
    st.dataframe(avg_price_by_body)

if __name__ == "__main__":
    main()
