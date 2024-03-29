# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:16:46 2024

@author: Laptop
"""
import streamlit as st
import pandas as pd
import altair as alt

def plot_cumulative(df):
    try:
        # Check if 'total' and 'reduced' columns exist in the DataFrame
        if 'total' not in df.columns or 'reduced' not in df.columns:
            raise ValueError("CSV should have 'total' and 'reduced' columns.")

        # Check if the 'total' and 'reduced' columns have the same length
        if len(df['total']) != len(df['reduced']):
            raise ValueError("Something was wrong with the previous step of the analysis. The columns have different lengths.")

        # Create a cumulative sum of the columns
        df['total_cumulative'] = df['total'].cumsum()
        df['reduced_cumulative'] = df['reduced'].cumsum()

        # Calculate the ITS metric
        last_total_cumulative = df['total_cumulative'].iloc[-1]
        last_reduced_cumulative = df['reduced_cumulative'].iloc[-1]
        its_metric = last_reduced_cumulative / last_total_cumulative
        
        total=str(last_total_cumulative)
        reduced=str(last_reduced_cumulative)

        # Display the ITS metric
        st.write(f"ITS (metric)= reduced_codes/total_codes = "+reduced+"/"+total+" = "+str(its_metric))

        # Plot two connected scatterplots on the same set of axes with legend
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('observation:O', title='Observations', axis=alt.Axis(labelAngle=0)),
            y='total_cumulative:Q',
            color=alt.value('blue'),
            tooltip=['total_cumulative:Q']
        ).properties(
            width=600,
            height=400
        )

        chart += alt.Chart(df).mark_line(point=True, color='orange').encode(
            x=alt.X('observation:O', title='Observations', axis=alt.Axis(labelAngle=0)),
            y='reduced_cumulative:Q',
            color=alt.value('orange'),  # Specify color for the 'reduced' line
            tooltip=['reduced_cumulative:Q']
        )

        chart = chart.encode(
            color=alt.Color('legend:N', scale=alt.Scale(domain=['total', 'reduced'], range=['blue', 'orange']), title='Legend')
        )

        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.header(":orange[Inductive Thematic Saturation]")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the first few rows of the DataFrame
        st.dataframe(df)

        # Add observation column starting from 1
        df['observation'] = range(1, len(df) + 1)

        # Plot cumulative sums
        plot_cumulative(df)

if __name__ == "__main__":
    main()
