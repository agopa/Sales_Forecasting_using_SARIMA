import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import chardet

st.set_page_config(page_title="Forecast Sales from Orders", layout="centered")
st.title("📦 SARIMA Forecast: Order-Based Sales Data")
st.write("Upload your order data CSV and get a monthly sales forecast.")

uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file:
    try:
        # Detect encoding
        raw_data = uploaded_file.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding'] if detected['encoding'] and detected['encoding'].lower() not in ['johab', None] else 'ISO-8859-1'
        uploaded_file.seek(0)

        # Read CSV
        df = pd.read_csv(uploaded_file, encoding=encoding)

        # Check required columns
        required_cols = {'ORDERDATE', 'SALES'}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ Your file must include the columns: {required_cols}")
            st.stop()

        # Parse date and convert to month
        df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
        df = df.dropna(subset=['ORDERDATE'])  # Drop rows with bad dates
        df['Month'] = df['ORDERDATE'].dt.to_period('M').dt.to_timestamp()
        
        # Group by month and sum sales
        monthly_sales = df.groupby('Month')['SALES'].sum().reset_index()
        monthly_sales.set_index('Month', inplace=True)

        st.success("✅ Processed and aggregated monthly sales.")
        st.line_chart(monthly_sales)

        # SARIMA Model
        with st.spinner("Training SARIMA model..."):
            model = SARIMAX(monthly_sales['SALES'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            results = model.fit(disp=False)

        # Forecast next 12 months
        forecast_steps = 12
        forecast = results.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=monthly_sales.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Show forecast
        st.subheader("📅 Forecast (Next 12 Months)")
        forecast_df = pd.DataFrame({'Forecast': forecast_values}, index=forecast_index)
        st.write(forecast_df)

        # Plot forecast
        st.subheader("📈 Forecast Plot")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_sales.index, monthly_sales['SALES'], label='Historical', color='blue')
        ax.plot(forecast_index, forecast_values, label='Forecast', color='green')
        ax.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='lightgreen', alpha=0.3)
        ax.set_title("Monthly Sales Forecast (SARIMA)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("👈 Please upload your order CSV to get started.")


# Download forecast CSV
st.download_button(
    label="📥 Download Forecast as CSV",
    data=forecast_df.to_csv().encode('utf-8'),
    file_name="forecast_output.csv",
    mime="text/csv"
)
