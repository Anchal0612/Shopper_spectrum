import streamlit as st
import pandas as pd
import joblib
from scipy import sparse
import pickle

st.set_page_config(page_title="Shopper Spectrum", layout="centered")

# 🎯 Title & Description
st.title("🛍️ Shopper Spectrum")
st.write("A smart retail analytics tool using RFM & Collaborative Filtering")

# ✅ Load all assets
@st.cache_resource
def load_all():
    # Load compressed sparse similarity matrix
    sparse_matrix = sparse.load_npz("similarity_sparse_top5.npz")

    # Load row/column labels (product codes)
    with open("similarity_labels.pkl", "rb") as f:
        labels = pickle.load(f)

    # Convert to DataFrame
    similarity_df = pd.DataFrame(sparse_matrix.toarray(), index=labels['index'], columns=labels['columns'])

    # Load product name dictionary
    product_names = pd.read_pickle("product_names.pkl")

    # Load clustering model and scaler
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")

    return similarity_df, product_names, kmeans, scaler

# Load data
similarity, product_names, kmeans, scaler = load_all()

# Tabs for navigation
tab1, tab2 = st.tabs(["🔍 Product Recommendation", "👥 Customer Segment Prediction"])

# ---------------------------------------------
# 🔍 Tab 1: Product Recommendation
# ---------------------------------------------
with tab1:
    st.subheader("🔍 Product Recommendation Engine")

    product_id = st.text_input("Enter Product Code (e.g., 85123A)")

    if st.button("Get Recommendations"):
        if product_id in similarity.columns:
            top_products = similarity[product_id].sort_values(ascending=False)[1:6]

            st.success(f"Top 5 similar products to **{product_id}**:")
            for code in top_products.index:
                name = product_names.get(code, "Unknown")
                st.markdown(f"🔹 **{code}** — {name}")
        else:
            st.error("Product ID not found. Please try another one.")

# ---------------------------------------------
# 👥 Tab 2: Customer Segmentation
# ---------------------------------------------
with tab2:
    st.subheader("👥 Customer Segment Predictor (RFM Based)")

    recency = st.number_input("Recency (days ago)", min_value=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1)
    monetary = st.number_input("Monetary Value (total spend)", min_value=1)

    if st.button("Predict Segment"):
        try:
            scaled_data = scaler.transform([[recency, frequency, monetary]])
            cluster = kmeans.predict(scaled_data)[0]

            segment_names = {
                0: "High-Value",
                1: "At-Risk",
                2: "Regular",
                3: "Occasional"
            }

            segment = segment_names.get(cluster, f"Segment {cluster}")
            st.success(f"Predicted Segment: **{segment}**")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
