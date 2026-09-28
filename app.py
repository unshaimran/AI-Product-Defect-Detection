import streamlit as st
from PIL import Image
from transformers import pipeline

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Product Defect Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI Product Defect Detection System")
st.write("Upload a product image for AI-powered defect analysis.")


# -----------------------------
# Load AI Vision Model
# -----------------------------
@st.cache_resource
def load_model():
    model = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )
    return model


classifier = load_model()


# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "📷 Upload Product Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Product",
        use_container_width=True
    )

    # -----------------------------
    # Analyze Product
    # -----------------------------
    if st.button("🔍 Analyze Product"):

        with st.spinner("AI is analyzing the product..."):

            candidate_labels = [
                "a normal product",
                "a scratched product",
                "a cracked product",
                "a damaged product",
                "a defective product"
            ]

            results = classifier(
                image,
                candidate_labels=candidate_labels
            )

        # -----------------------------
        # Main Result
        # -----------------------------
        st.subheader("🤖 AI Defect Analysis")

        top_result = results[0]

        label = top_result["label"]
        score = top_result["score"] * 100

        st.write(f"### Prediction: {label}")
        st.write(f"### Confidence: {score:.2f}%")

        # -----------------------------
        # Status
        # -----------------------------
        if "normal" in label.lower():

            st.success("🟢 Product appears normal.")

        else:

            st.error("🔴 Possible product defect detected.")

        # -----------------------------
        # Confidence Progress Bar
        # -----------------------------
        st.subheader("📈 Confidence Level")

        st.progress(int(score))

        if score >= 80:
            st.success("🟢 High confidence")
        elif score >= 50:
            st.warning("🟡 Medium confidence")
        else:
            st.error("🔴 Low confidence")

        # -----------------------------
        # Other Results
        # -----------------------------
        st.divider()

        st.subheader("📊 Other Possibilities")

        for result in results[1:]:

            result_label = result["label"]
            result_score = result["score"] * 100

            st.write(
                f"**{result_label}** — {result_score:.2f}%"
            )

        # -----------------------------
        # Final Defect Status
        # -----------------------------
        st.divider()

        st.subheader("📌 Final Status")

        if "normal" in label.lower():

            st.info(
                "🟢 Product looks normal."
            )

        else:

            st.warning(
                "🔴 Possible defect found. "
                "Manual inspection is recommended."
            )

        # -----------------------------
        # Analysis Summary
        # -----------------------------
        st.subheader("📝 Analysis Summary")

        st.write(
            f"The AI model classified the uploaded image as "
            f"**{label}** with a confidence score of "
            f"**{score:.2f}%**."
        )

        # -----------------------------
        # Analysis Completed
        # -----------------------------
        st.success("✅ Image analysis completed!")

        # -----------------------------
        # Download Defect Report
        # -----------------------------
        st.divider()

        st.subheader("📄 Download Analysis Report")

        report = f"""
AI PRODUCT DEFECT DETECTION REPORT
===================================

Prediction: {label}
Confidence: {score:.2f}%

Analysis Results:
"""

        for result in results:

            result_label = result["label"]
            result_score = result["score"] * 100

            report += (
                f"- {result_label}: "
                f"{result_score:.2f}%\n"
            )

        if "normal" in label.lower():

            report += (
                "\nFinal Status: "
                "Product appears normal."
            )

        else:

            report += (
                "\nFinal Status: "
                "Possible product defect detected."
            )

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="product_defect_report.txt",
            mime="text/plain"
        )