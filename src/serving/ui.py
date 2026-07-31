"""Gradio UI for the Telco churn prediction model."""
import gradio as gr
from src.serving.inference import predict_churn

INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
YES_NO = ["Yes", "No"]
YES_NO_PHONE = ["Yes", "No", "No phone service"]
YES_NO_INTERNET = ["Yes", "No", "No internet service"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


def predict_from_form(gender, senior_citizen, partner, dependents, tenure,
                       phone_service, multiple_lines, internet_service,
                       online_security, online_backup, device_protection,
                       tech_support, streaming_tv, streaming_movies,
                       contract, paperless_billing, payment_method,
                       monthly_charges, total_charges):
    customer = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }
    result = predict_churn(customer)
    explanation = (
        f"{result['churn_prediction']} "
        f"(probability {result['churn_probability']:.2%} vs. "
        f"decision threshold {result['threshold_used']:.0%})"
    )
    return (
        explanation,
        f"{result['churn_probability']:.2%}",
    )


def build_ui():
    with gr.Blocks(title="Telco Customer Churn Predictor") as demo:
        gr.Markdown("# Telco Customer Churn Predictor")
        gr.Markdown("Enter customer details to predict churn likelihood.")

        with gr.Row():
            with gr.Column():
                gender = gr.Dropdown(["Female", "Male"], label="Gender", value="Female")
                senior_citizen = gr.Dropdown([0, 1], label="Senior Citizen", value=0)
                partner = gr.Dropdown(YES_NO, label="Partner", value="No")
                dependents = gr.Dropdown(YES_NO, label="Dependents", value="No")
                tenure = gr.Number(label="Tenure (months)", value=12)
                phone_service = gr.Dropdown(YES_NO, label="Phone Service", value="Yes")
                multiple_lines = gr.Dropdown(YES_NO_PHONE, label="Multiple Lines", value="No")

            with gr.Column():
                internet_service = gr.Dropdown(INTERNET_SERVICES, label="Internet Service", value="Fiber optic")
                online_security = gr.Dropdown(YES_NO_INTERNET, label="Online Security", value="No")
                online_backup = gr.Dropdown(YES_NO_INTERNET, label="Online Backup", value="No")
                device_protection = gr.Dropdown(YES_NO_INTERNET, label="Device Protection", value="No")
                tech_support = gr.Dropdown(YES_NO_INTERNET, label="Tech Support", value="No")
                streaming_tv = gr.Dropdown(YES_NO_INTERNET, label="Streaming TV", value="No")
                streaming_movies = gr.Dropdown(YES_NO_INTERNET, label="Streaming Movies", value="No")

            with gr.Column():
                contract = gr.Dropdown(CONTRACTS, label="Contract", value="Month-to-month")
                paperless_billing = gr.Dropdown(YES_NO, label="Paperless Billing", value="Yes")
                payment_method = gr.Dropdown(PAYMENT_METHODS, label="Payment Method", value="Electronic check")
                monthly_charges = gr.Number(label="Monthly Charges ($)", value=70.35)
                total_charges = gr.Number(label="Total Charges ($)", value=845.50)

        submit_btn = gr.Button("Predict", variant="primary")

        with gr.Row():
            prediction_output = gr.Textbox(label="Prediction")
            probability_output = gr.Textbox(label="Churn Probability")

        submit_btn.click(
            fn=predict_from_form,
            inputs=[gender, senior_citizen, partner, dependents, tenure,
                    phone_service, multiple_lines, internet_service,
                    online_security, online_backup, device_protection,
                    tech_support, streaming_tv, streaming_movies,
                    contract, paperless_billing, payment_method,
                    monthly_charges, total_charges],
            outputs=[prediction_output, probability_output],
        )

    return demo