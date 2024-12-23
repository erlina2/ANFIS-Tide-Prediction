import streamlit as st
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import traceback
import logging

# Konfigurasi logging
logging.basicConfig(filename='streamlit_app.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Rebuild ANFIS model class
class ANFIS:
    def __init__(self, c1_initial, c2_initial, a1_initial, a2_initial):
        self.c11, self.c21 = c1_initial[0], c1_initial[1]
        self.c12, self.c22 = c2_initial[0], c2_initial[1]
        self.a11, self.a21 = a1_initial[0], a1_initial[1]
        self.a12, self.a22 = a2_initial[0], a2_initial[1]
        self.consequences = None

    def bell_membership(self, x, mean, std_dev):
        return 1 / (1 + ((x - mean) / std_dev) ** 2)

    def layer1_membership(self, x1, x2):
        µA1 = self.bell_membership(x1, self.c11, self.a11)
        µA2 = self.bell_membership(x1, self.c21, self.a21)
        µB1 = self.bell_membership(x2, self.c12, self.a12)
        µB2 = self.bell_membership(x2, self.c22, self.a22)
        return µA1, µA2, µB1, µB2

    def layer2_firing_strength(self, µA1, µA2, µB1, µB2):
        w1 = µA1 * µB1
        w2 = µA2 * µB2
        return np.array([w1, w2])

    def layer3_normalize(self, firing_strengths):
        w_sum = np.sum(firing_strengths)
        return firing_strengths / (w_sum + 1e-10)

    def layer4_consequent(self, norm_firing_strengths, x1, x2):
        if self.consequences is None:
            return np.zeros(len(norm_firing_strengths))

        outputs = []
        for i, norm_w in enumerate(norm_firing_strengths):
            p, q, r = self.consequences[i]
            outputs.append(norm_w * (p * x1 + q * x2 + r))
        return np.array(outputs)

    def layer5_output(self, rule_outputs):
        return np.sum(rule_outputs)

    def predict_single(self, x1, x2):
        µA1, µA2, µB1, µB2 = self.layer1_membership(x1, x2)
        firing_strengths = self.layer2_firing_strength(µA1, µA2, µB1, µB2)
        norm_firing_strengths = self.layer3_normalize(firing_strengths)
        rule_outputs = self.layer4_consequent(norm_firing_strengths, x1, x2)
        return self.layer5_output(rule_outputs)

# Function to load parameters and rebuild the model
def load_model(model_path='anfis_model.pkl'):
    """Load model parameters and rebuild ANFIS model"""
    try:
        with open(model_path, 'rb') as file:
            params = dill.load(file)
        
        # Rebuild model
        loaded_anfis = ANFIS(
            c1_initial=params['c1_initial'],
            c2_initial=params['c2_initial'],
            a1_initial=params['a1_initial'],
            a2_initial=params['a2_initial']
        )
        loaded_anfis.consequences = params['consequences']
        
        return loaded_anfis
    except Exception as e:
        logging.error(f"Error loading model parameters: {e}")
        st.error(f"Gagal memuat parameter model: {e}")
        return None

def single_prediction(model, x1, x2):
    """Predict single value using ANFIS model with error handling"""
    try:
        if model is None:
            st.error("Model tidak tersedia.")
            return None
        prediction = model.predict_single(x1, x2)
        return prediction
    except Exception as e:
        logging.error(f"Error in single prediction: {e}")
        st.error(f"Gagal melakukan prediksi: {e}")
        st.error(traceback.format_exc())
        return None

def multi_step_prediction(model, x1_start, x2_start, steps=24):
    """
    Predict multiple steps into the future using the ANFIS model.
    
    Parameters:
    - model: Trained ANFIS model.
    - x1_start, x2_start: Initial inputs for prediction (last observed values).
    - steps: Number of hours to predict into the future.
    
    Returns:
    - A list of predictions for the next `steps` hours.
    """
    try:
        predictions = []
        x1, x2 = x1_start, x2_start  # Initialize with starting values

        for _ in range(steps):
            # Predict the next value
            predicted = model.predict_single(x1, x2)
            predictions.append(predicted)

            # Shift inputs for the next prediction
            x1, x2 = x2, predicted  # Update x1 and x2 for next step

        return predictions
    except Exception as e:
        logging.error(f"Error in multi-step prediction: {e}")
        raise
    
def predict_from_file(model, data, steps):
    try:
        # Pastikan kolom Value ada
        if 'Value' not in data.columns:
            raise ValueError("Kolom 'Value' tidak ditemukan dalam file CSV.")
        
        # Ambil nilai aktual
        actual_values = data['Value'].values

        # Generate x1 dan x2 dari actual_values
        x1_values = actual_values[:-2]  # Nilai 2 jam sebelumnya
        x2_values = actual_values[1:-1]  # Nilai 1 jam sebelumnya
        actual_values = actual_values[2:]  # Nilai aktual yang akan dibandingkan

        # Batasi jumlah prediksi sesuai rentang
        predictions = []
        for i in range(min(steps, len(actual_values))):
            x1, x2 = x1_values[i], x2_values[i]
            pred = model.predict_single(x1, x2)
            predictions.append(pred)

        # Sesuaikan nilai aktual untuk membandingkan
        actual_subset = actual_values[:len(predictions)]
        differences = np.abs(np.array(predictions) - actual_subset)
        percentage_errors = (differences / actual_subset) * 100
        mean_error = np.mean(percentage_errors)

        return predictions, differences, percentage_errors, mean_error
    except ValueError as ve:
        st.error(str(ve))
        return None, None, None, None
    except Exception as e:
        logging.error(f"Error in prediction from file: {e}")
        st.error(f"Gagal melakukan prediksi dari file: {e}")
        return None, None, None, None


def plot_predictions(actual, predicted):
    """Create a plot comparing actual and predicted values"""
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(actual, label='Actual', color='blue')
        plt.plot(predicted, label='Predicted', color='red', linestyle='--')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        logging.error(f"Error plotting predictions: {e}")
        st.error(f"Gagal membuat plot: {e}")
        return None

def main():
    st.set_page_config(page_title='ANFIS Tide Prediction', page_icon='🌊')
    
    # Load the model with error handling
    try:
        model = load_model('anfis_model.pkl')  # Update to load parameters
    except Exception as e:
        st.error(f"Fatal error loading model: {e}")
        logging.error(f"Fatal error loading model: {e}")
        st.stop()
    
    st.title('ANFIS Tide Level Prediction 🌊')
    st.sidebar.header('Prediction Options')
    
    # Sidebar for selecting prediction mode
    prediction_mode = st.sidebar.radio(
        'Select Prediction Mode', 
        ['Single Input', 'Multi-Step Forecast', 'File Prediction']
    )

    if prediction_mode == 'File Prediction':
        st.header('File-Based Prediction')

        uploaded_file = st.file_uploader("Upload your file (CSV format)", type=['csv'])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(data.head())

                forecast_steps = st.selectbox('Select Forecasting Duration', [12, 24, 36, 48], index=0)

                if st.button('Predict from File'):
                    predictions, differences, percentage_errors, mean_error = predict_from_file(model, data, forecast_steps)

                    if predictions is not None:
                        result_df = pd.DataFrame({
                            'Actual': data['Value'].values[2:len(predictions)+2],
                            'Predicted': predictions,
                            'Difference': differences,
                            'Percentage Error (%)': percentage_errors
                        })
                        st.dataframe(result_df)
                        st.success(f'Average Percentage Error: {mean_error:.2f}%')

                        # Tambahkan tombol unduh
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        st.download_button(
                            label="Download File Prediction Results as CSV",
                            data=csv_data,
                            file_name="file_prediction_results.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}")

    if prediction_mode == 'Single Input':
        st.header('Single Input Prediction')

        # Input untuk prediksi tunggal
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input('Input 2 hours before (x1)', value=2.3, step=0.1, format="%.2f")
        with col2:
            x2 = st.number_input('Input 1 hour before (x2)', value=2.5, step=0.1, format="%.2f")

        actual_value = st.number_input('Actual Tide Level (if available)', value=None, format="%.2f", help="Leave empty if not available")

        if st.button('Predict'):
            try:
                prediction = single_prediction(model, x1, x2)
                if prediction is not None:
                    st.success(f'Predicted Tide Level: {prediction:.3f}')

                if actual_value:
                    difference = prediction - actual_value
                    percentage_error = (abs(difference) / actual_value) * 100
                    st.info(f"Difference: {difference:.3f}")
                    st.info(f"Percentage Error: {percentage_error:.2f}%")
                    
                    # Buat DataFrame hasil prediksi
                    result_df = pd.DataFrame({
                        'x1': [x1],
                        'x2': [x2],
                        'Predicted': [prediction],
                    })
                    if actual_value:
                        difference = prediction - actual_value
                        percentage_error = (abs(difference) / actual_value) * 100
                        
                        # Masukkan nilai ke DataFrame dengan pembulatan
                        result_df['Actual'] = [round(actual_value, 3)]
                        result_df['Difference'] = [round(difference, 3)]
                        result_df['Percentage Error (%)'] = [round(percentage_error, 2)]

                    st.dataframe(result_df)

                    # Tambahkan tombol unduh
                    csv_buffer = io.StringIO()
                    result_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    st.download_button(
                        label="Download Single Prediction as CSV",
                        data=csv_data,
                        file_name="single_prediction.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

    if prediction_mode == 'Multi-Step Forecast':
        st.header('Multi-Step Forecasting')

        col1, col2 = st.columns(2)
        with col1:
            x1_start = st.number_input('Input 2 hours before (x1)', value=2.3, step=0.1, format="%.2f")
        with col2:
            x2_start = st.number_input('Input 1 hour before (x2)', value=2.5, step=0.1, format="%.2f")

        forecast_steps = st.selectbox('Select Forecasting Duration', [4, 8, 12, 24], index=0)

        if st.button('Predict Future Values'):
            try:
                future_predictions = multi_step_prediction(model, x1_start, x2_start, steps=forecast_steps)

                if future_predictions:
                    forecast_df = pd.DataFrame({
                        'Hour': range(1, forecast_steps + 1),
                        'Predicted Value': [round(p, 3) for p in future_predictions]
                    })
                    st.dataframe(forecast_df)

                    # Tambahkan tombol unduh
                    csv_buffer = io.StringIO()
                    forecast_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    st.download_button(
                        label="Download Multi-Step Forecast as CSV",
                        data=csv_data,
                        file_name="multi_step_forecast.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

if __name__ == '__main__':
    main()