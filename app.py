from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.signal import butter, lfilter, periodogram, detrend

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# --- Signal Processing ---
def bandpass_filter(data, lowcut=0.75, highcut=3.0, fs=30.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# --- Health Check Route ---
@app.route('/', methods=['GET'])
def index():
    return "✅ Heart Rate Monitor Backend is Running"

# --- Heart Rate Analysis ---
@app.route('/analyze', methods=['POST', 'OPTIONS'])  # OPTIONS added for preflight
def analyze():
    if request.method == 'OPTIONS':
        return '', 204  # ✅ Preflight check OK

    try:
        data = request.get_json()
        signal = np.array(data['green_signal'], dtype=np.float32)
        fs = float(data.get('fs', 30.0))

        if len(signal) < fs * 3:
            return jsonify({"error": "Not enough data"}), 400

        filtered = bandpass_filter(detrend(signal), fs=fs)
        f, Pxx = periodogram(filtered, fs)
        valid = (f >= 0.75) & (f <= 3.0)

        if np.sum(valid) == 0:
            return jsonify({"error": "No valid peak"}), 400

        peak_freq = f[valid][np.argmax(Pxx[valid])]
        bpm = peak_freq * 60
        return jsonify({"heart_rate": bpm})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Render Port Binding ---
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
