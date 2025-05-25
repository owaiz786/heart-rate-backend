
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, periodogram, detrend
import io

app = Flask(__name__)
CORS(app)

def bandpass_filter(data, lowcut=0.75, highcut=3.0, fs=30.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

@app.route('/', methods=['GET'])
def index():
    return "✅ Heart Rate Monitor Backend with Graphs is Running"

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        signal = np.array(data['green_signal'], dtype=np.float32)
        fs = float(data.get('fs', 30.0))

        if len(signal) < fs * 3:
            return jsonify({"error": "Not enough data"}), 400

        detrended = detrend(signal)
        filtered = bandpass_filter(detrended, fs=fs)
        f, Pxx = periodogram(filtered, fs)
        valid = (f >= 0.75) & (f <= 3.0)

        if np.sum(valid) == 0:
            return jsonify({"error": "No valid peak"}), 400

        peak_freq = f[valid][np.argmax(Pxx[valid])]
        bpm = peak_freq * 60

        # --- Generate Plot ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

        ax1.plot(signal, color='green')
        ax1.set_title("Raw Green Signal")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Intensity")

        ax2.plot(filtered, color='red')
        ax2.set_title("Filtered Signal (0.75–3.0 Hz)")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Amplitude")

        ax3.plot(f * 60, Pxx, color='blue')
        ax3.axvline(bpm, color='orange', linestyle='--', label=f'Peak: {bpm:.1f} BPM')
        ax3.set_xlim([40, 180])
        ax3.set_title("Power Spectrum")
        ax3.set_xlabel("Frequency (BPM)")
        ax3.set_ylabel("Power")
        ax3.legend()

        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
