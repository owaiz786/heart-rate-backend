from flask import Flask, request, jsonify
import numpy as np
from scipy.signal import butter, lfilter, periodogram, detrend

app = Flask(__name__)

def bandpass_filter(data, lowcut=0.75, highcut=3.0, fs=30.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

@app.route('/analyze', methods=['POST'])
def analyze():
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

if __name__ == '__main__':
    app.run(debug=True)
