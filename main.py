import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for Render
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, periodogram, detrend
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

# ✅ Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Input model using Pydantic
class SignalData(BaseModel):
    green_signal: list[float]
    fs: float = 30.0

# ✅ Signal processing function
def bandpass_filter(data, lowcut=0.75, highcut=3.0, fs=30.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

@app.get("/")
def index():
    return {"message": "✅ FastAPI Heart Rate Monitor is running."}

@app.post("/analyze")
async def analyze(data: SignalData):
    signal = np.array(data.green_signal, dtype=np.float32)
    fs = data.fs

    if len(signal) < fs * 3:
        return {"error": "Not enough data"}

    # Process signal
    detrended = detrend(signal)
    filtered = bandpass_filter(detrended, fs=fs)
    f, Pxx = periodogram(filtered, fs)
    valid = (f >= 0.75) & (f <= 3.0)

    if np.sum(valid) == 0:
        return {"error": "No valid peak"}

    peak_freq = f[valid][np.argmax(Pxx[valid])]
    bpm = peak_freq * 60

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(signal, color='green')
    ax1.set_title("Raw Green Signal")

    ax2.plot(filtered, color='red')
    ax2.set_title("Filtered Signal (0.75–3.0 Hz)")

    ax3.plot(f * 60, Pxx, color='blue')
    ax3.axvline(bpm, color='orange', linestyle='--', label=f'Peak: {bpm:.1f} BPM')
    ax3.set_xlim([40, 180])
    ax3.set_title("Power Spectrum")
    ax3.legend()

    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)

    return StreamingResponse(img, media_type="image/png")
