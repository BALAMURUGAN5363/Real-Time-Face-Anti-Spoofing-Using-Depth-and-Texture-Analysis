// top-level script
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const imageInput = document.getElementById("image");

// UI elements for real-time status
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const badgeEl = document.getElementById("badge");

// internal state
let detecting = false;
let loopTimer = null;

// 1️⃣ Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.onplaying = () => {
            // ensure canvas has correct size
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
        };
        setStatus("idle", "Camera ready. Click Start Live Verify to begin.");
    })
    .catch(err => {
        alert("Camera access denied!");
        console.error(err);
        setStatus("error", "Camera access denied. Please allow and retry.");
    });

// status helper
function setStatus(kind, text) {
    if (!badgeEl || !statusEl) return;
    badgeEl.className = "badge badge-" + kind;
    badgeEl.textContent = kind.toUpperCase();
    statusEl.textContent = text;
}

// draw current frame to base64
function snapshotToBase64() {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg");
}

// start/stop live detection
async function tickOnce() {
    try {
        const dataURL = snapshotToBase64();
        const formData = new FormData();
        formData.append("image", dataURL);

        const res = await fetch("/verify", { method: "POST", body: formData });
        const html = await res.text();

        if (html.includes("Access Granted")) {
            setStatus("real", "Real face detected. Redirecting to exam...");
            stopDetection();
            // Render server's response (exam page) directly
            document.open(); document.write(html); document.close();
            return;
        }
        if (html.includes("Access Blocked")) {
            setStatus("spoof", "Spoof detected. Hold still and retry.");
            return;
        }
        if (html.includes("No face detected")) {
            setStatus("warn", "No face detected. Center your face and improve lighting.");
            return;
        }
        // default fallback
        setStatus("idle", "Analyzing...");
    } catch (e) {
        console.error(e);
        setStatus("error", "Network error. Check connection and retry.");
    }
}

function startDetection() {
    if (detecting) return;
    detecting = true;
    setStatus("idle", "Starting live verification...");
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;

    // tick every 1200ms to avoid spamming the backend
    loopTimer = setInterval(tickOnce, 1200);
}

function stopDetection() {
    detecting = false;
    if (loopTimer) clearInterval(loopTimer);
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    setStatus("idle", "Live verification stopped.");
}

// wire buttons if present
startBtn?.addEventListener("click", startDetection);
stopBtn?.addEventListener("click", stopDetection);

// 2️⃣ Capture image and send to server (manual single-shot)
function capture() {
    // Set canvas size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to Base64 image
    const dataURL = canvas.toDataURL("image/jpeg");

    // Put image into hidden input
    imageInput.value = dataURL;

    // Submit form to Flask (/verify)
    document.getElementById("form").submit();
}
