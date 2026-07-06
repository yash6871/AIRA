/*
 * Reusable browser camera helper.
 * Works identically on laptop (webcam) and mobile (front/back camera) —
 * it's just the standard getUserMedia API, no native Python camera code needed.
 * NOTE: mobile browsers require HTTPS to allow camera access (Render gives free HTTPS).
 */
async function startCamera(videoEl, facingMode = "user") {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
    });
    videoEl.srcObject = stream;
    await videoEl.play();
    return stream;
}

function captureFrame(videoEl) {
    const canvas = document.createElement("canvas");
    canvas.width = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.9);
}

function stopCamera(stream) {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
    }
}
