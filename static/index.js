const videoElement = document.getElementById('video');
const canvas = document.createElement('canvas');
const context = canvas.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    videoElement.srcObject = stream;
    videoElement.play();
  })
  .catch((err) => {
    alert('Error accessing webcam: ' + err);
  });


canvas.width = 640;  
canvas.height = 480; 

async function captureAndPredict() {
  context.clearRect(0, 0, canvas.width, canvas.height);

  context.save(); 
  context.scale(-1, 1); 
  context.drawImage(videoElement, -canvas.width, 0, canvas.width, canvas.height); 
  context.restore(); 

  const imageData = canvas.toDataURL('image/png');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: JSON.stringify({ image: imageData }),
      headers: {
        'Content-Type': 'application/json'
      }
    });
    if (response.ok) {
      const result = await response.json();
      document.getElementById('result-container').classList.remove('hidden');
      document.getElementById('prediction').textContent = result.prediction;
    } else {
      console.error('Failed to process the image. Please try again.');
    }
  } catch (error) {
    console.error('Error:', error);
  }
  requestAnimationFrame(captureAndPredict);
}
requestAnimationFrame(captureAndPredict);