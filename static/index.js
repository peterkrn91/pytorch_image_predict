document.getElementById('upload-form').addEventListener('submit', async function (e) {
  e.preventDefault();
  const fileInput = document.getElementById('image');
  if (!fileInput.files.length) {
    alert('Please upload an image!');
    return;
  }

  const formData = new FormData();
  formData.append('image', fileInput.files[0]);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const result = await response.json();
      document.getElementById('result-container').classList.remove('hidden');
      document.getElementById('prediction').textContent = result.prediction;
    } else {
      alert('Failed to process the image. Please try again.');
    }
  } catch (error) {
    console.error('Error:', error);
    alert('An error occurred. Please try again later.');
  }
});
