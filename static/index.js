// Clear input field when returning to index.html
var fileInput = document.querySelector('input[type="file"]');
fileInput.value = null;

// Add event listener to the button
document.addEventListener("DOMContentLoaded", function () {
  var button = document.getElementById("upload-btn");
  button.addEventListener("click", handleFileSelect);
});

// Function to handle file selection
function handleFileSelect(event) {
  const fileInput = document.querySelector('input[type="file"]');
  const files = fileInput.files;
  if (files.length > 0) {
    const fileName = files[0].name;
    alert("Selected file: " + fileName);
  } else {
    alert("No file selected.");
    event.preventDefault(); // Prevent form submission
  }
}
