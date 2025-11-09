const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const submitBtn = document.getElementById('submitBtn');
const uploadForm = document.getElementById('uploadForm');
const loader = document.getElementById('loader');

// Drag and drop functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

uploadForm.addEventListener('submit', (e) => {
    submitBtn.style.display = 'none';
    loader.style.display = 'block';
});

function handleFile(file) {
    // Check file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG, JPEG, or PNG)');
        return;
    }
    
    // Check file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size should not exceed 16MB');
        return;
    }
    
    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
    
    // Set file to input
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
}

function resetUpload() {
    uploadArea.style.display = 'block';
    previewArea.style.display = 'none';
    imagePreview.src = '';
    fileInput.value = '';
    submitBtn.disabled = true;
}
