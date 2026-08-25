document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    
    // Containers
    const previewContainer = document.getElementById('preview-container');
    const spinner = document.getElementById('loading-spinner');
    
    // YOLO section
    const yoloPlaceholder = document.getElementById('yolo-placeholder');
    const yoloResult = document.getElementById('yolo-result');
    
    // CNN section
    const cnnPlaceholder = document.getElementById('cnn-placeholder');
    const cnnResult = document.getElementById('cnn-result');

    // UI Elements
    const errorMsg = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    const resetBtn = document.getElementById('reset-btn');

    // Result display elements
    const origImg = document.getElementById('result-original-img');
    const cropImg = document.getElementById('result-crop-img');
    const plateNumber = document.getElementById('result-plate-number');
    const confidence = document.getElementById('result-confidence');
    const procTime = document.getElementById('result-time');

    // Drag and Drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Click to upload
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFile(this.files[0]);
        }
    });

    function handleFile(file) {
        // Hide errors
        errorMsg.classList.add('hidden');
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
            showError('الرجاء اختيار ملف صورة صالح.');
            return;
        }

        // Show local preview immediately before sending to server
        const reader = new FileReader();
        reader.onload = function(e) {
            origImg.src = e.target.result;
            dropZone.classList.add('hidden');
            spinner.classList.remove('hidden'); // Show loading instead of preview initially
        }
        reader.readAsDataURL(file);

        const formData = new FormData();
        formData.append('file', file);

        const startTime = performance.now();

        // Send to backend
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            const endTime = performance.now();
            const timeTaken = Math.round(endTime - startTime);

            spinner.classList.add('hidden');
            previewContainer.classList.remove('hidden'); // Show the image preview now with reset button

            if (data.success) {
                displayResults(data, timeTaken);
            } else {
                showError(data.error || 'حدث خطأ غير معروف أثناء المعالجة.');
            }
        })
        .catch(error => {
            spinner.classList.add('hidden');
            dropZone.classList.remove('hidden');
            showError('فشل الاتصال بالخادم: ' + error.message);
        });
    }

    function displayResults(data, timeTaken) {
        // We override the local preview with the backend one just in case it drew bounding boxes
        origImg.src = data.original_image;
        cropImg.src = data.plate_crop;
        
        plateNumber.textContent = data.plate_number;
        confidence.textContent = (data.confidence * 100).toFixed(1) + '%';
        procTime.textContent = timeTaken + ' ms';

        // Toggle placeholders and results
        yoloPlaceholder.classList.add('hidden');
        cnnPlaceholder.classList.add('hidden');
        yoloResult.classList.remove('hidden');
        cnnResult.classList.remove('hidden');
    }

    function showError(message) {
        errorText.textContent = message;
        errorMsg.classList.remove('hidden');
        fileInput.value = ''; // Reset input
    }

    resetBtn.addEventListener('click', () => {
        // Reset Box 1
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        fileInput.value = '';
        errorMsg.classList.add('hidden');

        // Reset Box 2
        yoloResult.classList.add('hidden');
        yoloPlaceholder.classList.remove('hidden');

        // Reset Box 3
        cnnResult.classList.add('hidden');
        cnnPlaceholder.classList.remove('hidden');
    });
});
