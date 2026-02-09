// ================================
// Configuration (Auto-Detect API)
// ================================
let API_BASE_URL;

// If running on localhost (desktop)
if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    API_BASE_URL = "http://localhost:5001/api";
} else {
    // Use the same LAN IP as the frontend but point to port 5001 (Flask)
    const currentHost = window.location.hostname; // e.g. 192.168.1.47
    API_BASE_URL = `http://${currentHost}:5001/api`;
}

console.log("🌐 Using API base:", API_BASE_URL);


// Utility Functions
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

    const titleInput = document.getElementById("titleInput"); // if you have one
    const textarea = document.getElementById("newsText");

function clearText () {
    if (titleInput) titleInput.value = '';
    if (textarea) {
        textarea.value = '';
        textarea.style.height = '200px'; // Reset height
    }
    if (typeof hideResult === "function") hideResult();
    if (typeof updateCharacterCount === "function") updateCharacterCount();
};



async function pasteText() {
    try {
        const text = await navigator.clipboard.readText();
        document.getElementById('newsText').value = text;
        updateCharacterCount();
    } catch (err) {
        alert('Please paste manually using Ctrl+V (Windows/Linux) or Cmd+V (Mac)');
    }
}

function updateCharacterCount() {
    const text = document.getElementById('newsText').value;
    const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
    console.log(`Words: ${wordCount}, Characters: ${text.length}`);
}

// Main Analysis Function
async function analyzeNews() {
    const text = document.getElementById('newsText').value.trim();
    
    if (!text) {
        alert('Please enter some news content to analyze');
        return;
    }
    
    if (text.length < 20) {
        alert('Please enter at least 20 characters for accurate analysis');
        return;
    }

    const analyzeBtn = document.getElementById('analyzeBtn');
    const originalText = analyzeBtn.innerHTML;
    
    analyzeBtn.innerHTML = '<div class="loading-spinner" style="width: 20px; height: 20px; margin-right: 10px;"></div>Analyzing...';
    analyzeBtn.disabled = true;
    showLoading();
    hideResult();
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, timestamp: new Date().toISOString() })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (!result.prediction || result.confidence === undefined) {
            throw new Error('Invalid response from server');
        }
        
        showResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Error analyzing news: ${error.message}`);
    } finally {
        analyzeBtn.innerHTML = originalText;
        analyzeBtn.disabled = false;
        hideLoading();
    }
}

// Results Display Function
function showResult(result) {
    const container = document.getElementById('resultContainer');
    const icon = document.getElementById('resultIcon');
    const title = document.getElementById('resultTitle');
    const subtitle = document.getElementById('resultSubtitle');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    const details = document.getElementById('resultDetails');
    
    const confidencePercent = Math.round(result.confidence * 100);
    
    if (result.prediction === 'Credible') {
        container.className = 'result-container result-real';
        icon.textContent = '✅';
        title.textContent = 'Likely Credible News';
        subtitle.textContent = 'This content appears to be authentic and reliable';
        details.innerHTML = `
            <strong>Analysis Summary:</strong><br>
            • Content shows characteristics of legitimate journalism<br>
            • Language patterns suggest factual reporting<br>
            • Structure aligns with credible news sources<br>
            • Confidence level: ${confidencePercent}%
        `;
    } else {
        container.className = 'result-container result-fake';
        icon.textContent = '⚠️';
        title.textContent = 'Potentially Not Credible News';
        subtitle.textContent = 'This content may contain misinformation or bias';
        details.innerHTML = `
            <strong>Warning Signs Detected:</strong><br>
            • Sensational language patterns identified<br>
            • Content structure raises credibility concerns<br>
            • May contain unverified claims<br>
            • Please verify with multiple sources
        `;
    }
    
    confidenceFill.style.width = confidencePercent + '%';
    confidenceText.textContent = `${confidencePercent}%`;
    container.style.display = 'block';
    setTimeout(() => {
        container.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
}

function hideResult() {
    document.getElementById('resultContainer').style.display = 'none';
}

// Tool Selection Function
function selectTool(toolType) {
    const toolNames = {
        'bias': 'Bias Detector',
        'sentiment': 'Sentiment Analyzer',
        'source': 'Source Checker',
        'fact': 'Fact Checker',
        'readability': 'Readability Score',
        'ai': 'AI Content Detector',
        'upgrade': 'Upgrade Features',
        'explore': 'Explore'

    };
    alert(`${toolNames[toolType]} selected!\n\nThis feature will be integrated in a future version.`);
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('newsText');
    
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.max(200, this.scrollHeight) + 'px';
        updateCharacterCount();
    });
    
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            analyzeNews();
        }
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
            e.preventDefault();
            clearText();
        }
    });
    
    textarea.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const start = this.selectionStart;
            const end = this.selectionEnd;
            this.value = this.value.substring(0, start) + '\n' + this.value.substring(end);
            this.selectionStart = this.selectionEnd = start + 1;
        }
    });
});

// API Health Check
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('✅ Backend API is running');
            return true;
        }
    } catch (error) {
        console.log('⚠️ Backend API not available');
        return false;
    }
}

// Initialize app
window.addEventListener('load', function() {
    checkAPIHealth();
    console.log('🔍 TruthBot initialized');
    console.log('Keyboard shortcuts: Ctrl+Enter = Analyze, Ctrl+Shift+C = Clear text');
});
document.getElementById("closeResult").addEventListener("click", function() {
    document.getElementById("resultContainer").style.display = "none";
});
function openModal() {
    const modal = document.getElementById("modelModal");
    modal.classList.add("active");
    // document.body.style.overflow = "hidden"; // ❌ Removed to keep background scroll enabled
  }
  
  function closeModal() {
    const modal = document.getElementById("modelModal");
    modal.classList.remove("active");
    // document.body.style.overflow = ""; // Not needed anymore
  }
  
  // Close on overlay click
  document.getElementById("modelModal").addEventListener("click", function (e) {
    if (e.target === this) {
      closeModal();
    }
  });
  
  // Close on Escape key
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      closeModal();
    }
  });
  