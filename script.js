// Configuration
const API_BASE_URL = 'http://localhost:5001/api';

// Sample news articles for demonstration
const sampleArticles = [
    "Scientists at Stanford University have discovered a new species of marine life in the Pacific Ocean depths. The research team found previously unknown bioluminescent creatures living near underwater volcanic vents at depths exceeding 3,000 meters. The discovery, published in Nature magazine, could provide insights into how life adapts to extreme conditions.",
    
    "Local community center in downtown receives $50,000 federal grant for youth programs. The funding will support after-school activities, summer camps, and educational workshops for children in underserved neighborhoods. Mayor Johnson announced the grant during yesterday's city council meeting, emphasizing the importance of investing in youth development.",
    
    "Government announces major infrastructure investment plan worth $2 billion. The initiative aims to modernize roads, bridges, and public transportation systems across the state over the next five years. Construction is expected to begin in early 2024, creating thousands of jobs in the engineering and construction sectors.",
    
    "Breaking: Alien spacecraft spotted hovering over major city for 3 hours, military confirms extraterrestrial contact established. Government officials admit to secret negotiations with alien beings who demand immediate surrender of all world governments. Citizens advised to stock up on supplies as invasion appears imminent."
];

// Utility Functions
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function clearText() {
    document.getElementById('newsText').value = '';
    hideResult();
    updateCharacterCount();
}

async function pasteText() {
    try {
        const text = await navigator.clipboard.readText();
        document.getElementById('newsText').value = text;
        updateCharacterCount();
    } catch (err) {
        // Fallback for browsers that don't support clipboard API
        alert('Please paste manually using Ctrl+V (Windows/Linux) or Cmd+V (Mac)');
    }
}

function generateSample() {
    const randomSample = sampleArticles[Math.floor(Math.random() * sampleArticles.length)];
    document.getElementById('newsText').value = randomSample;
    updateCharacterCount();
}

function updateCharacterCount() {
    const text = document.getElementById('newsText').value;
    const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
    
    // You can add a character counter display here if needed
    console.log(`Words: ${wordCount}, Characters: ${text.length}`);
}

// Main Analysis Function
async function analyzeNews() {
    const text = document.getElementById('newsText').value.trim();
    
    // Input validation
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
    
    // Show loading state
    analyzeBtn.innerHTML = '<div class="loading-spinner" style="width: 20px; height: 20px; margin-right: 10px;"></div>Analyzing...';
    analyzeBtn.disabled = true;
    showLoading();
    hideResult();
    
    try {
        // Make API call to backend
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                text: text,
                timestamp: new Date().toISOString()
            })
        });
        
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('Backend server not found. Please make sure the Python server is running on port 5000.');
            } else if (response.status === 500) {
                const errorData = await response.json();
                throw new Error(`Server error: ${errorData.error || 'Unknown error occurred'}`);
            } else {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        }
        
        const result = await response.json();
        
        // Validate response
        if (!result.prediction || result.confidence === undefined) {
            throw new Error('Invalid response from server');
        }
        
        showResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        
        // Show user-friendly error messages
        let errorMessage = 'Error analyzing news. ';
        
        if (error.message.includes('fetch')) {
            errorMessage += 'Please make sure the backend server is running on http://localhost:5000';
        } else if (error.message.includes('Backend server not found')) {
            errorMessage += 'Backend server not found. Please start the Python server first.';
        } else {
            errorMessage += error.message;
        }
        
        alert(errorMessage);
        
        // Fallback to demo mode if backend is not available
        if (error.message.includes('fetch') || error.message.includes('Backend server not found')) {
            if (confirm('Backend server not available. Would you like to try demo mode?')) {
                const demoResult = simulatePrediction(text);
                showResult(demoResult);
            }
        }
        
    } finally {
        // Reset button state
        analyzeBtn.innerHTML = originalText;
        analyzeBtn.disabled = false;
        hideLoading();
    }
}

// Demo mode fallback
function simulatePrediction(text) {
    // Simple heuristic for demo purposes
    const words = text.toLowerCase().split(' ');
    const suspiciousWords = ['breaking', 'shocking', 'unbelievable', 'exclusive', 'secret', 'exposed', 'alien', 'invasion', 'conspiracy'];
    const reliableWords = ['study', 'research', 'university', 'published', 'according', 'official', 'government', 'announced'];
    
    const suspiciousCount = words.filter(word => suspiciousWords.includes(word)).length;
    const reliableCount = words.filter(word => reliableWords.includes(word)).length;
    
    // Simple scoring algorithm for demo
    const isFake = suspiciousCount > reliableCount || text.length < 50;
    const baseConfidence = Math.random() * 0.2 + 0.75; // 75-95%
    const confidence = Math.min(0.95, Math.max(0.60, baseConfidence));
    
    return {
        prediction: isFake ? 'FAKE' : 'REAL',
        confidence: confidence
    };
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
    
    if (result.prediction === 'REAL') {
        container.className = 'result-container result-real';
        icon.textContent = '✅';
        title.textContent = 'Likely Real News';
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
        title.textContent = 'Potentially Fake News';
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
    
    // Scroll to results
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
        'ai': 'AI Content Detector'
    };
    
    alert(`${toolNames[toolType]} selected!\n\nThis feature would integrate additional AI models for comprehensive content analysis. Coming soon in the premium version!`);
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('newsText');
    
    // Auto-resize textarea
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.max(200, this.scrollHeight) + 'px';
        updateCharacterCount();
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter or Cmd+Enter to analyze
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            analyzeNews();
        }
        
        // Ctrl+Shift+C or Cmd+Shift+C to clear
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
            e.preventDefault();
            clearText();
        }
    });
    
    // Prevent form submission on Enter
    textarea.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            // Add new line instead
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
        console.log('⚠️ Backend API not available - using demo mode');
        return false;
    }
}

// Initialize app
window.addEventListener('load', function() {
    checkAPIHealth();
    console.log('🔍 TruthBot initialized');
    console.log('Keyboard shortcuts:');
    console.log('- Ctrl+Enter: Analyze news');
    console.log('- Ctrl+Shift+C: Clear text');
});