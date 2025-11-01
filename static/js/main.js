// main.js
if (document.getElementById('imageInput')) {
    // 이미지 미리보기
    document.getElementById('imageInput').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.getElementById('previewImage');
                preview.src = e.target.result;
                preview.style.display = 'block';
                document.querySelector('.upload-placeholder').style.display = 'none';
                document.getElementById('analyzeButton').style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    // 분석 시작
    document.getElementById('analyzeButton').addEventListener('click', async function() {
        const fileInput = document.getElementById('imageInput');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('이미지를 선택해주세요!');
            return;
        }

        // UI 전환
        document.getElementById('uploadSection').style.display = 'none';
        document.getElementById('loadingSection').style.display = 'block';

        // FormData 생성
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/analyze-similarity', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // 로딩 숨기기
                document.getElementById('loadingSection').style.display = 'none';
                
                // 결과 표시
                displayAnimalResults(data);
                document.getElementById('resultSection').style.display = 'block';
            } else {
                alert('오류: ' + data.error);
                location.reload();
            }
        } catch (error) {
            console.error('Error:', error);
            alert('분석 중 오류가 발생했습니다: ' + error.message);
            location.reload();
        }
    });
}

function displayAnimalResults(data) {
    // 성격 분석 표시
    const trait = data.personality.main_trait;
    const score = data.personality.score.toFixed(1);
    
    document.getElementById('personalityTrait').textContent = trait;
    document.getElementById('personalityScore').textContent = `${score}%`;

    // 닮은 동물 결과 표시
    const resultsContainer = document.getElementById('similarityResults');
    resultsContainer.innerHTML = '';

    data.similar_animals.forEach((animal, index) => {
        const medal = ['🥇', '🥈', '🥉'][index];
        const card = document.createElement('div');
        card.className = 'similarity-card';
        card.innerHTML = `
            <div class="rank-badge">${medal} ${index + 1}위</div>
            <div class="similarity-image-wrapper">
                <img src="${animal.image}" alt="${animal.name}" class="similarity-image">
            </div>
            <div class="similarity-info">
                <h3 class="celebrity-name">${animal.name}</h3>
                <div class="similarity-score">
                    <span class="score-number">${animal.similarity.toFixed(1)}%</span>
                    <span class="score-label">유사도</span>
                </div>
                <p class="similarity-comment">${animal.comment}</p>
                <p class="celebrity-description">${animal.description}</p>
            </div>
        `;
        resultsContainer.appendChild(card);
    });
}

// ===== Mode2: 면접 연습 모드 =====
if (document.getElementById('webcam')) {
    let webcamStream = null;
    let isRecording = false;
    let emotionHistory = [];
    let analysisInterval = null;
    let startTime = null;
    let timerInterval = null;

    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    // 페이지 로드 시 웹캠 미리보기
    window.addEventListener('DOMContentLoaded', async function() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            webcam.srcObject = stream;
            webcamStream = stream;
            console.log('웹캠 미리보기 시작');
        } catch (error) {
            alert('웹캠 접근 권한이 필요합니다!');
            console.error('Webcam error:', error);
        }
    });

    // 연습 시작
    document.getElementById('startButton').addEventListener('click', async function() {
        if (!webcamStream) {
            alert('웹캠이 준비되지 않았습니다!');
            return;
        }

        isRecording = true;
        emotionHistory = [];
        startTime = Date.now();

        // UI 업데이트
        document.getElementById('startButton').style.display = 'none';
        document.getElementById('stopButton').style.display = 'block';
        document.getElementById('practiceTimer').style.display = 'flex';
        document.getElementById('emotionOverlay').style.display = 'block';

        // 타이머 시작
        timerInterval = setInterval(updateTimer, 1000);

        // 감정 분석 시작 (2초마다)
        analysisInterval = setInterval(analyzeEmotion, 2000);
        
        console.log('연습 시작!');
    });

    // 연습 종료
    document.getElementById('stopButton').addEventListener('click', function() {
        stopRecording();
        generateReport();
    });

    function stopRecording() {
        isRecording = false;
        
        // 인터벌 정리
        clearInterval(analysisInterval);
        clearInterval(timerInterval);
        
        console.log('연습 종료. 분석 횟수:', emotionHistory.length);
    }

    function updateTimer() {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const seconds = (elapsed % 60).toString().padStart(2, '0');
        document.getElementById('timerText').textContent = `${minutes}:${seconds}`;
    }

    async function analyzeEmotion() {
        if (!isRecording) return;

        // 캔버스에 현재 프레임 캡처
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        ctx.drawImage(webcam, 0, 0);

        // Base64로 변환
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        try {
            const response = await fetch('/analyze-emotion-realtime', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();

            if (data.success) {
                const emotion = data.emotion;
                
                // 기록 저장
                emotionHistory.push({
                    timestamp: Date.now() - startTime,
                    emotions: emotion.emotions,
                    dominant_emotion: emotion.dominant_emotion,
                    confidence_score: emotion.confidence_score,
                    frame: imageData
                });

                // UI 업데이트
                updateEmotionDisplay(emotion);
                document.getElementById('frameCount').textContent = 
                    `분석 횟수: ${emotionHistory.length}`;
                
                console.log('분석 완료:', emotionHistory.length, '/', emotion.dominant_emotion);
            }
        } catch (error) {
            console.error('Emotion analysis error:', error);
        }
    }

    function updateEmotionDisplay(emotion) {
        const emotionEmojis = {
            'happy': '😊',
            'neutral': '😐',
            'fear': '😰',
            'sad': '😢',
            'angry': '😠',
            'surprise': '😮',
            'disgust': '🤢'
        };

        const emotionNames = {
            'happy': '행복',
            'neutral': '중립',
            'fear': '불안',
            'sad': '슬픔',
            'angry': '화남',
            'surprise': '놀람',
            'disgust': '혐오'
        };

        document.getElementById('emotionEmoji').textContent = 
            emotionEmojis[emotion.dominant_emotion] || '😐';
        document.getElementById('emotionName').textContent = 
            emotionNames[emotion.dominant_emotion] || '중립';

        const confidence = emotion.confidence_score;
        document.getElementById('confidenceFill').style.width = `${confidence}%`;
        document.getElementById('confidenceText').textContent = 
            `자신감: ${confidence.toFixed(1)}%`;

        // 색상 변경
        const fill = document.getElementById('confidenceFill');
        if (confidence >= 70) {
            fill.style.background = '#4CAF50';
        } else if (confidence >= 50) {
            fill.style.background = '#FF9800';
        } else {
            fill.style.background = '#F44336';
        }
    }

    async function generateReport() {
        console.log('리포트 생성 시작. 데이터 수:', emotionHistory.length);
        
        if (emotionHistory.length < 3) {
            alert(`데이터가 부족합니다. (${emotionHistory.length}개 수집됨)\n최소 3회 이상 분석이 필요합니다. 약 6초 이상 연습해주세요!`);
            location.reload();
            return;
        }

        // UI 전환
        document.getElementById('webcamSection').style.display = 'none';
        document.getElementById('loadingSection').style.display = 'block';

        try {
            const response = await fetch('/generate-practice-report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ emotion_history: emotionHistory })
            });

            const data = await response.json();

            if (data.success) {
                document.getElementById('loadingSection').style.display = 'none';
                displayReport(data);
                document.getElementById('resultSection').style.display = 'block';
            } else {
                alert('오류: ' + data.error);
                location.reload();
            }
        } catch (error) {
            console.error('Report generation error:', error);
            alert('리포트 생성 중 오류가 발생했습니다.');
            location.reload();
        }
    }

    function displayReport(data) {
        const feedback = data.feedback;

        // 피드백 카드
        document.getElementById('feedbackGrade').textContent = feedback.grade;
        document.getElementById('feedbackTitle').textContent = feedback.title;
        document.getElementById('feedbackMessage').textContent = feedback.message;
        document.getElementById('avgConfidence').textContent = 
            feedback.avg_confidence.toFixed(1) + '%';

        // 카드 색상
        const feedbackCard = document.getElementById('feedbackCard');
        feedbackCard.style.borderColor = feedback.color;

        // 팁
        const tipsList = document.getElementById('tipsList');
        tipsList.innerHTML = '';
        feedback.tips.forEach(tip => {
            const li = document.createElement('li');
            li.textContent = tip;
            tipsList.appendChild(li);
        });

        // 그래프
        document.getElementById('graphImage').src = data.report_image;

        // 베스트 순간
        const bestMomentsContainer = document.getElementById('bestMoments');
        bestMomentsContainer.innerHTML = '';
        
        data.best_moments.forEach((moment, index) => {
            const momentCard = document.createElement('div');
            momentCard.className = 'best-moment-card';
            momentCard.innerHTML = `
                <div class="moment-rank">${index + 1}위</div>
                <div class="moment-score">점수: ${moment.score.toFixed(1)}</div>
                <p class="moment-emotion">${moment.emotion}</p>
            `;
            bestMomentsContainer.appendChild(momentCard);
        });
    }
}