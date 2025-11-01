from deepface import DeepFace
import cv2
import numpy as np

def analyze_face_emotion(image_path):
    """
    얼굴 감정 분석 (면접/발표 연습용)
    
    Args:
        image_path: 이미지 경로
        
    Returns:
        분석 결과 딕셔너리
    """
    try:
        # 감정만 분석 (빠르게!)
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],  # age, gender 제거!
            enforce_detection=False,
            detector_backend='opencv'  # 더 빠른 백엔드
        )
        
        # 결과가 리스트로 올 수 있음
        if isinstance(result, list):
            result = result[0]
        
        return {
            'emotions': result['emotion'],
            'dominant_emotion': result['dominant_emotion'],
            'age': 0,  # 사용 안 함
            'gender': 'Unknown',  # 사용 안 함
            'confidence_score': calculate_confidence(result['emotion'])
        }
    except Exception as e:
        print(f"감정 분석 오류: {e}")
        return {
            'emotions': {
                'happy': 0,
                'sad': 0,
                'angry': 0,
                'surprise': 0,
                'fear': 0,
                'disgust': 0,
                'neutral': 100
            },
            'dominant_emotion': 'neutral',
            'age': 0,
            'gender': 'Unknown',
            'confidence_score': 50
        }


def calculate_confidence(emotions):
    """
    감정 데이터로 자신감 점수 계산 (개선된 버전)
    
    Args:
        emotions: 감정 딕셔너리
        
    Returns:
        confidence_score: 0-100 점수
    """
    # 긍정적 감정: happy, surprise
    # 부정적 감정: fear, sad, angry
    # 중립: neutral
    
    happy = emotions.get('happy', 0)
    surprise = emotions.get('surprise', 0)
    neutral = emotions.get('neutral', 0)
    fear = emotions.get('fear', 0)
    sad = emotions.get('sad', 0)
    angry = emotions.get('angry', 0)
    
    # 새로운 계산 방식 - 더 후하게!
    positive_score = happy * 1.0 + surprise * 0.7
    neutral_score = neutral * 0.5
    negative_score = (fear + sad + angry) * 0.3
    
    # 기본 점수를 60에서 시작 (더 관대하게)
    base_score = 60
    
    confidence = base_score + positive_score - negative_score + neutral_score * 0.3
    
    # 최소 30, 최대 100
    confidence = max(30, min(100, confidence))
    
    return round(confidence, 1)


def analyze_best_moment(emotion_history):
    """
    연습 세션에서 가장 좋았던 순간 찾기
    
    Args:
        emotion_history: 감정 기록 리스트
        
    Returns:
        best_moments: 상위 3개 순간
    """
    scored_moments = []
    
    for idx, record in enumerate(emotion_history):
        score = record.get('confidence_score', 0)
        scored_moments.append({
            'index': idx,
            'timestamp': record.get('timestamp', idx),
            'score': score,
            'emotion': record.get('dominant_emotion', 'neutral'),
            'frame': record.get('frame', None)
        })
    
    # 점수 순으로 정렬
    scored_moments.sort(key=lambda x: x['score'], reverse=True)
    
    return scored_moments[:3]


def generate_feedback(emotion_history):
    """
    전체 연습 세션 피드백 생성
    
    Args:
        emotion_history: 감정 기록 리스트
        
    Returns:
        feedback: 피드백 딕셔너리
    """
    if not emotion_history:
        return {
            'title': '데이터 없음',
            'message': '연습 데이터가 충분하지 않습니다.',
            'tips': ['먼저 연습을 시작해주세요!']
        }
    
    # 평균 감정 계산
    avg_emotions = {}
    emotion_keys = emotion_history[0]['emotions'].keys()
    
    for key in emotion_keys:
        avg_emotions[key] = sum(e['emotions'][key] for e in emotion_history) / len(emotion_history)
    
    # 평균 자신감
    avg_confidence = sum(e.get('confidence_score', 0) for e in emotion_history) / len(emotion_history)
    
    # 지배적 감정
    dominant_emotion = max(avg_emotions, key=avg_emotions.get)
    
    # 피드백 생성
    feedback = {
        'avg_confidence': round(avg_confidence, 1),
        'dominant_emotion': dominant_emotion,
        'avg_emotions': {k: round(v, 1) for k, v in avg_emotions.items()}
    }
    
    # 감정별 맞춤 피드백
    if dominant_emotion == 'happy':
        feedback['title'] = '😊 훌륭해요!'
        feedback['message'] = f"밝고 자신감 넘치는 표정이었어요! (행복 {avg_emotions['happy']:.1f}%)"
        feedback['tips'] = [
            '✅ 긍정적인 에너지가 느껴져요',
            '💡 이 표정을 면접/발표에서도 유지하세요',
            '⭐ 미소는 청중과의 연결고리가 됩니다'
        ]
        feedback['grade'] = 'A'
        feedback['color'] = '#4CAF50'
        
    elif dominant_emotion == 'neutral':
        feedback['title'] = '😐 안정적이에요'
        feedback['message'] = f"차분하고 안정적인 표정이었어요 (중립 {avg_emotions['neutral']:.1f}%)"
        feedback['tips'] = [
            '✅ 침착한 모습이 좋아요',
            '💡 중요한 포인트에서는 미소를 더해보세요',
            '⭐ 표정 변화로 메시지를 강조할 수 있어요'
        ]
        feedback['grade'] = 'B+'
        feedback['color'] = '#2196F3'
        
    elif dominant_emotion == 'fear':
        feedback['title'] = '😰 긴장하셨나요?'
        feedback['message'] = f"약간 긴장된 모습이 보였어요 (불안 {avg_emotions['fear']:.1f}%)"
        feedback['tips'] = [
            '💡 심호흡을 하고 천천히 말해보세요',
            '⭐ 연습을 반복하면 자신감이 생깁니다',
            '🎯 청중을 친구라고 생각해보세요',
            '✨ 당신은 충분히 잘하고 있어요!'
        ]
        feedback['grade'] = 'C+'
        feedback['color'] = '#FF9800'
        
    elif dominant_emotion == 'surprise':
        feedback['title'] = '😮 생동감 있어요!'
        feedback['message'] = f"표정이 풍부하고 생동감 있었어요 (놀람 {avg_emotions['surprise']:.1f}%)"
        feedback['tips'] = [
            '✅ 표정이 살아있어요',
            '💡 과하지 않게 조절하면 완벽해요',
            '⭐ 중요한 순간에 이런 표정을 활용하세요'
        ]
        feedback['grade'] = 'B'
        feedback['color'] = '#9C27B0'
        
    else:  # sad, angry, disgust
        feedback['title'] = '🤔 표정을 밝게!'
        feedback['message'] = "표정이 조금 어두워 보였어요"
        feedback['tips'] = [
            '💡 거울을 보며 미소 연습을 해보세요',
            '⭐ 긍정적인 마인드로 시작해보세요',
            '✨ 연습할 때 좋아하는 음악을 들어보세요'
        ]
        feedback['grade'] = 'C'
        feedback['color'] = '#F44336'
    
    return feedback


def get_emotion_timeline(emotion_history):
    """
    시간대별 감정 변화 데이터 생성 (그래프용)
    
    Args:
        emotion_history: 감정 기록 리스트
        
    Returns:
        timeline_data: 그래프 데이터
    """
    timeline = {
        'timestamps': [],
        'happy': [],
        'neutral': [],
        'fear': [],
        'confidence': []
    }
    
    for idx, record in enumerate(emotion_history):
        timeline['timestamps'].append(idx)
        timeline['happy'].append(record['emotions'].get('happy', 0))
        timeline['neutral'].append(record['emotions'].get('neutral', 0))
        timeline['fear'].append(record['emotions'].get('fear', 0))
        timeline['confidence'].append(record.get('confidence_score', 0))
    
    return timeline