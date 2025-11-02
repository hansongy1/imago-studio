# app.py
from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경용
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 모델 import
from models.clip_matcher import (
    initialize_animal_embeddings,
    find_similar_faces,
    get_personality_by_text,
    generate_comment
)
from models.face_analyzer import (
    analyze_face_emotion,
    generate_feedback,
    analyze_best_moment,
    get_emotion_timeline
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/uploads/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs('static/animals', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 동물 임베딩 캐시 (전역 변수)
animal_embeddings_cache = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_animal_embeddings():
    """연예인 임베딩 캐시 (최초 1회만 계산)"""
    global animal_embeddings_cache
    if animal_embeddings_cache is None:
        print("연예인 데이터베이스 초기화 중...")
        animal_embeddings_cache = initialize_animal_embeddings()
        print(f"총 {len(animal_embeddings_cache)}명의 데이터 로드 완료!")
    return animal_embeddings_cache


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/what-animal')
def mode1():
    """모드 1: CLIP 닮은꼴 찾기"""
    return render_template('mode_001.html')


@app.route('/presentation-practice')
def mode2():
    """모드 2: DeepFace 면접 연습"""
    return render_template('mode_002.html')


@app.route('/analyze-similarity', methods=['POST'])
def analyze_similarity():
    """
    모드 1: 사용자 얼굴과 닮은 동물 찾기
    """
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
    
    if file and allowed_file(file.filename):
        # 파일 저장
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 동물 임베딩 가져오기
            animal_embeddings = get_animal_embeddings()
            
            if not animal_embeddings:
                return jsonify({
                    'error': '동물 데이터베이스가 비어있습니다. static/animals/ 폴더에 이미지를 추가해주세요.'
                }), 500
            
            # 상위 3개 닮은꼴 찾기
            similar_faces = find_similar_faces(filepath, animal_embeddings, top_k=3)
            
            # ✨ 여기가 바로 새로운 로직의 핵심입니다! ✨
            # 1. 상위 결과들의 카테고리를 분석합니다.
            categories = {face['category'] for face in similar_faces} # set으로 중복 제거

            # 2. 카테고리 조합에 따라 재치있는 결과 타이틀을 생성합니다.
            result_title = ""
            if 'dogs' in categories and 'cats' in categories:
                result_title = "오묘한 매력의 ✨강냥이상✨이시네요!"
            elif 'dogs' in categories:
                result_title = "다채로운 매력의 🐶 강아지상🐶 입니다!"
            elif 'cats' in categories:
                result_title = "시크함과 귀여움이 공존하는 🐱고양이상🐱이네요!"
            else:
                result_title = "세상에, 동물나라에서 온 귀염둥이상이에요! 🥰"


            # 성격 분석 (텍스트 기반)
            personality = get_personality_by_text(filepath)
            
            # 각 결과에 코멘트 추가
            for face in similar_faces:
                face['comment'] = generate_comment(face['similarity'])
            
            return jsonify({
                'success': True,
                'user_image': f"/static/uploads/{filename}",
                'similar_faces': similar_faces,
                'personality': personality,
                'result_title': result_title
            })
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return jsonify({'error': f'분석 중 오류: {str(e)}'}), 500
    
    return jsonify({'error': '유효하지 않은 파일 형식'}), 400


@app.route('/analyze-emotion-realtime', methods=['POST'])
def analyze_emotion_realtime():
    """
    모드 2: 실시간 감정 분석 (웹캠)
    """
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        
        # Base64 디코드
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 임시 저장
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_webcam.jpg')
        cv2.imwrite(temp_path, img)
        
        # DeepFace 감정 분석
        emotion_result = analyze_face_emotion(temp_path)
        
        return jsonify({
            'success': True,
            'emotion': emotion_result
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'분석 중 오류: {str(e)}'}), 500


@app.route('/generate-practice-report', methods=['POST'])
def generate_practice_report():
    """
    모드 2: 발표 연습 최종 리포트 생성
    """
    try:
        data = request.json
        emotion_history = data.get('emotion_history', [])
        
        if not emotion_history:
            return jsonify({'error': '데이터가 없습니다'}), 400
        
        # 피드백 생성
        feedback = generate_feedback(emotion_history)
        
        # 베스트 순간 찾기
        best_moments = analyze_best_moment(emotion_history)
        
        # 타임라인 데이터
        timeline = get_emotion_timeline(emotion_history)
        
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        plt.rcParams['axes.unicode_minus'] = False
        
        # 감정별 시간 추이
        plt.plot(timeline['timestamps'], timeline['happy'], 
                marker='o', label='Happy 😊', linewidth=2, color='#4CAF50')
        plt.plot(timeline['timestamps'], timeline['neutral'], 
                marker='s', label='Neutral 😐', linewidth=2, color='#2196F3')
        plt.plot(timeline['timestamps'], timeline['fear'], 
                marker='^', label='Fear 😰', linewidth=2, color='#FF9800')
        plt.plot(timeline['timestamps'], timeline['confidence'], 
                marker='D', label='Confidence 💪', linewidth=2, 
                color='#9C27B0', linestyle='--')
        
        plt.xlabel('Time (frames)', fontsize=12)
        plt.ylabel('Score (%)', fontsize=12)
        plt.title('Emotion Timeline During Practice', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(
            app.config['RESULT_FOLDER'], 
            f"report_{timestamp}.png"
        )
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return jsonify({
            'success': True,
            'report_image': f"/static/uploads/results/report_{timestamp}.png",
            'feedback': feedback,
            'best_moments': best_moments
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'리포트 생성 중 오류: {str(e)}'}), 500


@app.route('/save-best-frame', methods=['POST'])
def save_best_frame():
    """
    베스트 순간 이미지 저장
    """
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        
        # Base64 디코드
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"best_frame_{timestamp}.jpg"
        filepath = os.path.join(app.config['RESULT_FOLDER'], filename)
        cv2.imwrite(filepath, img)
        
        return jsonify({
            'success': True,
            'image_path': f"/static/uploads/results/{filename}"
        })
        
    except Exception as e:
        return jsonify({'error': f'저장 중 오류: {str(e)}'}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("🎭 Face Match Studio 서버 시작!")
    print("=" * 50)
    print("📍 URL: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)