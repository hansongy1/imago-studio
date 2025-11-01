# clip_matcher.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from scipy.spatial.distance import cosine
import os

# CLIP 모델 로드 (싱글톤)
model = None
processor = None

# ver1
def get_clip_model():
    """CLIP 모델 싱글톤"""
    global model, processor
    if model is None:
        print("CLIP 모델 로딩 중...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP 모델 로드 완료!")
    return model, processor

# def get_clip_model():
#     """CLIP 모델 싱글톤"""
#     global model, processor
#     if model is None:
#         print("CLIP 모델 로딩 중...")
        
#         # 1. 어디서 계산할지 정하기 (GPU가 있으면 GPU, 없으면 CPU)
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"CLIP 모델을 '{device}' 장치에서 실행합니다.")

#         # 2. 모델을 지정된 장치로 보내기
#         model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#         processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#         print("CLIP 모델 로드 완료!")
#     return model, processor


def get_image_embedding(image_path):
    """
    이미지를 CLIP 임베딩으로 변환
    """
    model, processor = get_clip_model()
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    embedding = image_features.cpu().numpy()[0]
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

# def get_image_embedding(image_path):
#     """
#     이미지를 CLIP 임베딩으로 변환
    
#     Args:
#         image_path: 이미지 경로
        
#     Returns:
#         embedding: numpy array
#     """
#     model, processor = get_clip_model()
    
#     # 이미지 로드
#     image = Image.open(image_path).convert('RGB')
    
#     # 전처리
#     inputs = processor(images=image, return_tensors="pt")
    
#     # 임베딩 추출
#     with torch.no_grad():
#         image_features = model.get_image_features(**inputs)
    
#     # numpy로 변환 및 정규화
#     embedding = image_features.cpu().numpy()[0]
#     embedding = embedding / np.linalg.norm(embedding)
    
#     return embedding

# def get_image_embedding(image_path):
#     """
#     이미지를 CLIP 임베딩으로 변환
#     """
#     model, processor = get_clip_model()
    
#     # 모델이 있는 장치(device) 정보 가져오기
#     device = model.device

#     # 이미지 로드
#     image = Image.open(image_path).convert('RGB')
    
#     # 전처리
#     inputs = processor(images=image, return_tensors="pt")
    
#     # 3. 전처리된 이미지 데이터(pixel_values)를 모델과 같은 장치로 보내기
#     pixel_values = inputs.pixel_values.to(device)
    
#     # 임베딩 추출
#     with torch.no_grad():
#         # 수정된 데이터로 모델 실행
#         image_features = model.get_image_features(pixel_values=pixel_values)
    
#     # numpy로 변환 및 정규화
#     embedding = image_features.cpu().numpy()[0]
#     embedding = embedding / np.linalg.norm(embedding)
    
#     return embedding


def get_text_embedding(text):
    """
    텍스트를 CLIP 임베딩으로 변환
    """
    model, processor = get_clip_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    embedding = text_features.cpu().numpy()[0]
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

# def get_text_embedding(text):
#     """
#     텍스트를 CLIP 임베딩으로 변환
    
#     Args:
#         text: 텍스트 설명
        
#     Returns:
#         embedding: numpy array
#     """
#     model, processor = get_clip_model()
    
#     # 전처리
#     inputs = processor(text=[text], return_tensors="pt", padding=True)
    
#     # 임베딩 추출
#     with torch.no_grad():
#         text_features = model.get_text_features(**inputs)
    
#     # numpy로 변환 및 정규화
#     embedding = text_features.cpu().numpy()[0]
#     embedding = embedding / np.linalg.norm(embedding)
    
#     return embedding


# 동물 데이터베이스 - 강아지, 고양이, 귀여운 동물들!
# static/animals/ 폴더에 이미지를 저장하고 사용
ANIMAL_DATABASE = {
    'dogs': [
        {   
            'name': '골든 리트리버', 
            'images': [
                'static/animals/goldenRetriever_001.jpg',
                'static/animals/goldenRetriever_002.jpg',
                'static/animals/goldenRetriever_003.jpg'
            ], 
            'main_image': 'static/animals/goldenRetriever_001.jpg', # 대표 이미지
            'description': 'friendly and warm, always smiling'
        },
        {
            'name': '진돗개', 
            'images': [
                'static/animals/jindo_001.jpg',
                'static/animals/jindo_002.jpg',
                'static/animals/jindo_003.jpg'
            ], 
            'main_image': 'static/animals/jindo_001.jpg',
            'description': 'loyal and brave'
        },
        {   
            'name': '포메라니안', # pomeranian(끝)
            'images': [
                'static/animals/pomeranian_001.jpg',
                'static/animals/pomeranian_002.jpg',
                'static/animals/pomeranian_003.jpg'
            ], 
            'main_image': 'static/animals/pomeranian_001.jpg',
            'description': '귀여워요'
        },
        {
            'name': '웰시코기', # corgi(끝)
            'images': [
                'static/animals/corgi_001.jpg',
                'static/animals/corgi_002.jpg',
                'static/animals/corgi_003.jpg'
            ], 
            'main_image': 'static/animals/corgi_001.jpg',
            'description': 'short legs and cheerful'
        },
        {
            'name': '사모예드', # samoyed(끝)
            'images': [
                'static/animals/samoyed_001.jpg',
                'static/animals/samoyed_002.jpg',
                'static/animals/samoyed_003.jpg'
            ], 
            'main_image': 'static/animals/samoyed_001.jpg',
            'description': 'fluffy cloud, always happy'
        },
        {
            'name': '푸들', # poodle(끝)
            'images': [
                'static/animals/poodle_001.jpg',
                'static/animals/poodle_002.jpg',
                'static/animals/poodle_003.jpg'
            ], 
            'main_image': 'static/animals/poodle_001.jpg',
            'description': 'elegant and smart'
        },
        {
            'name': '비글', # beagle(끝)
            'images': [
                'static/animals/beagle_001.jpg',
                'static/animals/beagle_002.jpg',
                'static/animals/beagle_003.jpg'
            ], 
            'main_image': 'static/animals/beagle_001.jpg',
            'description': 'curious and energetic'
        },
        
    ],
    # 고양이
    'cats': [
        {
            'name': '러시안블루', # russianBlue(끝)
            'images': [
                'static/animals/russianBlue_001.jpg',
                'static/animals/russianBlue_002.jpg',
                'static/animals/russianBlue_003.jpg'
            ], 
            'main_image': 'static/animals/russianBlue_001.jpg',
            'description': 'elegant and mysterious gray'
        },
        {
            'name': '스코티시폴드', # scottishFold(끝)
            'images': [
                'static/animals/scottishFold_001.jpg',
                'static/animals/scottishFold_002.jpg',
                'static/animals/scottishFold_003.jpg'
            ], 
            'main_image': 'static/animals/scottishFold_001.jpg',
            'description': 'round face and gentle'
        },
        {
            'name': '페르시안', # persian(끝)
            'images': [
                'static/animals/persian_001.jpg',
                'static/animals/persian_002.jpg',
                'static/animals/persian_003.jpg'
            ], 
            'main_image': 'static/animals/persian_001.jpg',
            'description': 'fluffy and sophisticated'
        },
        {
            'name': '샴', # siamese(끝)
            'images': [
                'static/animals/siamese_001.jpg',
                'static/animals/siamese_002.jpg',
                'static/animals/siamese_003.jpg'
            ], 
            'main_image': 'static/animals/siamese_001.jpg',
            'description': 'sleek and vocal'
        },
        {
            'name': '코숏', # koreanShorthair(끝)
            'images': [
                'static/animals/koreanShorthair_001.jpg',
                'static/animals/koreanShorthair_002.jpg',
                'static/animals/koreanShorthair_003.jpg'
            ], 
            'main_image': 'static/animals/koreanShorthair_001.jpg',
            'description': 'typical cute kitty'
        },
        {
            'name': '브리티시숏헤어', # britishShorthair(끝)
            'images': [
                'static/animals/britishShorthair_001.jpg',
                'static/animals/britishShorthair_002.jpg',
                'static/animals/britishShorthair_003.jpg'
            ], 
            'main_image': 'static/animals/britishShorthair_001.jpg',
            'description': 'round and chubby'
        },
        {
            'name': '뱅갈', # bengal(끝)
            'images': [
                'static/animals/bengal_001.jpg',
                'static/animals/bengal_002.jpg',
                'static/animals/bengal_003.jpg'
            ], 
            'main_image': 'static/animals/bengal_001.jpg',
            'description': 'wild and energetic'
        },
    ],
    'cute_animals': [
        {
            'name': '팬더', # panda(끝)
            'images': [
                'static/animals/panda_001.jpg',
                'static/animals/panda_002.jpg',
                'static/animals/panda_003.jpg'
            ], 
            'main_image': 'static/animals/panda_001.jpg',
            'description': 'chubby and adorable'
        },
        {
            'name': '토끼', # rabbit(끝)
            'images': [
                'static/animals/rabbit_001.jpg',
                'static/animals/rabbit_002.jpg',
                'static/animals/rabbit_003.jpg'
            ], 
            'main_image': 'static/animals/rabbit_001.jpg',
            'description': 'soft and gentle'
        },
        {
            'name': '햄스터', # hamster(끝)
            'images': [
                'static/animals/hamster_001.jpg',
                'static/animals/hamster_002.jpg',
                'static/animals/hamster_003.jpg'
            ], 
            'main_image': 'static/animals/hamster_001.jpg',
            'description': 'tiny and cute'
        },
        {
            'name': '페넥여우', # fennecFox(끝)
            'images': [
                'static/animals/fennecFox_001.jpg',
                'static/animals/fennecFox_002.jpg',
                'static/animals/fennecFox_003.jpg'
            ], 
            'main_image': 'static/animals/fennecFox_001.jpg',
            'description': 'big ears and playful'
        },
        {
            'name': '알파카', # alpaca(끝)
            'images': [
                'static/animals/alpaca_001.jpg',
                'static/animals/alpaca_002.jpg',
                'static/animals/alpaca_003.jpg'
            ], 
            'main_image': 'static/animals/alpaca_001.jpg',
            'description': 'fluffy and calm'
        },
        {
            'name': '물범', #seal(끝)
            'images': [
                'static/animals/seal_001.jpg',
                'static/animals/seal_002.jpg',
                'static/animals/seal_003.jpg'
            ], 
            'main_image': 'static/animals/seal_001.jpg',
            'description': 'round and squishy'
        },
    ]
}


def initialize_animal_embeddings():
    """
    동물 이미지 데이터베이스의 임베딩을 미리 계산합니다.
    여러 장의 이미지가 있는 경우, 임베딩의 평균을 계산하여 대표값으로 사용합니다.
    """
    embeddings = {}
    
    database_to_use = ANIMAL_DATABASE
    
    for category, animals in database_to_use.items():
        for animal in animals:
            animal_name = animal['name']
            
            # 'images' 리스트가 있는지 확인
            image_paths = animal.get('images') 
            
            # 'images' 리스트가 없으면, 기존처럼 'image' 단일 경로를 사용 (둘 다 호환되도록 함)
            if not image_paths and 'image' in animal:
                image_paths = [animal['image']]

            if not image_paths:
                print(f"경고: '{animal_name}'에 대한 이미지가 없습니다.")
                continue

            all_embeddings = []
            valid_image_found = False
            # ✨ 여기가 핵심! 리스트의 각 경로를 하나씩 꺼내서 확인합니다.
            for img_path in image_paths:
                if os.path.exists(img_path):
                    try:
                        # 각 이미지의 임베딩을 계산해서 리스트에 추가
                        embedding = get_image_embedding(img_path)
                        all_embeddings.append(embedding)
                        valid_image_found = True
                    except Exception as e:
                        print(f"'{animal_name}'의 이미지 '{img_path}' 처리 중 오류: {e}")
                else:
                    print(f"경고: '{img_path}' 파일을 찾을 수 없습니다.")

            # 유효한 이미지가 하나라도 있으면, 모든 임베딩의 '평균'을 계산
            if valid_image_found:
                avg_embedding = np.mean(np.array(all_embeddings), axis=0)
                
                embeddings[animal_name] = {
                    'embedding': avg_embedding,
                    # 결과 화면에 보여줄 대표 이미지를 지정
                    'image': animal.get('main_image') or image_paths[0],
                    'description': animal['description'],
                    'category': category
                }

    return embeddings


def find_similar_faces(user_image_path, animal_embeddings, top_k=3):
    """
    사용자 얼굴과 가장 닮은 동물 찾기
    
    Args:
        user_image_path: 사용자 이미지 경로
        animal_embeddings: 미리 계산된 동물 임베딩
        top_k: 상위 k개 결과
        
    Returns:
        results: 닮은꼴 리스트
    """
    # 사용자 이미지 임베딩
    user_embedding = get_image_embedding(user_image_path)
    
    # 유사도 계산
    similarities = []
    
    for name, data in animal_embeddings.items():
        animal_embedding = data['embedding']
        
        # 코사인 유사도 (1 - 코사인 거리)
        similarity = 1 - cosine(user_embedding, animal_embedding)
        similarity_percent = similarity * 100
        
        similarities.append({
            'name': name,
            'similarity': similarity_percent,
            'image': data['image'],
            'description': data['description'],
            'category': data['category']
        })
    
    # 상위 k개 정렬
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similarities[:top_k]


def get_personality_by_text(user_image_path):
    """
    텍스트 기반으로 성격 분석
    CLIP의 텍스트-이미지 매칭 활용
    
    Args:
        user_image_path: 사용자 이미지 경로
        
    Returns:
        personality: 성격 분석 결과
    """
    user_embedding = get_image_embedding(user_image_path)
    
    # 다양한 성격 키워드
    personality_keywords = [
        "cute and adorable face",
        "charismatic and confident face",
        "gentle and kind face",
        "cool and sophisticated face",
        "energetic and bright face",
        "calm and peaceful face",
        "mysterious and elegant face",
        "friendly and warm face"
    ]
    
    # 각 키워드와 유사도 계산
    scores = {}
    for keyword in personality_keywords:
        text_embedding = get_text_embedding(keyword)
        similarity = 1 - cosine(user_embedding, text_embedding)
        scores[keyword] = similarity * 100
    
    # 가장 높은 점수
    top_personality = max(scores.items(), key=lambda x: x[1])
    
    return {
        'main_trait': top_personality[0],
        'score': top_personality[1],
        'all_scores': scores
    }


def generate_comment(similarity_score):
    """
    유사도에 따른 재밌는 코멘트 생성
    """
    if similarity_score >= 80:
        return "🎭 도플갱어 발견! 혹시 친척 아니세요?"
    elif similarity_score >= 70:
        return "👨‍👩‍👦 꽤 많이 닮았네요! 사진 보여주면 믿을 듯!"
    elif similarity_score >= 60:
        return "😊 어느 정도 닮은 구석이 있어요!"
    elif similarity_score >= 50:
        return "🤔 살짝 닮은 느낌이 드는데요?"
    else:
        return "🌟 당신만의 독특한 매력이 있어요!"