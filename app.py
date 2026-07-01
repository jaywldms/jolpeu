import os
import math
import gc
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel

# 그래픽 카드 에러 방지 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = FastAPI()

# 프론트엔드 HTML 페이지와의 통신을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------
# 글로벌 설정 및 제공해주신 쿼리셋 데이터 탑재
# --------------------------------------------------------
GOOGLE_DRIVE_FOLDER_ID = "1xTQq4aptRlXQUPrnI_zVKWyQzZoYHwSO"
VISUAL_DB_PATH = "./artwork_db_total.pt"
SENTIMENT_DB_PATH = "./artwork_sentiment_db_total.pt"

ALL_SENTIMENT_QUERIES = [
    "A fluttering excitement like a first love", "Evoking warm and gentle comfort", "Explosive joy",
    "Awe that fills the heart", "A thrilling surge of shivers", "Noble and radiant sacrifice",
    "A deep and moving emotional resonance", "An aura that feels reverent and radiant",
    "Utterly calm, soothing the mind like a windless stillness", "Evoking an untouchable sense of dignity",
    "Overflowing vitality and burning passion", "So vivid it feels like it could leap out of the frame",
    "Pure and ideally beautiful in a transparent way", "Warm and full of tenderness",
    "A refreshing sense of release that opens up the chest", "Dreamlike and surreal, as if in a dream",
    "A fairytale-like fantasy", "A faint light in the darkness, evoking a sense of hope",
    "A natural, unembellished scene evoking intimate everyday life", "Evoking free and playful innocence",
    "Deep shadows and stillness, evoking quiet solace", "A calm and gentle atmosphere, evoking nostalgic intimacy",
    "Carrying an inner gloom", "Tense with an unstable composition",
    "Evoking the fragility and futility of life, as if about to crumble", "Revealing the harshness of a cold reality",
    "A humorous and satirical depiction with an underlying sense of sorrow", "Still and lifeless, as if time has stopped",
    "A fragile precariousness, like thin ice about to break", "Cold and sharp lighting, evoking a sense of alienation",
    "A deep sense of longing", "An overwhelming and suffocating pressure", "Fear as if the heart might stop",
    "A dizzying desire", "An empty and barren space, evoking deep solitude", "A heart-wrenching sadness",
    "A shock as if everything has come to a halt", "A passion that has gone cold",
    "Pain that feels like it pierces the heart", "Evoking violent energy", "Intense, as if emotions are erupting",
    "Evoking confusion", "A refined and balanced composition, evoking elegance",
    "Fragmented and disjointed visual elements, evoking reconstructed memory",
    "A calm and restrained scene, evoking quiet inner tension", "An ambiguous and undefined presence, evoking enigma",
    "A fleeting and softened moment, evoking ephemerality",
]

USED_SENTIMENT_QUERIES = [
    "A fluttering excitement like a first love", "Evoking warm and gentle comfort", "Explosive joy",
    "Awe that fills the heart", "A thrilling surge of shivers", "Noble and radiant sacrifice",
    "A deep and moving emotional resonance", "Utterly calm, soothing the mind like a windless stillness",
    "Evoking an untouchable sense of dignity", "Overflowing vitality and burning passion",
    "So vivid it feels like it could leap out of the frame", "Pure and ideally beautiful in a transparent way",
    "A refreshing sense of release that opens up the chest", "Dreamlike and surreal, as if in a dream",
    "A fairytale-like fantasy", "A faint light in the darkness, evoking a sense of hope",
    "A natural, unembellished scene evoking intimate everyday life", "Evoking free and playful innocence",
    "Deep shadows and stillness, evoking quiet solace", "A calm and gentle atmosphere, evoking nostalgic intimacy",
    "Carrying an inner gloom", "Tense with an unstable composition",
    "Evoking the fragility and futility of life, as if about to crumble", "Revealing the harshness of a cold reality",
    "A humorous and satirical depiction with an underlying sense of sorrow", "Still and lifeless, as if time has stopped",
    "A fragile precariousness, like thin ice about to break", "Cold and sharp lighting, evoking a sense of alienation",
    "A deep sense of longing", "An overwhelming and suffocating pressure", "Fear as if the heart might stop",
    "A dizzying desire", "An empty and barren space, evoking deep solitude",
    "A refined and balanced composition, evoking elegance", "A calm and restrained scene, evoking quiet inner tension",
]

SELECTED_SENTIMENT_INDICES = [ALL_SENTIMENT_QUERIES.index(q) for q in USED_SENTIMENT_QUERIES]

# --------------------------------------------------------
# 인프라 및 핵심 수학 유틸 함수 함수 정의
# --------------------------------------------------------

def path_key(path):
    p = str(path).replace("\\", "/")
    parts = [x for x in p.split("/") if x]
    if "wikiart_images" in parts:
        idx = parts.index("wikiart_images")
        return "/".join(parts[idx + 1:]).lower()
    if len(parts) >= 2:
        return "/".join(parts[-2:]).lower()
    return os.path.basename(p).lower()


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def min_max_norm(tensor):
    tensor = tensor.float()
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

def select_sentiment_dimensions(sentiment_tensor):
    sentiment_tensor = sentiment_tensor.float()
    if sentiment_tensor.ndim == 1:
        return sentiment_tensor[SELECTED_SENTIMENT_INDICES]
    return sentiment_tensor[:, SELECTED_SENTIMENT_INDICES]

def extract_drive_id_from_path(path_string):
    """
    제공해주신 노트북 아웃풋 구조에 맞춰 파일명 추출 및 가공 처리 함수
    """
    base_name = os.path.basename(path_string)
    return base_name

# --------------------------------------------------------
# AI 분석기 모델 클래스 (서버 부팅 시 메모리 상주)
# --------------------------------------------------------
class IntegratedArtworkAnalyzer:
    def __init__(self, expected_vit_dim=1000):
        self.device = get_device()
        print(f"[*] 연산 연산 장치 배정 완료: {self.device}")
        
        # ResNet50 로드
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_model = nn.Sequential(*list(resnet.children())[:-1]).to(self.device).eval()
        
        self.visual_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # CLIP 모델 로드
        clip_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
        self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device).eval()
        
        # ViT 모델 로드 (노트북 사양인 1000차원 유지 모드 적용)
        self.vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(self.device).eval()

    def extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        x = self.visual_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cnn_vec = self.cnn_model(x).flatten().cpu()
            vit_vec = self.vit_model(x).flatten().cpu()
            
            inputs = self.clip_processor(text=ALL_SENTIMENT_QUERIES, images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.clip_model(**inputs)
            probs_47 = F.softmax(outputs.logits_per_image, dim=1).flatten().cpu()
            probs_35 = select_sentiment_dimensions(probs_47)
            
        return cnn_vec.float(), vit_vec.float(), probs_35.float()

# 서버 구동 시 모델 및 고용량 DB를 단 1회 전역 로드
analyzer = None
aligned_db = {}

@app.on_event("startup")
def load_system():
    global analyzer, aligned_db
    print("[*] 시스템 데이터베이스와 AI 가중치 파일을 서버에 업로드 및 정렬하는 중입니다...")
    
    # 1. 두 파일 로드
    visual_db = torch.load(VISUAL_DB_PATH, map_location="cpu")
    sentiment_db = torch.load(SENTIMENT_DB_PATH, map_location="cpu")
    
    visual_paths = visual_db["paths"]
    sentiment_paths = sentiment_db["paths"]
    
    # 2. 경로 맵 구축 및 공통 키 추출 (노트북 매칭 알고리즘 적용)
    visual_map = {path_key(p): i for i, p in enumerate(visual_paths)}
    sentiment_map = {path_key(p): i for i, p in enumerate(sentiment_paths)}
    
    common_keys = sorted(set(visual_map.keys()) & set(sentiment_map.keys()))
    
    visual_indices = [visual_map[k] for k in common_keys]
    sentiment_indices = [sentiment_map[k] for k in common_keys]
    
    # 3. 정렬된 공통 인덱스 기준으로 텐서 크기 동기화 (80,735개로 일치)
    aligned_db["cnn"] = visual_db["cnn"][visual_indices].float()
    aligned_db["vit"] = visual_db["vit"][visual_indices].float()
    
    raw_sentiment = sentiment_db["probs"][sentiment_indices].float()
    aligned_db["sentiment"] = select_sentiment_dimensions(raw_sentiment)
    
    # 최종 결과 화면에 파일명을 뿌려주기 위해 정렬된 경로 저장
    aligned_db["paths"] = [visual_paths[i] for i in visual_indices]
    
    # 모델 로드
    analyzer = IntegratedArtworkAnalyzer()
    print(f"[*] 데이터 동기화 완료: 최종 공통 미술 작품 수 = {len(aligned_db['paths']):,}개")
    print("[*] 모든 AI 아키텍처 및 텐서 연산 준비가 완료되었습니다.")

# --------------------------------------------------------
# 이미지 업로드 수신 및 추천 점수 연산 파이프라인 엔드포인트
# --------------------------------------------------------
@app.post("/recommend")
async def get_recommendation(file: UploadFile = File(...)):
    # 임시 폴더 생성 및 업로드 파일 저장
    os.makedirs("./temp", exist_ok=True)
    temp_file_path = f"./temp/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # 사용자가 업로드한 대상 이미지 특징 추출
    target_cnn, target_vit, target_sent = analyzer.extract_features(temp_file_path)
    
    # 코사인 유사도 일괄 연산 처리 (PyTorch 벡터 연산 활용)
    cnn_sim = F.cosine_similarity(target_cnn.unsqueeze(0), aligned_db["cnn"])
    vit_sim = F.cosine_similarity(target_vit.unsqueeze(0), aligned_db["vit"])
    sent_sim = F.cosine_similarity(target_sent.unsqueeze(0), aligned_db["sentiment"])
    
    # 정규화
    cnn_score = min_max_norm(cnn_sim)
    vit_score = min_max_norm(vit_sim)
    sentiment_score = min_max_norm(sent_sim)
    
    # 가중치 결합 설계 파라미터 제어 (제공된 알고리즘 수식)
    # 1단계: 시각 가중치 분리 결합
    visual_score = (0.50 * cnn_score) + (0.50 * vit_score)
    # 2단계: 기본 최종 유사도 스케일링
    base_final_score = (0.70 * visual_score) + (0.30 * sentiment_score)
    # 3단계: 제안식 가중치 수식 산출법 (Novelty 보정항)
    novelty_score = (1.0 - visual_score).clamp(0.0, 1.0)
    novelty_bonus = novelty_score * sentiment_score
    
    # 발표자료 덧셈식(additive) 모드로 연산 점수 취합
    recommend_score = base_final_score + (0.10 * novelty_bonus)
    
    # 최상위 20개 선정 추출
    top_scores, top_indices = torch.topk(recommend_score, 20)
    
    response_data = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        raw_path = aligned_db["paths"][idx]
        filename = extract_drive_id_from_path(raw_path)
        
        # 구글 드라이브 파일 고유 ID 및 썸네일 변환 주소 빌드
        # 공유 폴더 내에 위치한 파일 고유의 ID가 경로명에 포함되어 있다는 아키텍처에 근거합니다.
        drive_file_id = filename.split('.')[0]  # 확장자 제거 처리
        
        # 대용량 이미지 부하를 막기 위해 width 400 썸네일 변환 옵션 부여
        image_url = f"https://drive.google.com/thumbnail?id={drive_file_id}&sz=w400"
        
        response_data.append({
            "filename": filename,
            "score": score,
            "image_url": image_url
        })
        
    # 메모리 누수 방지 및 임시 파일 디스크 삭제
    os.remove(temp_file_path)
    gc.collect()
    
    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)