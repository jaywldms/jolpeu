import os
import math
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel

# --------------------------------------------------
# [Streamlit UI 초기 설정] (반드시 최상단에 위치)
# --------------------------------------------------
st.set_page_config(
    page_title="예술 작품 추천 시스템",
    page_icon="🎨",
    layout="wide"
)

# --------------------------------------------------
# [환경 설정 및 상수] (노트북 코드 기반)
# --------------------------------------------------
VISUAL_DB_PATH = "./artwork_db_total.pt"
SENTIMENT_DB_PATH = "./artwork_sentiment_db_total.pt"
DATASET_ROOT = "./wikiart_images"

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
    "A deep sense of longing", "An overwhelming and suffocating pressure",
    "Fear as if the heart might stop", "A dizzying desire", "An empty and barren space, evoking deep solitude",
    "A heart-wrenching sadness", "A shock as if everything has come to a halt", "A passion that has gone cold",
    "Pain that feels like it pierces the heart", "Evoking violent energy", "Intense, as if emotions are erupting",
    "Evoking confusion", "A refined and balanced composition, evoking elegance",
    "Fragmented and disjointed visual elements, evoking reconstructed memory",
    "A calm and restrained scene, evoking quiet inner tension", "An ambiguous and undefined presence, evoking enigma",
    "A fleeting and softened moment, evoking ephemerality"
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
    "A deep sense of longing", "An overwhelming and suffocating pressure",
    "Fear as if the heart might stop", "A dizzying desire", "An empty and barren space, evoking deep solitude",
    "A refined and balanced composition, evoking elegance", "A calm and restrained scene, evoking quiet inner tension"
]

SELECTED_SENTIMENT_INDICES = [ALL_SENTIMENT_QUERIES.index(q) for q in USED_SENTIMENT_QUERIES]

# --------------------------------------------------
# [핵심 유틸리티 함수 내부 구현] (노트북 소스 연동)
# --------------------------------------------------
def select_sentiment_dimensions(probs_47, indices):
    return probs_47[indices]

def load_and_align_databases(visual_db_path, sentiment_db_path, selected_indices):
    """노트북의 DB 정렬 및 로드 함수 코드를 여기에 그대로 매핑합니다."""
    v_db = torch.load(visual_db_path, map_location="cpu")
    s_db = torch.load(sentiment_db_path, map_location="cpu")
    
    # 딕셔너리 정렬 및 매칭 로직
    aligned_paths = []
    cnn_features = []
    vit_features = []
    sentiment_features = []
    
    for path in v_db.keys():
        if path in s_db:
            aligned_paths.append(path)
            cnn_features.append(v_db[path]["cnn"].flatten())
            vit_features.append(v_db[path]["vit"].flatten())
            
            # 감성 벡터 추출 및 선택된 차원만 필터링
            full_sentiment = s_db[path]["sentiment"].flatten()
            filtered_sentiment = select_sentiment_dimensions(full_sentiment, selected_indices)
            sentiment_features.append(filtered_sentiment)
            
    return {
        "paths": aligned_paths,
        "cnn": torch.stack(cnn_features),
        "vit": torch.stack(vit_features),
        "sentiment": torch.stack(sentiment_features)
    }

def minmax_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val - min_val > 1e-6:
        return (tensor - min_val) / (max_val - min_val)
    return torch.zeros_like(tensor)

def recommend_artworks(target_features, aligned_db, top_k=10, 
                       cnn_in_visual_weight=0.5, vit_in_visual_weight=0.5,
                       visual_weight=0.7, sentiment_weight=0.3, 
                       novelty_weight=0.1, novelty_mode="stable"):
    """제안식(Stable) 및 덧셈식(Additive) 통합 추천 점수 계산 함수"""
    
    # 코사인 유사도 계산
    sim_cnn = F.cosine_similarity(aligned_db["cnn"], target_features["cnn"].unsqueeze(0), dim=1)
    sim_vit = F.cosine_similarity(aligned_db["vit"], target_features["vit"].unsqueeze(0), dim=1)
    sim_sentiment = F.cosine_similarity(aligned_db["sentiment"], target_features["sentiment"].unsqueeze(0), dim=1)
    
    # 정규화
    sim_cnn_norm = minmax_normalize(sim_cnn)
    sim_vit_norm = minmax_normalize(sim_vit)
    sim_sentiment_norm = minmax_normalize(sim_sentiment)
    
    # 1. S_visual = a * S_cnn + (1 - a) * S_vit
    S_visual = cnn_in_visual_weight * sim_cnn_norm + (1 - vit_in_visual_weight) * sim_vit_norm
    
    # 2. S_final = b * S_visual + (1 - b) * S_clip
    S_final = visual_weight * S_visual + sentiment_weight * sim_sentiment_norm
    
    # 3. Novelty = 1 - S_visual
    Novelty = 1.0 - S_visual
    
    # 4. 수식 계산 (제안식 Stable vs 발표자료 Additive)
    if novelty_mode == "stable":
        # S_recommend = (1 - λ) * S_final + λ * Novelty * S_clip
        S_recommend = (1.0 - novelty_weight) * S_final + novelty_weight * Novelty * sim_sentiment_norm
    else:
        # Additive: S_recommend = S_final + λ * Novelty
        S_recommend = S_final + novelty_weight * Novelty
        
    # 정렬 및 Top-K 추출
    scores, indices = torch.topk(S_recommend, k=min(top_k, len(S_recommend)))
    
    results = []
    for rank, (score, idx) in enumerate(zip(scores, indices), 1):
        idx_item = idx.item()
        results.append({
            "rank": rank,
            "path": aligned_db["paths"][idx_item],
            "recommend_score": score.item(),
            "visual_score": S_visual[idx_item].item(),
            "sentiment_score": sim_sentiment_norm[idx_item].item(),
            "novelty_score": Novelty[idx_item].item()
        })
    return results

def resolve_image_path(db_path, dataset_root):
    """DB 내 저장된 상대 경로와 로컬 dataset_root를 결합"""
    cleaned_path = db_path.lstrip("./")
    if cleaned_path.startswith("wikiart_images/"):
        cleaned_path = cleaned_path.replace("wikiart_images/", "", 1)
    return os.path.join(dataset_root, cleaned_path)

# --------------------------------------------------
# [캐싱 처리 함수] DB와 모델 로드 (최초 1회만 실행)
# --------------------------------------------------
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    cnn_model = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
    
    # CLIP
    clip_name = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
    
    # ViT-B/16
    vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device).eval()
    
    visual_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return device, cnn_model, vit_model, clip_processor, clip_model, visual_preprocess

@st.cache_data
def get_aligned_db():
    return load_and_align_databases(VISUAL_DB_PATH, SENTIMENT_DB_PATH, SELECTED_SENTIMENT_INDICES)

# --------------------------------------------------
# [Streamlit UI 구성]
# --------------------------------------------------
st.title("🎨 감성 + 참신성 기반 예술 작품 추천 시스템")
st.markdown("이미지를 업로드하면 시각적 유사도와 CLIP 감성 분석, 그리고 안정형 참신성 수식을 결합하여 최적의 작품을 추천합니다.")

# 사이드바 설정 영역
st.sidebar.header("⚙️ 가중치 설정")
cnn_vit_weight = st.sidebar.slider("CNN vs ViT 비중 (a)", 0.0, 1.0, 0.50, 0.05)
visual_weight = st.sidebar.slider("시각적 가중치 (b)", 0.0, 1.0, 0.70, 0.05)
sentiment_weight = 1.0 - visual_weight
st.sidebar.text(f"감성적 가중치 (1-b): {sentiment_weight:.2f}")

novelty_weight = st.sidebar.slider("참신성 가중치 (λ)", 0.0, 1.0, 0.10, 0.05)
novelty_mode = st.sidebar.selectbox("참신성 수식 모드", ["stable", "additive"], index=0)
top_k = st.sidebar.slider("추천 작품 수", 5, 30, 10, 1)

# 로딩 및 데이터 로드 실행
with st.spinner("AI 모델 및 예술 작품 데이터베이스를 로딩 중입니다..."):
    device, cnn_model, vit_model, clip_processor, clip_model, visual_preprocess = load_models()
    try:
        aligned_db = get_aligned_db()
    except Exception as e:
        st.error(f"❌ DB 로드 실패: {e}. 'artwork_db_total.pt'와 'artwork_sentiment_db_total.pt' 파일이 현재 폴더에 있는지 확인해 주세요.")
        st.stop()

# 파일 업로드 인터페이스
uploaded_file = st.file_uploader("추천의 기준이 될 이미지를 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    target_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(target_image, caption="내가 업로드한 타겟 이미지", use_container_width=True)
        
    with col2:
        st.success("이미지 분석 및 추천 점수를 연산하고 있습니다...")
        
        # 1. 타겟 이미지 특징 추출
        x = visual_preprocess(target_image).unsqueeze(0).to(device)
        with torch.no_grad():
            cnn_vec = cnn_model(x).flatten().cpu()
            vit_vec = vit_model(x).flatten().cpu()
            
        # CLIP 감성 추출
        inputs = clip_processor(text=ALL_SENTIMENT_QUERIES, images=target_image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs_47 = F.softmax(outputs.logits_per_image, dim=1).flatten().cpu()
            
        probs_selected = select_sentiment_dimensions(probs_47, SELECTED_SENTIMENT_INDICES)
        
        target_features = {
            "cnn": cnn_vec.float(),
            "vit": vit_vec.float(),
            "sentiment": probs_selected.float()
        }
        
        # 2. 추천 시스템 연산 엔진 호출
        results = recommend_artworks(
            target_features=target_features,
            aligned_db=aligned_db,
            top_k=top_k,
            cnn_in_visual_weight=cnn_vit_weight,
            vit_in_visual_weight=1.0 - cnn_vit_weight,
            visual_weight=visual_weight,
            sentiment_weight=sentiment_weight,
            novelty_weight=novelty_weight,
            novelty_mode=novelty_mode
        )
        
        st.balloons()
        
    # --------------------------------------------------
    # [Grid 형태 결과 시각화]
    # --------------------------------------------------
    st.write("---")
    st.subheader(f"📊 실시간 추천 결과 (Top {top_k})")
    
    cols_per_row = 5
    rows = math.ceil(len(results) / cols_per_row)
    
    for row in range(rows):
        st_cols = st.columns(cols_per_row)
        for col in range(cols_per_row):
            idx = row * cols_per_row + col
            if idx < len(results):
                res = results[idx]
                img_path = resolve_image_path(res["path"], dataset_root=DATASET_ROOT)
                
                with st_cols[col]:
                    if os.path.exists(img_path):
                        try:
                            artwork_img = Image.open(img_path)
                            st.image(artwork_img, use_container_width=True)
                        except Exception:
                            st.error("이미지 손상 또는 로드 실패")
                    else:
                        st.warning(f"이미지 없음\n({os.path.basename(img_path)})")
                    
                    st.markdown(f"**🏅 Rank {res['rank']}**")
                    st.caption(f"📁 {os.path.basename(img_path)}")
                    st.metric(label="최종 추천 점수", value=f"{res['recommend_score']:.4f}")
                    st.text(f"시각 유사도: {res['visual_score']:.3f}")
                    st.text(f"감성 유사도: {res['sentiment_score']:.3f}")
                    st.text(f"참신성(Nov): {res['novelty_score']:.3f}")