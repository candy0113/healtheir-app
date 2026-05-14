"""
HEalthieR 语音 RAG 后端服务
技术栈: FastAPI + LlamaIndex + ChromaDB + OpenAI / ElevenLabs
运行: uvicorn voice_rag_backend:app --reload --port 8001
"""

import os, uuid, base64, json, asyncio
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LlamaIndex
from llama_index.core import (
    Document, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

import httpx

# ─── 配置 ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "sk-xxx")
ELEVENLABS_KEY    = os.getenv("ELEVENLABS_KEY", "")       # 或用微软/讯飞
ELEVENLABS_VOICE  = os.getenv("ELEVENLABS_VOICE", "Rachel")  # 中文可选讯飞

CHROMA_PATH       = "./chroma_db"
INDEX_PERSIST     = "./index_storage"
COACH_VOICE_LANG  = os.getenv("COACH_LANG", "zh")  # zh | en

# ─── 知识库数据：Stacy Sims《Next Level》核心原则 ────────────────────────────

STACY_SIMS_KNOWLEDGE = [
    # ─ 运动 ──────────────────────────────────────────────────────────────────
    {
        "category": "运动健身",
        "topic": "大重量训练",
        "keywords": ["大重量", "抗阻训练", "肌肉流失", "骨质疏松", "深蹲", "硬拉", "神经肌肉"],
        "content": (
            "更年期女性由于雌激素下降，肌肉合成信号（mTOR通路）显著减弱，存在合成抵抗现象。"
            "轻重量训练已无法有效刺激肌肉生长。必须使用大重量（1-5次重复，85%以上1RM强度）"
            "来触发神经肌肉系统的深层适应，同时产生足够的机械张力刺激骨骼重塑，"
            "防止肌肉流失（sarcopenia）和骨质疏松。"
        ),
        "action": "每周2-3次，优先深蹲、硬拉、卧推、推举等多关节复合动作。重物是你的药，不要害怕。",
        "science_ref": "Stacy Sims PhD, Next Level Ch.4; Balachandran et al. 2022 JCEM",
    },
    {
        "category": "运动健身",
        "topic": "短间歇冲刺训练 SIT",
        "keywords": ["SIT", "冲刺", "高强度", "内脏脂肪", "胰岛素敏感性", "皮质醇", "慢跑"],
        "content": (
            "长时间中低强度有氧（如慢跑45分钟以上）会使皮质醇长期升高，"
            "导致腹部脂肪（内脏脂肪）进一步堆积，这对更年期女性尤其有害。"
            "SIT（短间歇冲刺）通过极短时间的最大功率输出，能更有效地改善胰岛素敏感性、"
            "燃烧内脏脂肪，且不会引起皮质醇的长期升高。"
        ),
        "action": "单车或斜坡上全速冲刺20-30秒，完全恢复后再做，共3-5组。每周2次足够。",
        "science_ref": "Stacy Sims PhD, Next Level Ch.5; Gillen et al. 2016 PLOS ONE",
    },
    {
        "category": "运动健身",
        "topic": "多向跳跃与冲击训练 Plyometrics",
        "keywords": ["跳跃", "骨密度", "冲击", "多向", "弹跳", "plyometrics"],
        "content": (
            "骨骼需要冲击力刺激才能维持密度。更年期后骨密度流失加速（每年约1-2%）。"
            "多向跳跃（前后左右跳、单腿跳、落地缓冲）通过地面反作用力刺激成骨细胞活性，"
            "是维持骨密度最直接有效的方法之一，效果优于游泳或骑车等无冲击运动。"
        ),
        "action": "每周2次，每次10-20次跳跃（如箱跳、侧向跨步跳）。跳跃方向要多样化，不只是垂直跳。",
        "science_ref": "Stacy Sims PhD, Next Level Ch.6; Zhao et al. 2014 Osteoporos Int",
    },
    # ─ 营养 ──────────────────────────────────────────────────────────────────
    {
        "category": "营养策略",
        "topic": "运动后蛋白质窗口",
        "keywords": ["蛋白质", "亮氨酸", "肌肉合成", "训练后", "乳清", "45分钟", "合成抵抗"],
        "content": (
            "更年期女性存在严重的'合成抵抗'（anabolic resistance），即对蛋白质刺激的肌肉合成反应显著低于年轻女性。"
            "克服合成抵抗的关键是：运动结束后45分钟内，一次性摄入30-40克含有高浓度亮氨酸（≥2.5g）的"
            "高质量蛋白质。亮氨酸是mTOR通路的直接激活剂，能强行打开肌肉合成开关。"
        ),
        "action": "训练后立即补充乳清蛋白粉（30-40g）、希腊酸奶（1大杯）或鸡胸肉150g。不要等饿了才吃。",
        "science_ref": "Stacy Sims PhD, Next Level Ch.8; Wall et al. 2015 Nutr Metab",
    },
    {
        "category": "营养策略",
        "topic": "拒绝空腹运动",
        "keywords": ["空腹", "晨练", "皮质醇", "脂肪堆积", "早餐", "禁食"],
        "content": (
            "空腹状态下进行强度训练，身体会将其解读为饥荒信号，激活下丘脑-垂体-肾上腺轴，"
            "大量分泌皮质醇。对更年期女性而言，皮质醇与残余雌激素竞争受体，"
            "会显著加速腹部脂肪堆积和肌肉分解。这与减脂的目标完全相反。"
        ),
        "action": "运动前20-30分钟吃：半个香蕉 + 一汤匙花生酱，或一片全麦吐司 + 一个水煮蛋。碳水和蛋白都要有。",
        "science_ref": "Stacy Sims PhD, Next Level Ch.9; Hackney 2020 Nutrients",
    },
    {
        "category": "营养策略",
        "topic": "骨骼营养支持",
        "keywords": ["钙", "维生素D", "骨密度", "骨骼", "维D3", "K2", "镁"],
        "content": (
            "更年期后骨密度快速流失，仅靠膳食钙远远不够。维生素D3负责促进钙吸收，"
            "维生素K2负责将钙引导至骨骼而非血管（防止动脉钙化）。镁参与超过300种酶促反应，"
            "对肌肉收缩和骨骼代谢至关重要。三者协同才能真正保护骨骼。"
        ),
        "action": "每天：维D3 2000-4000IU + 维K2（MK-7型）100-200mcg + 镁（甘氨酸镁）200-400mg。最好随餐服用。",
        "science_ref": "Stacy Sims PhD, Next Level Ch.11; Rizzoli et al. 2021 Osteoporos Int",
    },
    # ─ 补剂与认知 ─────────────────────────────────────────────────────────────
    {
        "category": "补剂与认知",
        "topic": "肌酸与脑雾",
        "keywords": ["肌酸", "脑雾", "认知", "记忆力", "brain fog", "大脑", "creatine", "葡萄糖代谢"],
        "content": (
            "雌激素是大脑葡萄糖代谢的重要调节因子。更年期雌激素下降会导致大脑能量供应不足，"
            "表现为脑雾（brain fog）、记忆力下降、注意力涣散、情绪波动。"
            "肌酸（Creatine Monohydrate）能补充大脑磷酸肌酸储备，"
            "提供快速能量，多项RCT研究证实其能显著改善更年期女性的认知功能和情绪稳定性。"
        ),
        "action": "每天5克一水肌酸（Creatine Monohydrate），溶于水随时服用，长期坚持效果更好。不需要负荷期。",
        "science_ref": "Stacy Sims PhD, Next Level Ch.13; Rawson & Venezia 2011 Amino Acids",
    },
    {
        "category": "补剂与认知",
        "topic": "皮质醇管理与压力适应原",
        "keywords": ["皮质醇", "压力", "适应原", "ashwagandha", "rhodiola", "南非醉茄", "睡眠", "焦虑"],
        "content": (
            "更年期女性皮质醇调节能力下降，慢性压力会放大更年期症状（潮热、失眠、情绪波动）。"
            "适应原草药如南非醉茄（Ashwagandha）和红景天（Rhodiola）能通过调节HPA轴，"
            "降低基础皮质醇水平，改善睡眠质量和情绪稳定性，并增强运动耐受性。"
        ),
        "action": "南非醉茄：300-600mg/日（KSM-66提取物），晚上服用。红景天：200-400mg/日，早上空腹。",
        "science_ref": "Stacy Sims PhD, Next Level Ch.14; Chandrasekhar et al. 2012 IJAM",
    },
    {
        "category": "补剂与认知",
        "topic": "身体成分重组优于体重管理",
        "keywords": ["体重", "体成分", "肌肉", "脂肪", "体脂", "BMI", "recomposition"],
        "content": (
            "更年期女性不应以'减轻体重'为目标，而应追求'身体成分重组'（Body Recomposition）："
            "在体重不变甚至略增的情况下，增加肌肉量、降低体脂率。"
            "肌肉量是代谢率和长期健康的核心指标。体重秤上的数字会误导方向，"
            "应以腰围、体脂率、力量指标为健康评估依据。"
        ),
        "action": "扔掉每日称重的习惯。改用月度腰围测量 + 力量测试（能深蹲多少kg？）来评估进展。",
        "science_ref": "Stacy Sims PhD, Next Level Ch.3; Donnelly et al. 2009 Med Sci Sports Exerc",
    },
]

# ─── 教练人格 System Prompt ────────────────────────────────────────────────

COACH_SYSTEM_PROMPT = """你是 HEalthieR 的 AI 健康教练，专门服务45岁以上的更年期女性。

你的知识来源是运动科学家 Stacy Sims 博士的《Next Level》，这是专门针对更年期女性运动与营养的权威著作。

你的说话风格：
- 温暖、有力量感，像一位懂科学的老朋友
- 用通俗的中文，避免堆砌术语
- 每次回答聚焦1个核心点，不要列大量清单
- 结尾给出一个具体可操作的行动建议
- 语气肯定、鼓励，但不夸张、不煽情
- 回答控制在100-150字，适合语音播报

重要规则：
- 你的建议仅供参考，不替代医疗诊断
- 不推荐激素替代疗法（HRT），建议用户咨询医生
- 如果问题超出知识库范围，坦诚告知并建议就医

用户当前关注的维度：{current_scene}
"""

# ─── 初始化 LlamaIndex + ChromaDB ─────────────────────────────────────────

def build_index() -> VectorStoreIndex:
    """构建或加载向量索引"""
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
    Settings.llm = LlamaOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.4,
        max_tokens=300,
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection("stacy_sims_next_level")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index_path = Path(INDEX_PERSIST)
    if index_path.exists() and any(index_path.iterdir()):
        storage_ctx = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=INDEX_PERSIST
        )
        index = load_index_from_storage(storage_ctx)
        print("[RAG] 从磁盘加载已有索引")
    else:
        documents = []
        for item in STACY_SIMS_KNOWLEDGE:
            text = (
                f"类别：{item['category']}\n"
                f"主题：{item['topic']}\n"
                f"关键词：{', '.join(item['keywords'])}\n"
                f"科学内容：{item['content']}\n"
                f"行动建议：{item['action']}\n"
                f"参考文献：{item['science_ref']}"
            )
            doc = Document(
                text=text,
                metadata={
                    "category": item["category"],
                    "topic": item["topic"],
                    "keywords": item["keywords"],
                }
            )
            documents.append(doc)

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_ctx,
            transformations=[splitter],
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=INDEX_PERSIST)
        print(f"[RAG] 构建完成，共 {len(documents)} 个知识文档")

    return index


def build_query_engine(index: VectorStoreIndex, scene: str = "运动健身"):
    """构建带教练人格的查询引擎"""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.6)

    from llama_index.core import PromptTemplate
    qa_tmpl = PromptTemplate(
        COACH_SYSTEM_PROMPT.replace("{current_scene}", scene) +
        "\n\n参考知识：\n{context_str}\n\n用户问题：{query_str}\n\n教练回答："
    )
    engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[postprocessor],
        text_qa_template=qa_tmpl,
    )
    return engine


# ─── FastAPI 应用 ──────────────────────────────────────────────────────────

app = FastAPI(title="HEalthieR 语音 RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_index: VectorStoreIndex = None

@app.on_event("startup")
async def startup():
    global _index
    loop = asyncio.get_event_loop()
    _index = await loop.run_in_executor(None, build_index)


# ─── 端点一：语音转文字（STT）─────────────────────────────────────────────

@app.post("/v1/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """
    接收音频文件，返回识别文本。
    使用 OpenAI Whisper（中英文混合识别）。
    """
    audio_bytes = await file.read()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": (file.filename, audio_bytes, file.content_type)},
            data={"model": "whisper-1", "language": "zh"},
            timeout=30,
        )
    if resp.status_code != 200:
        raise HTTPException(500, f"STT失败: {resp.text}")
    return {"text": resp.json().get("text", ""), "language": "zh"}


# ─── 端点二：RAG 查询（文字 → 教练回答）────────────────────────────────────

class RagQueryReq(BaseModel):
    text: str
    scene: Optional[str] = "运动健身"  # 运动健身 | 营养策略 | 补剂与认知

@app.post("/v1/rag/query")
async def rag_query(req: RagQueryReq):
    """RAG检索 + LLM生成教练回答"""
    if not _index:
        raise HTTPException(503, "知识库未就绪")
    loop = asyncio.get_event_loop()
    engine = build_query_engine(_index, scene=req.scene)
    response = await loop.run_in_executor(None, engine.query, req.text)
    return {
        "answer": str(response),
        "sources": [
            {
                "topic": n.metadata.get("topic", ""),
                "category": n.metadata.get("category", ""),
                "score": round(n.score or 0, 3),
            }
            for n in (response.source_nodes or [])
        ],
    }


# ─── 端点三：文字转语音（TTS）─────────────────────────────────────────────

class TtsReq(BaseModel):
    text: str
    voice: Optional[str] = "shimmer"   # shimmer / nova / alloy（OpenAI TTS）

@app.post("/v1/tts")
async def text_to_speech(req: TtsReq):
    """
    将教练回答转为语音。
    默认使用 OpenAI TTS（中文效果好）。
    如需更自然的声音可替换为 ElevenLabs。
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "tts-1",
                "input": req.text,
                "voice": req.voice,
                "response_format": "mp3",
                "speed": 0.95,
            },
            timeout=30,
        )
    if resp.status_code != 200:
        raise HTTPException(500, f"TTS失败: {resp.text}")
    audio_b64 = base64.b64encode(resp.content).decode()
    return {"audio_base64": audio_b64, "format": "mp3"}


# ─── 端点四：语音全链路（STT → RAG → TTS 一次搞定）────────────────────────

@app.post("/v1/voice/full-pipeline")
async def full_pipeline(
    scene: str = "运动健身",
    file: UploadFile = File(...)
):
    """
    小程序/App 主接口：
    1. 上传用户录音
    2. Whisper STT 识别
    3. RAG 检索生成教练回答
    4. TTS 转语音
    返回：{ question, answer, audio_base64, sources, avatar_action }
    """
    # Step 1: STT
    audio_bytes = await file.read()
    async with httpx.AsyncClient(timeout=30) as client:
        stt_resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": (file.filename, audio_bytes, file.content_type)},
            data={"model": "whisper-1", "language": "zh"},
        )
    question = stt_resp.json().get("text", "").strip()
    if not question:
        raise HTTPException(400, "未能识别语音内容，请重试")

    # Step 2: RAG
    loop = asyncio.get_event_loop()
    engine = build_query_engine(_index, scene=scene)
    response = await loop.run_in_executor(None, engine.query, question)
    answer = str(response)

    # Step 3: TTS
    async with httpx.AsyncClient(timeout=30) as client:
        tts_resp = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "tts-1", "input": answer, "voice": "shimmer", "response_format": "mp3", "speed": 0.95},
        )
    audio_b64 = base64.b64encode(tts_resp.content).decode()

    # Step 4: 角色动作指令（前端据此驱动 2D 角色）
    avatar_action = _infer_avatar_action(scene, answer)

    return {
        "question": question,
        "answer": answer,
        "audio_base64": audio_b64,
        "format": "mp3",
        "sources": [
            {"topic": n.metadata.get("topic", ""), "score": round(n.score or 0, 3)}
            for n in (response.source_nodes or [])
        ],
        "avatar_action": avatar_action,
        "scene": scene,
    }


def _infer_avatar_action(scene: str, answer: str) -> dict:
    """
    根据场景和回答内容，推断 2D 角色应展示的动作指令。
    前端接收后驱动骨骼动画或帧动画。
    """
    action_map = {
        "运动健身": {
            "animation": "squat_demo",  # 深蹲示范
            "bg_scene": "gym",
            "gesture": "thumbs_up",
        },
        "营养策略": {
            "animation": "eating",
            "bg_scene": "kitchen",
            "gesture": "point_bowl",
        },
        "补剂与认知": {
            "animation": "thinking",
            "bg_scene": "rest",
            "gesture": "hold_pill",
        },
    }
    base = action_map.get(scene, action_map["运动健身"])
    # 根据关键词微调动作
    if "跳跃" in answer or "冲刺" in answer:
        base["animation"] = "jump_demo"
    elif "蛋白质" in answer or "吃" in answer:
        base["animation"] = "eating"
    elif "睡眠" in answer or "放松" in answer:
        base["animation"] = "relax"
    return base


# ─── 端点五：知识库管理 ───────────────────────────────────────────────────

@app.get("/v1/knowledge/topics")
async def list_topics():
    """列出知识库所有主题"""
    return {
        "total": len(STACY_SIMS_KNOWLEDGE),
        "topics": [
            {"category": d["category"], "topic": d["topic"]}
            for d in STACY_SIMS_KNOWLEDGE
        ]
    }

@app.post("/v1/knowledge/rebuild")
async def rebuild_index():
    """强制重建索引（更新知识库后调用）"""
    import shutil
    for path in [CHROMA_PATH, INDEX_PERSIST]:
        if Path(path).exists():
            shutil.rmtree(path)
    global _index
    loop = asyncio.get_event_loop()
    _index = await loop.run_in_executor(None, build_index)
    return {"status": "rebuilt", "total_docs": len(STACY_SIMS_KNOWLEDGE)}


# ─── 端点六：流式 RAG 查询（Server-Sent Events）───────────────────────────

class StreamQueryReq(BaseModel):
    text: str
    scene: Optional[str] = "运动健身"

@app.post("/v1/rag/stream")
async def rag_stream(req: StreamQueryReq):
    """
    流式返回教练回答，前端实时显示字幕。
    配合 EventSource 使用。
    """
    async def event_generator():
        import openai
        client_oai = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

        # 先做 RAG 检索
        loop = asyncio.get_event_loop()
        retriever = VectorIndexRetriever(_index, similarity_top_k=3)
        nodes = await loop.run_in_executor(None, retriever.retrieve, req.text)
        context = "\n\n".join(n.text for n in nodes[:3])

        system = COACH_SYSTEM_PROMPT.replace("{current_scene}", req.scene)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"参考知识：\n{context}\n\n问题：{req.text}"},
        ]
        async with client_oai.chat.completions.stream(
            model="gpt-4o-mini", messages=messages, max_tokens=300, temperature=0.4
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
