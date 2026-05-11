"""
aigc_service.py - CV 端 SD 微服务
基于 diffusers + IP-Adapter FaceID + insightface，暴露内网 API 供 BE-B 调用

运行: python aigc_service.py
端口: 58000

API:
  POST /aigc/generate  - 生图推理
  GET  /aigc/health    - 健康检查

存储: 当前使用本地文件 (待 MinIO 就绪后切换)
"""

import os
import sys
from pathlib import Path

# ⚠️ 必须在任何 huggingface/diffusers import 之前设置 ⚠️
_BASE_DIR = Path(__file__).parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_BASE_DIR / ".hf_cache"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import time
import uuid
import logging
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

# diffusers - SD 推理核心
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import load_image as _diffusers_load_image

# insightface - 人脸检测 + ID embedding 提取
import insightface
from insightface.app import FaceAnalysis

# ── 日志配置 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aigc")

# ── 路径配置 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "sd_models" / "v1-5-pruned-emaonly.safetensors"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CONTROLNET_DIR = BASE_DIR / "controlnet_models"

# MinIO 配置（待 FS 提供凭证后启用）
# MINIO_CONFIG = {
#     "endpoint": "localhost:9000",
#     "access_key": "your_access_key",
#     "secret_key": "your_secret_key",
#     "bucket": "aigc-images",
#     "secure": False,
# }

# ── 应用初始化 ────────────────────────────────────────────
app = FastAPI(title="AIGC Visual Engine", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 模型全局变量 ──────────────────────────────────────────
pipe: Optional[StableDiffusionImg2ImgPipeline] = None
face_app: Optional[FaceAnalysis] = None
ip_adapter_loaded: bool = False
start_time: float = 0.0


# ── 数据模型 ──────────────────────────────────────────────

class GenerateRequest(BaseModel):
    source_image_url: str = Field(..., description="原始图片 URL（本地路径或 HTTP URL）")
    positive_prompt: str = Field(..., description="正向 Prompt")
    negative_prompt: str = Field(default="", description="反向 Prompt")
    denoising_strength: float = Field(default=0.55, ge=0.0, le=1.0, description="去噪强度")
    cfg_scale: float = Field(default=8.0, ge=1.0, le=20.0, description="CFG 引导系数")
    seed: int = Field(default=-1, description="随机种子，-1 表示随机")
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    gender: str = Field(default="", description="性别 male/female，用于保持面部一致性")
    ip_adapter_scale: float = Field(default=0.7, ge=0.0, le=1.0, description="IP-Adapter 面部保留强度")
    # ── LoRA 参数（可选） ──
    lora_weights: dict = Field(
        default_factory=lambda: {"aging": 0.0, "fatigue": 0.0, "obesity": 0.0},
        description="LoRA 权重 {name: weight}",
    )


class GenerateResponse(BaseModel):
    generated_image_url: str
    generation_time_ms: int
    status: str  # "success" | "failed"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    ip_adapter_loaded: bool
    uptime_seconds: int


# ── 存储工具 ──────────────────────────────────────────────

def load_image_smart(source: str) -> tuple:
    """智能加载图片：支持本地路径 / HTTP URL / Base64 data URI
    返回: (pil_image_rgb, cv2_image_bgr)
    pil_image_rgb: 给 diffusers / CLIP 用
    cv2_image_bgr: 给 insightface 用
    """
    import base64, io

    if source.startswith("data:image"):
        # Base64 编码图片 —— 用 cv2.imdecode 保证 BGR 正确
        base64_data = source.split(",", 1)[1]
        img_data = base64.b64decode(base64_data)
        np_arr = np.frombuffer(img_data, np.uint8)
        cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 自动 BGR
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        logger.info(f"从 Base64 加载图片, 尺寸={pil_img.size}, cv2 shape={cv2_img.shape}")
        return pil_img, cv2_img
    elif source.startswith("http://") or source.startswith("https://"):
        # HTTP URL
        pil_img = _diffusers_load_image(source).convert("RGB")
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        logger.info(f"从 URL 加载图片, 尺寸={pil_img.size}")
        return pil_img, cv2_img
    else:
        # 本地路径 —— 用 cv2.imread 保证 BGR 正确
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {source}")
        cv2_img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # 自动 BGR
        if cv2_img is None:
            raise ValueError(f"cv2.imread 读取失败: {source}")
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        logger.info(f"从本地路径加载图片, 尺寸={pil_img.size}, cv2 shape={cv2_img.shape}")
        return pil_img, cv2_img

def save_image_local(image: Image.Image) -> str:
    """保存图片到本地 outputs/ 目录，返回可访问路径"""
    filename = f"{uuid.uuid4().hex}.png"
    filepath = OUTPUT_DIR / filename
    image.save(filepath, "PNG")
    return str(filepath.resolve())


# ── 人脸检测 + ID embedding 提取 ──────────────────────────

def _init_face_analyzer():
    """初始化 insightface 人脸分析器"""
    global face_app
    try:
        # 设置模型缓存到项目目录，不要下到 C 盘
        model_dir = BASE_DIR / ".insightface_models"
        model_dir.mkdir(exist_ok=True)
        face_app = FaceAnalysis(
            name="buffalo_l",
            root=str(model_dir),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # ctx_id: 0=GPU, -1=CPU
        # onnxruntime 没有 CUDAExecutionProvider 时必须用 -1
        import onnxruntime as ort
        has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        ctx_id = 0 if has_cuda else -1
        face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info(f"insightface 人脸分析器初始化成功 (ctx_id={ctx_id}, CUDA={'可用' if has_cuda else '不可用'})")
    except Exception as e:
        logger.warning(f"insightface 初始化失败: {e}，人脸 ID 提取将不可用")
        face_app = None


def extract_face_id_embedding(cv2_img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """从 BGR 图片中提取人脸 ID embedding（用于 IP-Adapter FaceID）
    输入必须是 cv2 BGR 格式，直接给 insightface 用，不做任何转换
    """
    if face_app is None:
        return None
    try:
        # 魔法补丁：给图片加 20% 黑边，专治"大脸贴边识别不出"
        h, w = cv2_img_bgr.shape[:2]
        pad_h, pad_w = int(h * 0.2), int(w * 0.2)
        padded_img = cv2.copyMakeBorder(
            cv2_img_bgr, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        logger.info(f"insightface 输入: 原始={cv2_img_bgr.shape}, 加边后={padded_img.shape}")

        # 照妖镜：保存 insightface 实际看到的图片
        debug_path = BASE_DIR / "tests" / "debug_insightface_input.jpg"
        cv2.imwrite(str(debug_path), padded_img)

        faces = face_app.get(padded_img)
        if not faces:
            logger.warning("未检测到人脸！请检查 debug_insightface_input.jpg")
            return None
        # 取最大的人脸
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        logger.info(f"检测到人脸, 置信度={face.det_score:.2f}")
        return face.normed_embedding.reshape(1, -1)
    except Exception as e:
        logger.warning(f"人脸 ID 提取失败: {e}")
        return None


# ── CLIP Image Encoder（供 IP-Adapter 使用）───────────────────────
image_encoder = None
image_processor = None

def _load_image_encoder():
    """加载 CLIP image encoder（IP-Adapter 需要）"""
    global image_encoder, image_processor
    encoder_dir = CONTROLNET_DIR / "image_encoder"
    if not encoder_dir.exists():
        logger.warning(f"image_encoder 目录不存在: {encoder_dir}")
        return
    try:
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # 1. 加载 image encoder 模型
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(encoder_dir), torch_dtype=dtype
        ).to(device)

        # 2. 构造 CLIPImageProcessor（标准 ViT-L/14 参数，无需额外文件）
        image_processor = CLIPImageProcessor(
            crop_size=224,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            feature_extractor_type="CLIPFeatureExtractor",
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            size=224,
        )

        logger.info("CLIP image encoder 加载成功")
    except Exception as e:
        logger.warning(f"CLIP image encoder 加载失败: {e}")
        image_encoder = None
        image_processor = None

def compute_ip_adapter_embeds(pil_image: Image.Image):
    """计算 IP-Adapter FaceID Plus 需要的 CLIP embeds
    返回原始 3D: (1, 257, 1280)，后续在 generate 中做升维 + CFG 拼接
    """
    if image_encoder is None or image_processor is None:
        return None
    try:
        device = image_encoder.device
        inputs = image_processor(images=pil_image, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = image_encoder(**inputs, output_hidden_states=True)
        # FaceID Plus 需要 penultimate hidden state: (1, 257, 1280)
        clip_embeds = outputs.hidden_states[-2]
        logger.info(f"clip_embeds 原始 shape: {clip_embeds.shape}")
        return clip_embeds
    except Exception as e:
        logger.warning(f"计算 clip_embeds 失败: {e}")
        return None


# ── LoRA 加载 ─────────────────────────────────────────────

LORA_DIR = BASE_DIR / "lora_models"
LORA_DIR.mkdir(exist_ok=True)

_LORA_FILES = {
    "aging": LORA_DIR / "aging.safetensors",
    "fatigue": LORA_DIR / "tiredness_slider.safetensors",
    "obesity": LORA_DIR / "body_weight_slider.safetensors",
}


def apply_loras(pipeline, weights: dict):
    """动态加载并调整 LoRA 权重（可选，文件不存在时静默跳过）"""
    has_any = any(weight > 0 for weight in weights.values())
    if not has_any:
        # 没有需要加载的 LoRA，尝试 unfuse 之前可能加载的
        try:
            pipeline.unfuse_lora()
        except Exception:
            pass
        return
    # 先 unfuse 之前的 LoRA
    try:
        pipeline.unfuse_lora()
    except Exception:
        pass
    logger.info(f"加载 LoRA 模型, weights={weights}...")
    for name, weight in weights.items():
        if weight <= 0:
            continue
        lora_path = _LORA_FILES.get(name)
        if not lora_path or not lora_path.exists():
            logger.warning(f"LoRA [{name}] 文件不存在: {lora_path}")
            continue
        try:
            pipeline.load_lora_weights(str(lora_path))
            pipeline.fuse_lora(lora_scale=weight)
            logger.info(f"LoRA [{name}] 已加载, 权重={weight}")
        except Exception as e:
            logger.warning(f"LoRA [{name}] 加载跳过: {e}")


# ── IP-Adapter 加载（面部特征保留） ────────────────────────

def _load_ip_adapter(pipeline):
    """加载 IP-Adapter Plus Face（使用 CLIP image_encoder + 面部参考图）"""
    global ip_adapter_loaded
    ip_path = CONTROLNET_DIR / "ip-adapter-faceid-plus_sd15.bin"
    encoder_dir = CONTROLNET_DIR / "image_encoder"

    if not ip_path.exists():
        logger.warning(f"IP-Adapter 模型未找到: {ip_path}，跳过")
        return

    if not encoder_dir.exists() or not (encoder_dir / "model.safetensors").exists():
        logger.warning(f"image_encoder 未找到: {encoder_dir}，跳过 IP-Adapter")
        return

    try:
        # 方法：用本地目录加载 IP-Adapter（需要 image_encoder 子目录）
        pipeline.load_ip_adapter(
            str(CONTROLNET_DIR),
            subfolder="",
            weight_name="ip-adapter-faceid-plus_sd15.bin",
        )
        pipeline.set_ip_adapter_scale(0.7)
        ip_adapter_loaded = True
        logger.info("IP-Adapter FaceID Plus 加载成功")
    except Exception as e:
        logger.warning(f"IP-Adapter 加载失败: {e}")
        # 降级：尝试从 HuggingFace 仓库在线加载
        try:
            pipeline.load_ip_adapter(
                "h94/IP-Adapter-FaceID",
                subfolder="",
                weight_name="ip-adapter-faceid-plus_sd15.bin",
            )
            pipeline.set_ip_adapter_scale(0.7)
            ip_adapter_loaded = True
            logger.info("IP-Adapter FaceID Plus (在线) 加载成功")
        except Exception as e2:
            logger.warning(f"IP-Adapter 在线加载也失败: {e2}")


# ── 模型加载 ──────────────────────────────────────────────

@app.on_event("startup")
def load_model():
    """启动时加载 SD 1.5 模型 + IP-Adapter FaceID + insightface"""
    global pipe, start_time
    start_time = time.time()

    logger.info("正在加载 SD 1.5 模型...")

    try:
        # 1. 加载 SD 1.5 pipeline
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            str(MODEL_PATH),
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

        # 2. 设置 scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            final_sigmas_type="sigma_min",
        )

        # 3. GPU 加速
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("模型已加载到 GPU (xformers 已启用)")

        # 4. 加载 CLIP image encoder（必须在 IP-Adapter 之前！）
        _load_image_encoder()

        # 5. 只有 image_encoder 可用时才加载 IP-Adapter
        #    因为 load_ip_adapter() 会修改 UNet 结构，一旦加载就
        #    必须每次调用都传入 added_cond_kwargs，否则崩溃
        if image_encoder is not None:
            # 关键：把 image_encoder 设到 pipeline 上
            # 这样 pipeline.__call__ 才能用 ip_adapter_image 参数
            # 自动编码图片并构造 added_cond_kwargs 传给 UNet
            pipe.image_encoder = image_encoder
            pipe.feature_extractor = image_processor
            _load_ip_adapter(pipe)
        else:
            logger.warning("image_encoder 不可用，跳过 IP-Adapter 加载")

        # 6. 初始化 insightface 人脸分析器
        _init_face_analyzer()

        logger.info("SD 1.5 模型加载完成 (IP-Adapter: %s)", "已启用" if ip_adapter_loaded else "未启用")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.exception(e)
        pipe = None


# ── API ───────────────────────────────────────────────────

@app.post("/aigc/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """核心生图接口"""
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    t0 = time.time()
    logger.info(f"收到生图请求, prompt={req.positive_prompt[:50]}...")

    try:
        # 0. 注入性别保持关键词
        positive = req.positive_prompt
        negative = req.negative_prompt
        gender = req.gender.strip().lower() if req.gender else ""
        if gender == "male":
            positive = "1boy, 1man, male, masculine face, same person, same face, " + positive
            if negative:
                negative = "1girl, woman, female, feminine, girly, " + negative
            else:
                negative = "1girl, woman, female, feminine, girly"
        elif gender == "female":
            positive = "1girl, 1woman, female, feminine face, same person, same face, " + positive
            if negative:
                negative = "1boy, man, male, masculine, boyish, " + negative
            else:
                negative = "1boy, man, male, masculine, boyish"

        # 1. 加载原图（同时得到 PIL RGB 和 cv2 BGR 两份）
        init_image, cv2_img = load_image_smart(req.source_image_url)
        # 注意：cv2_img 保留原始分辨率给 insightface，不 resize！
        # 只有给 SD pipeline 的 init_image 才 resize
        init_image = init_image.resize((req.width, req.height))

        # 2. 设置随机种子
        generator = None
        if req.seed >= 0:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(req.seed)

        # 3. 应用 LoRA（如果有）
        apply_loras(pipe, req.lora_weights)

        # 4. IP-Adapter 设置（FaceID Plus：需要 face + clip 两路特征）
        if ip_adapter_loaded:
            # 4a. 提取人脸 ID embedding (InsightFace)
            face_id_embed = extract_face_id_embedding(cv2_img)

            if face_id_embed is None:
                elapsed = int((time.time() - t0) * 1000)
                logger.warning("未检测到人脸，终止生图流程")
                return GenerateResponse(
                    generated_image_url="",
                    generation_time_ms=elapsed,
                    status="failed",
                )

            # 4b. 提取 CLIP image embedding
            clip_embeds = compute_ip_adapter_embeds(init_image)
            if clip_embeds is None:
                elapsed = int((time.time() - t0) * 1000)
                logger.warning("CLIP 特征提取失败，终止生图流程")
                return GenerateResponse(
                    generated_image_url="",
                    generation_time_ms=elapsed,
                    status="failed",
                )

            # 4c. 构造 face_embeds: [2, 1, 512]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            face_tensor = torch.from_numpy(face_id_embed).to(device=device, dtype=dtype)
            face_tensor = face_tensor.view(1, 1, -1)  # [1, 1, 512]
            uncond_face = torch.zeros_like(face_tensor)
            final_face_embeds = torch.cat([uncond_face, face_tensor], dim=0)  # [2, 1, 512]

            # 4d. 构造 clip_embeds: [2, 1, 257, 1280]
            clip_tensor = clip_embeds.to(dtype=dtype).unsqueeze(1)  # [1, 1, 257, 1280]
            uncond_clip = torch.zeros_like(clip_tensor)
            final_clip_embeds = torch.cat([uncond_clip, clip_tensor], dim=0)  # [2, 1, 257, 1280]

            pipe.set_ip_adapter_scale(req.ip_adapter_scale)

            # 终极 Hack：直接把 CLIP 特征注入 UNet 底层，绕过 Pipeline 的 chunk/cat Bug
            proj = pipe.unet.encoder_hid_proj
            if hasattr(proj, "image_projection_layers"):
                proj.image_projection_layers[0].clip_embeds = final_clip_embeds
            else:
                proj.clip_embeds = final_clip_embeds
            logger.info("CLIP 特征已注入 UNet 底层")

            logger.info(f"face_embeds: {final_face_embeds.shape}, clip_embeds: {final_clip_embeds.shape}")
            logger.info(f"IP-Adapter FaceID Plus 组装完毕 (scale={req.ip_adapter_scale})")

        # 5. 推理
        with torch.inference_mode():
            call_kwargs = dict(
                prompt=positive,
                negative_prompt=negative or None,
                image=init_image,
                strength=req.denoising_strength,
                guidance_scale=req.cfg_scale,
                num_inference_steps=30,
                generator=generator,
            )

            if ip_adapter_loaded:
                # 只传 face_embeds（3D 张量），CLIP 已提前注入 UNet
                call_kwargs["ip_adapter_image_embeds"] = [final_face_embeds]
                logger.info("仅传入 face_embeds，CLIP 已空投至 UNet 底层")

            result = pipe(**call_kwargs).images[0]

        # 6. 保存结果
        image_url = save_image_local(result)
        elapsed = int((time.time() - t0) * 1000)

        logger.info(f"生成完成, 耗时={elapsed}ms, IP-Adapter=on, 输出={image_url}")
        return GenerateResponse(
            generated_image_url=image_url,
            generation_time_ms=elapsed,
            status="success",
        )

    except Exception as e:
        elapsed = int((time.time() - t0) * 1000)
        logger.error(f"生成失败: {e}")
        logger.exception(e)
        return GenerateResponse(
            generated_image_url="",
            generation_time_ms=elapsed,
            status="failed",
        )


@app.get("/aigc/health", response_model=HealthResponse)
def health():
    """健康检查接口"""
    return HealthResponse(
        status="ok" if pipe is not None else "error",
        model_loaded=pipe is not None,
        ip_adapter_loaded=ip_adapter_loaded,
        uptime_seconds=int(time.time() - start_time),
    )


@app.get("/")
def root():
    return {
        "msg": "AIGC Visual Engine v2.0",
        "features": {
            "sd_model": "Stable Diffusion 1.5",
            "ip_adapter": "enabled" if ip_adapter_loaded else "disabled",
            "face_detection": "enabled" if face_app is not None else "disabled",
        },
        "docs": "/docs",
    }


# ── 主入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=58000, log_level="info")
