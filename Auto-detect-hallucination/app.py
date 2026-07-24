"""
客服回复幻觉检测 — Web 界面后端
FastAPI + SSE 实时推送检测进度
"""

import asyncio
import json
import threading
import uuid
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from detector import LLMDetector, MockDetector

# ── FastAPI 实例 ──
app = FastAPI(title="客服回复幻觉检测", version="1.0")

# ── 会话存储（生产环境应换 Redis） ──
sessions: dict = {}          # session_id → {files, results, progress}
sessions_lock = threading.Lock()

# ── 静态文件 ──
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ═══════════════════════════════════════════════════════════════
#  API 路由
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面。"""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/models")
async def list_models():
    """返回本地 Ollama 可用模型列表。"""
    try:
        import ollama
        models = ollama.list()
        names = [m["name"] for m in models.get("models", [])]
        # 过滤掉 embedding 模型，只保留 chat 模型
        chat_models = [n for n in names if "bge" not in n.lower()]
        return {"models": chat_models, "default": "qwen3:8b"}
    except Exception as e:
        # Ollama 不可用时返回默认列表
        return {
            "models": ["qwen3:8b", "qwen2.5:7b", "deepseek-r1:14b", "llama3.1:8b"],
            "default": "qwen3:8b",
            "error": str(e),
        }


@app.post("/upload")
async def upload(
    replies: UploadFile = File(...),
    ground_truth: Optional[UploadFile] = File(None),
):
    """上传检测数据和可选的标注答案。"""
    session_id = uuid.uuid4().hex[:8]

    # 读取 replies
    replies_bytes = await replies.read()
    try:
        samples = json.loads(replies_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        return JSONResponse({"error": f"JSON 解析失败: {e}"}, status_code=400)

    if not isinstance(samples, list) or len(samples) == 0:
        return JSONResponse({"error": "数据格式错误：需要 JSON 数组"}, status_code=400)

    # 验证必要字段
    required = ["id", "user_question", "system_reply", "knowledge_base"]
    missing = [f for f in required if f not in samples[0]]
    if missing:
        return JSONResponse(
            {"error": f"数据缺少必要字段: {missing}，需要 {required}"},
            status_code=400,
        )

    # 读取 ground_truth（可选）
    gt_data = None
    if ground_truth:
        gt_bytes = await ground_truth.read()
        try:
            gt_data = json.loads(gt_bytes.decode("utf-8"))
        except json.JSONDecodeError:
            pass  # 不强制，gt 解析失败就当没有

    with sessions_lock:
        sessions[session_id] = {
            "samples": samples,
            "ground_truth": gt_data,
            "results": [],
            "progress": {"current": 0, "total": len(samples), "status": "ready"},
        }

    return {
        "session_id": session_id,
        "total": len(samples),
        "has_ground_truth": gt_data is not None,
    }


@app.get("/detect/{session_id}")
async def detect_stream(session_id: str, request: Request, mock: bool = Query(False), model: str = Query("qwen3:8b")):
    """SSE 流式端点：逐条检测并推送进度。"""
    if session_id not in sessions:
        return JSONResponse({"error": "会话不存在或已过期"}, status_code=404)

    session = sessions[session_id]

    async def event_generator():
        samples = session["samples"]
        results = []
        total = len(samples)

        # 创建检测器
        try:
            if mock:
                detector = MockDetector()
            else:
                detector = LLMDetector(model=model)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'检测器初始化失败: {e}'}, ensure_ascii=False)}\n\n"
            return

        # 逐条检测
        for i, sample in enumerate(samples):
            # 检查客户端是否断开
            if await request.is_disconnected():
                break

            sid = sample.get("id", f"item_{i}")
            t0 = time.time()

            try:
                result = detector.detect(sample)
            except Exception as e:
                result = {
                    "id": sid,
                    "is_hallucination": None,
                    "hallucination_type": None,
                    "severity": None,
                    "detail": f"检测异常: {str(e)}",
                }

            elapsed = time.time() - t0
            results.append(result)

            # 推送单条结果
            event_data = {
                "type": "progress",
                "current": i + 1,
                "total": total,
                "item": {
                    "id": sid,
                    "is_hallucination": result.get("is_hallucination"),
                    "hallucination_type": result.get("hallucination_type"),
                    "severity": result.get("severity"),
                    "detail": result.get("detail", ""),
                    "elapsed": round(elapsed, 1),
                },
            }
            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

            # 给前端渲染的间隙
            await asyncio.sleep(0.05)

        # 全部完成
        session["results"] = results

        # 如果有 ground truth，计算指标
        evaluation = None
        if session["ground_truth"]:
            try:
                evaluation = compute_evaluation(results, session["ground_truth"])
                session["evaluation"] = evaluation
            except Exception as e:
                evaluation = {"error": str(e)}

        summary = {
            "type": "complete",
            "total": total,
            "hallucination_count": sum(1 for r in results if r.get("is_hallucination")),
            "normal_count": sum(1 for r in results if not r.get("is_hallucination")),
            "evaluation": evaluation,
        }
        yield f"data: {json.dumps(summary, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/download/{session_id}/{file_type}")
async def download(session_id: str, file_type: str):
    """下载检测结果。file_type: results | evaluation"""
    if session_id not in sessions:
        return JSONResponse({"error": "会话不存在"}, status_code=404)

    session = sessions[session_id]

    if file_type == "results":
        data = session.get("results", [])
        filename = "detection_results.json"
    elif file_type == "evaluation":
        data = session.get("evaluation")
        if not data:
            return JSONResponse({"error": "未上传 ground truth，无评估数据"}, status_code=404)
        filename = "evaluation.json"
    else:
        return JSONResponse({"error": "file_type 仅支持 results 或 evaluation"}, status_code=400)

    return JSONResponse(data, headers={
        "Content-Disposition": f"attachment; filename={filename}",
    })


# ═══════════════════════════════════════════════════════════════
#  评估计算（复用 evaluation_report.py 逻辑）
# ═══════════════════════════════════════════════════════════════

def _type_semantic_match(det_type, gt_type):
    """宽松的类型匹配。"""
    if det_type is None or gt_type is None:
        return det_type == gt_type
    mapping = {
        "政策规则幻觉": ["政策编造", "政策偏差"],
        "产品参数幻觉": ["参数编造"],
        "系统能力幻觉": ["能力越界"],
        "实体信息幻觉": ["信息编造"],
        "优惠活动幻觉": ["优惠编造"],
        "安全合规幻觉": ["安全误导"],
        "关键信息遗漏": ["信息遗漏"],
    }
    if det_type in mapping:
        return gt_type in mapping[det_type]
    return det_type == gt_type


def compute_evaluation(results, ground_truth):
    """计算评估指标。"""
    gt_map = {item["id"]: item for item in ground_truth}
    total = len(results)
    tp = tn = fp = fn = 0
    type_correct = 0
    details = []

    for r in results:
        rid = r["id"]
        gt = gt_map.get(rid)
        if not gt:
            continue
        det_hall = r.get("is_hallucination")
        gt_hall = gt["is_hallucination"]

        if det_hall and gt_hall:
            tp += 1
        elif not det_hall and not gt_hall:
            tn += 1
        elif det_hall and not gt_hall:
            fp += 1
        elif not det_hall and gt_hall:
            fn += 1

        is_correct = (det_hall == gt_hall)
        if det_hall and gt_hall and _type_semantic_match(r.get("hallucination_type"), gt.get("hallucination_type")):
            type_correct += 1

        details.append({
            "id": rid,
            "correct": is_correct,
            "detection": det_hall,
            "ground_truth": gt_hall,
        })

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "type_correct": type_correct,
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════
#  启动入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8081, reload=True)
