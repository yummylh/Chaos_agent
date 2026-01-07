import os
import json
import time
from datetime import datetime

HISTORY_DIR = "./chat_history"

if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

def generate_session_id():
    """生成唯一的会话ID (使用时间戳)"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_conversation(session_id, messages):
    """保存当前会话到 JSON 文件"""
    if not messages:
        return
    
    # 取第一句话作为标题，如果太长就截断
    title = "新对话"
    for msg in messages:
        if msg["role"] == "user":
            title = msg["content"][:20] # 取前20个字
            break
            
    file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    
    data = {
        "id": session_id,
        "title": title,
        "timestamp": time.time(),
        "messages": messages
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_conversation(session_id):
    """读取指定会话"""
    file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["messages"]
    return []

def get_history_list():
    """获取所有历史会话列表 (按时间倒序)"""
    files = [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]
    sessions = []
    
    for f in files:
        path = os.path.join(HISTORY_DIR, f)
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
                sessions.append({
                    "id": data.get("id", f.replace(".json", "")),
                    "title": data.get("title", "未命名对话"),
                    "timestamp": data.get("timestamp", 0)
                })
        except:
            continue
            
    # 按时间倒序排列（最新的在上面）
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return sessions

def delete_conversation(session_id):
    """删除会话"""
    file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)