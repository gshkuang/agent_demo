#!/usr/bin/env python3
"""
Agent记忆系统 - LangChain版本
核心: ConversationBufferWindowMemory + Chroma向量 + SQLite结构化
"""
import json, sqlite3, time, hashlib
from typing import List, Dict, Optional, Any
from pathlib import Path

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma


class MemoryManager:
    """统一记忆管理器 (LangChain复用)"""

    def __init__(self, agent_name: str = "default", llm=None, max_window: int = 10,
                 db_path: str = None, vector_path: str = None):
        self.agent_name = agent_name
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.db_path = db_path or str(Path.home() / "Desktop/agent_demo/memory/agent_memory.db")
        self.vector_path = vector_path or str(Path.home() / "Desktop/agent_demo/memory/vector_store")

        # LangChain组件
        self.history = ChatMessageHistory()
        self.window = ConversationBufferWindowMemory(k=max_window, return_messages=True, memory_key="chat_history")
        self.summary = ConversationSummaryMemory(llm=self.llm, return_messages=True, memory_key="summary")

        # 向量存储
        try:
            self.vector = Chroma(collection_name=f"agent_{agent_name}", embedding_function=OpenAIEmbeddings(),
                                persist_directory=self.vector_path)
            self.vector_ok = True
        except Exception as e:
            print(f"⚠️ 向量存储失败: {e}")
            self.vector = None
            self.vector_ok = False

        self._init_db()
        print(f"✅ MemoryManager [{agent_name}]")

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        for sql in [
            "CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, timestamp REAL, agent TEXT, input TEXT, output TEXT, summary TEXT)",
            "CREATE TABLE IF NOT EXISTS goals (id TEXT PRIMARY KEY, agent TEXT, goal TEXT, status TEXT, created REAL)",
            "CREATE TABLE IF NOT EXISTS reflections (id TEXT PRIMARY KEY, agent TEXT, task TEXT, outcome TEXT, lesson TEXT, timestamp REAL)"
        ]:
            conn.execute(sql)
        conn.commit()
        conn.close()

    def save(self, user_input: str, agent_output: str, summary: str = None):
        """保存交互到所有记忆层"""
        # 短期记忆
        self.history.add_user_message(user_input)
        self.history.add_ai_message(agent_output)
        self.window.save_context({"input": user_input}, {"output": agent_output})
        self.summary.save_context({"input": user_input}, {"output": agent_output})

        # 向量存储
        mem_id = hashlib.md5(f"{user_input}{time.time()}".encode()).hexdigest()[:12]
        if self.vector_ok:
            self.vector.add_texts([f"Q: {user_input}\nA: {agent_output}"],
                                 [{"agent": self.agent_name, "id": mem_id}])

        # SQLite
        summary = summary or f"{user_input[:40]}... -> {agent_output[:40]}..."
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?)",
                    (mem_id, time.time(), self.agent_name, user_input, agent_output, summary))
        conn.commit()
        conn.close()
        return mem_id

    def recall(self, query: str, top_k: int = 5) -> Dict:
        """回忆: 向量检索 + 短期上下文"""
        results = {"vector": [], "context": "", "summary": ""}
        if self.vector_ok:
            try:
                results["vector"] = [{"content": d.page_content, "meta": d.metadata}
                                    for d in self.vector.similarity_search(query, k=top_k)]
            except Exception as e:
                print(f"⚠️ 检索失败: {e}")
        try:
            results["context"] = self.window.load_memory_variables({}).get("chat_history", "")
            results["summary"] = self.summary.load_memory_variables({}).get("summary", "")
        except:
            pass
        return results

    def get_vars(self) -> Dict[str, Any]:
        """获取记忆变量 (注入Prompt)"""
        try:
            return {
                "chat_history": self.window.load_memory_variables({}).get("chat_history", []),
                "summary": self.summary.load_memory_variables({}).get("summary", ""),
            }
        except:
            return {"chat_history": [], "summary": ""}

    def goal(self, text: str) -> str:
        gid = hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:12]
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT OR REPLACE INTO goals VALUES (?,?,?,?,?)",
                    (gid, self.agent_name, text, "in_progress", time.time()))
        conn.commit()
        conn.close()
        return gid

    def active_goals(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT * FROM goals WHERE agent=? AND status='in_progress' ORDER BY created DESC",
                           (self.agent_name,)).fetchall()
        conn.close()
        return [{"id": r[0], "goal": r[2], "status": r[3]} for r in rows]

    def reflect(self, task: str, outcome: str, lesson: str):
        rid = hashlib.md5(f"{task}{time.time()}".encode()).hexdigest()[:12]
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO reflections VALUES (?,?,?,?,?,?)",
                    (rid, self.agent_name, task, outcome, lesson, time.time()))
        conn.commit()
        conn.close()
        if self.vector_ok:
            self.vector.add_texts([f"任务:{task}\n结果:{outcome}\n经验:{lesson}"],
                                 [{"agent": self.agent_name, "type": "reflection"}])

    def stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        stats = {
            "messages": len(self.history.messages),
            "vector_ok": self.vector_ok,
            "events": conn.execute("SELECT COUNT(*) FROM events WHERE agent=?", (self.agent_name,)).fetchone()[0],
            "goals": conn.execute("SELECT COUNT(*) FROM goals WHERE agent=? AND status='in_progress'", (self.agent_name,)).fetchone()[0],
            "reflections": conn.execute("SELECT COUNT(*) FROM reflections WHERE agent=?", (self.agent_name,)).fetchone()[0],
        }
        conn.close()
        return stats


def demo():
    print("=" * 50)
    print("Memory System - LangChain")
    print("=" * 50)

    try:
        import os
        llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"], temperature=0)
        mem = MemoryManager(agent_name="ethon", llm=llm, max_window=5)

        # 记录交互
        for q, a in [("分析沪电股份ROE", "ROE 18.5%"), ("三安光电股东", "香港中央结算新进")]:
            print(f"  ✅ 记录: {mem.save(q, a)[:8]}")

        # 回忆
        results = mem.recall("沪电股份财务", top_k=2)
        print(f"\n向量检索: {len(results['vector'])}条")

        # 目标
        mem.goal("完成本周组合分析")
        print(f"目标数: {len(mem.active_goals())}")

        # 反思
        mem.reflect("ROE分析", "成功", "用杜邦分析法")
        print(f"统计: {mem.stats()}")

    except Exception as e:
        print(f"\n⚠️ 需要OPENAI_API_KEY: {e}")


if __name__ == "__main__":
    demo()
