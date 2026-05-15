"""
企业智能知识库 - Streamlit 前端
提供文档管理、智能问答、Agent 助手、知识库概览四个页面

启动方式: streamlit run streamlit_app.py
"""

import streamlit as st
import httpx
import json

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="企业智能知识库",
    page_icon="📚",
    layout="wide",
)

# ==================== 常量 ====================

DEFAULT_API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 120.0

# ==================== Sidebar ====================

st.sidebar.title("📚 企业智能知识库")
st.sidebar.divider()

api_base = st.sidebar.text_input("API 地址", value=DEFAULT_API_BASE)
st.sidebar.divider()

page = st.sidebar.radio(
    "导航",
    ["📄 文档管理", "💬 智能问答", "🤖 Agent 助手", "📊 知识库概览"],
)

# ==================== 工具函数 ====================


def api_get(path: str):
    """GET 请求"""
    try:
        resp = httpx.get(f"{api_base}{path}", timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(f"无法连接到后端服务: {api_base}")
        return None
    except Exception as e:
        st.error(f"请求失败: {e}")
        return None


def api_post(path: str, json_data: dict = None, files: dict = None):
    """POST 请求"""
    try:
        resp = httpx.post(
            f"{api_base}{path}",
            json=json_data,
            files=files,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(f"无法连接到后端服务: {api_base}")
        return None
    except Exception as e:
        st.error(f"请求失败: {e}")
        return None


def api_delete(path: str):
    """DELETE 请求"""
    try:
        resp = httpx.delete(f"{api_base}{path}", timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(f"无法连接到后端服务: {api_base}")
        return None
    except Exception as e:
        st.error(f"请求失败: {e}")
        return None


# ==================== 页面：文档管理 ====================


def document_management_page():
    st.header("📄 文档管理")

    # 上传区
    st.subheader("上传文档")
    uploaded_file = st.file_uploader(
        "选择文件",
        type=["txt", "pdf", "docx", "md"],
        help="支持 .txt, .pdf, .docx, .md 格式",
    )

    if uploaded_file and st.button("上传", type="primary"):
        with st.spinner("正在上传并处理..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            result = api_post("/api/v1/documents/", files=files)
            if result:
                st.success(f"上传成功！文档 ID: {result.get('id', '未知')}")
                st.rerun()

    st.divider()

    # 文档列表
    st.subheader("文档列表")
    data = api_get("/api/v1/documents/")
    if data:
        docs = data.get("documents", [])
        if not docs:
            st.info("暂无文档，请上传文档开始使用")
        else:
            for doc in docs:
                col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
                col1.write(f"**{doc.get('filename', '未知')}**")
                col2.write(doc.get("file_type", ""))
                col3.write(f"{doc.get('chunk_count', 0)} 块")
                doc_id = doc.get("id", doc.get("document_id", ""))
                if col4.button("删除", key=f"del_{doc_id}"):
                    with st.spinner("删除中..."):
                        api_delete(f"/api/v1/documents/{doc_id}")
                    st.rerun()


# ==================== 页面：智能问答 ====================


def qa_page():
    st.header("💬 智能问答")

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # 显示历史
    for msg in st.session_state.qa_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 参考来源"):
                    for src in msg["sources"]:
                        name = src.get("document_name", "未知文件")
                        score = src.get("score", 0)
                        content = src.get("content", "")[:300]
                        st.write(f"**{name}** (相关度: {score:.2f})")
                        st.write(content)
                        st.divider()

    # 输入
    question = st.chat_input("请输入您的问题...")
    if question:
        st.session_state.qa_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索和生成回答..."):
                result = api_post("/api/v1/qa/", json_data={
                    "question": question,
                    "top_k": 5,
                    "include_sources": True,
                })
                if result:
                    st.write(result["answer"])
                    sources = result.get("sources", [])
                    st.session_state.qa_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": sources if sources else [],
                    })
                else:
                    st.error("问答请求失败")
                    st.session_state.qa_history.append({
                        "role": "assistant",
                        "content": "抱歉，请求失败，请检查后端服务。",
                    })


# ==================== 页面：Agent 助手 ====================


def agent_page():
    st.header("🤖 Agent 智能助手")
    st.caption("Agent 会自动选择工具来回答问题：搜索知识库、问答、列出文档等")

    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []

    # 显示历史
    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("steps"):
                with st.expander("🔍 推理过程"):
                    for i, step in enumerate(msg["steps"]):
                        step_type = step.get("step_type", "")
                        if step_type == "tool_call":
                            tool_name = step.get("tool_name", "未知")
                            tool_args = step.get("tool_arguments", {})
                            tool_result = step.get("tool_result", "")[:200]
                            st.write(f"**步骤 {i+1}**: 调用工具 `{tool_name}`")
                            st.code(
                                json.dumps(tool_args, ensure_ascii=False, indent=2),
                                language="json",
                            )
                            st.write(f"结果: {tool_result}")
                            st.divider()
                        elif step_type == "final_answer":
                            st.write("**最终回答生成完毕**")

    # 输入
    message = st.chat_input("请输入您的问题...")
    if message:
        st.session_state.agent_history.append({"role": "user", "content": message})
        with st.chat_message("user"):
            st.write(message)

        with st.chat_message("assistant"):
            with st.spinner("Agent 正在思考..."):
                result = api_post(
                    "/api/v1/agent/",
                    json_data={"message": message, "max_steps": 5},
                )
                if result:
                    st.write(result["answer"])
                    steps = result.get("steps", [])
                    if steps:
                        with st.expander("🔍 推理过程"):
                            for i, step in enumerate(steps):
                                step_type = step.get("step_type", "")
                                if step_type == "tool_call":
                                    tool_name = step.get("tool_name", "未知")
                                    tool_args = step.get("tool_arguments", {})
                                    tool_result = step.get("tool_result", "")[:200]
                                    st.write(f"**步骤 {i+1}**: 调用工具 `{tool_name}`")
                                    st.code(
                                        json.dumps(tool_args, ensure_ascii=False, indent=2),
                                        language="json",
                                    )
                                    st.write(f"结果: {tool_result}")
                                    st.divider()
                                elif step_type == "final_answer":
                                    st.write("**最终回答生成完毕**")

                    st.session_state.agent_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "steps": steps,
                    })
                else:
                    st.error("Agent 请求失败")
                    st.session_state.agent_history.append({
                        "role": "assistant",
                        "content": "抱歉，请求失败，请检查后端服务。",
                    })


# ==================== 页面：知识库概览 ====================


def stats_page():
    st.header("📊 知识库概览")

    # 系统状态
    status = api_get("/api/v1/status")
    if status:
        col1, col2, col3 = st.columns(3)
        col1.metric("系统状态", status.get("status", "未知"))
        col2.metric("模型", status.get("model", "未知"))
        llm_ok = status.get("llm_configured", False)
        col3.metric("LLM 已配置", "是" if llm_ok else "否")

    st.divider()

    # 统计信息
    stats = api_get("/api/v1/documents/stats")
    if stats:
        col1, col2, col3 = st.columns(3)
        col1.metric("文档总数", stats.get("document_count", 0))
        col2.metric("文档块总数", stats.get("total_chunks", 0))
        col3.metric("ChromaDB 记录数", stats.get("collection_count", 0))

        st.divider()

        docs = stats.get("documents", [])
        if docs:
            st.subheader("文档详情")
            st.dataframe(
                docs,
                column_config={
                    "id": "文档 ID",
                    "filename": "文件名",
                    "file_type": "类型",
                    "chunk_count": "块数",
                    "uploaded_at": "上传时间",
                },
                use_container_width=True,
            )
        else:
            st.info("暂无文档数据")


# ==================== 路由 ====================

if page == "📄 文档管理":
    document_management_page()
elif page == "💬 智能问答":
    qa_page()
elif page == "🤖 Agent 助手":
    agent_page()
elif page == "📊 知识库概览":
    stats_page()
