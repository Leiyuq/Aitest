import streamlit as st
st.set_page_config(page_title="测试用例智构系统", layout="wide")
import time
import re
from datetime import datetime
import pandas as pd

from config import AppConfig
from project_manager import ProjectManager
from knowledge_base import EnhancedKnowledgeBase
from llm_service import LLMService
from test_case_service import TestCaseService, ExportService
# ====================== 主视图======================
#测试用例智构系统 - 轻量级 RAG（NumPy TF‑IDF）  无用户登录、项目知识库
#阿里云OSS持久化存储   streamlit run app.py --server.address 0.0.0.0
class MainView:
    def __init__(self):
        if "current_project" not in st.session_state:
            projects = ProjectManager.get_all_projects()
            if projects:
                st.session_state.current_project = projects[0]
            else:
                ProjectManager.create_project("默认项目")
                st.session_state.current_project = "默认项目"
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
        if "show_new_project_input" not in st.session_state:
            st.session_state.show_new_project_input = False

        self.current_project = st.session_state.current_project
        self.kb = EnhancedKnowledgeBase(self.current_project)

    def render(self):
        st.title("测试用例智构系统")

        with st.sidebar:
            st.subheader("模型选择")
            model_key = st.selectbox("", AppConfig.get_model_list(), label_visibility="collapsed")
            st.divider()

            col_title, col_new = st.columns([3, 3])
            with col_title:
                st.subheader("项目管理")
            with col_new:
                if st.button("+ 新建项目", key="new_project_btn", use_container_width=True):
                    st.session_state.show_new_project_input = True

            if st.session_state.show_new_project_input:
                new_project_name = st.text_input("项目名称", placeholder="请输入项目名称", key="dialog_project_name")
                col_ok, col_cancel = st.columns(2)
                with col_ok:
                    if st.button("确定", key="confirm_new_project"):
                        if new_project_name and new_project_name.strip():
                            if ProjectManager.create_project(new_project_name):
                                st.success(f"项目 '{new_project_name}' 创建成功")
                                st.session_state.current_project = new_project_name
                                st.session_state.show_new_project_input = False
                                st.rerun()
                            else:
                                st.error("项目创建失败，可能名称已存在")
                        else:
                            st.warning("请输入项目名称")
                with col_cancel:
                    if st.button("取消", key="cancel_new_project", use_container_width=True):
                        st.session_state.show_new_project_input = False
                        st.rerun()

            projects = ProjectManager.get_all_projects()
            if projects:
                selected_project = st.selectbox(
                    "当前项目",
                    options=projects,
                    index=projects.index(self.current_project) if self.current_project in projects else 0,
                    key="project_selector"
                )
                if selected_project != self.current_project:
                    st.session_state.current_project = selected_project
                    st.rerun()
            else:
                st.warning("暂无项目，请先创建")

            st.divider()

            st.subheader("文档上传")
            uploaded_files = st.file_uploader(
                "选择文件（可多选）",
                type=list(AppConfig.ALLOWED_FILE_TYPES),
                accept_multiple_files=True,
                key=f"file_uploader_{st.session_state.uploader_key}",
                help="支持Word/PDF/Excel/PPT/Xmind等格式，图片OCR识别开发中",
            )
            if st.button("上传", key="upload_btn", use_container_width=True):
                if uploaded_files:
                    success_count = 0
                    fail_count = 0
                    for uploaded_file in uploaded_files:
                        success = self.kb.upload_file(uploaded_file.name, uploaded_file.getvalue())
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                    if success_count > 0:
                        st.success(f"成功上传 {success_count} 个文件")
                    if fail_count > 0:
                        st.warning(f"{fail_count} 个文件已存在，未重复上传")
                    st.session_state.uploader_key += 1
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("请先选择文件")

        self._kb_panel()
        st.divider()
        self._gen_panel(model_key)

    def _kb_panel(self):
        st.subheader(f"知识库 - {self.current_project}")
        files = self.kb.get_file_list()
        built_files = self.kb.get_built_files_safe()
        if files:
            for f in files:
                col1, col2, col3, _ = st.columns([0.5, 3, 0.5, 2])
                is_built = f['name'] in built_files
                if is_built:
                    col1.write("✅已构建")
                else:
                    col1.write("⏳未构建")
                col2.write(f"📄 {f['name']} ({f['size']})")
                delete_key = f"del_{self.current_project}_{f['name']}"
                if is_built:
                    with col3.popover("删除", use_container_width=False):
                        st.write(f"确定要删除文档 **{f['name']}** 吗？")
                        st.write("删除后，该文档构建的索引也会被清除。")
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("确认删除", key=f"confirm_{delete_key}", type="primary"):
                                self._delete_file_and_rebuild(f['name'])
                                st.rerun()
                else:
                    if col3.button("删除", key=delete_key):
                        self._delete_file_and_rebuild(f['name'])
                        st.rerun()
        else:
            st.info("暂无文档，请先上传文件")

        col_btn1, col_btn2, _ = st.columns([1, 1, 6])
        with col_btn1:
            if st.button("构建知识库", type="primary", use_container_width=True):
                if not files:
                    st.warning("请先上传文件")
                else:
                    with st.status("正在构建知识库...", expanded=True) as status:
                        prog = st.progress(0)
                        stat = st.empty()
                        def cb(p, msg):
                            prog.progress(p)
                            stat.text(msg)
                        res = self.kb.build_knowledge_base(cb, force=False)
                        if res["status"] == "success":
                            status.update(state="complete")
                            st.toast(res['message'], icon="✅")
                        else:
                            status.update(label="", state="error")
                            st.toast(res['message'], icon="❌")
                    time.sleep(1)
                    st.rerun()
        with col_btn2:
            if st.button("刷新索引", use_container_width=True):
                res = self.kb.refresh_index()
                if res["status"] == "success":
                    st.toast(res["message"], icon="🔄")
                else:
                    st.toast(res["message"], icon="⚠️")
                time.sleep(1)
                st.rerun()

    def _delete_file_and_rebuild(self, filename: str):
        self.kb.delete_file(filename)
        st.rerun()

    def _gen_panel(self, model_key):
        st.subheader("生成测试用例")
        input_type = st.radio("输入方式", ["文本输入", "RDM单号"], horizontal=True)
        if input_type == "文本输入":
            prompt = st.text_area("需求描述", height=150, key="prompt_text")
        else:
            rdm = st.text_input("RDM单号", key="rdm_input")
            prompt = st.text_area("需求描述", value=rdm, height=150, key="prompt_with_rdm") if rdm else ""

        col_check, _ = st.columns([1, 6])
        with col_check:
            use_rag = st.checkbox("启用RAG知识库", value=True,
                                  help="从当前项目的知识库中检索相关内容，提升生成质量")

        col_gen, col_clear, _ = st.columns([1, 1, 6])
        with col_gen:
            generate_clicked = st.button("生成测试用例", type="primary", use_container_width=True)
        with col_clear:
            clear_clicked = st.button("清空结果", use_container_width=True)

        if clear_clicked:
            if "cases" in st.session_state:
                del st.session_state.cases
            st.rerun()

        if generate_clicked:
            if not prompt:
                st.warning("请输入需求描述")
                return

            with st.spinner("生成中..."):
                context = ""
                if use_rag:
                    files = self.kb.get_file_list()
                    if files:
                        if not self.kb.index_loaded:
                            self.kb.refresh_index()
                        if self.kb.index_loaded:
                            with st.expander("RAG检索详情"):
                                results = self.kb.search_knowledge(prompt, top_k=5)
                                if results:
                                    for r in results:
                                        st.write(f"相似度 {r['similarity']:.3f} | 来源 {r['metadata'].get('source','')}")
                                        st.caption(r['content'][:200])
                                else:
                                    st.info("未检索到相关知识")
                            context = self.kb.get_knowledge_context(prompt, max_chunks=5)
                        else:
                            st.info("知识库尚未构建，将直接使用模型生成。如需检索知识，请先构建知识库。")
                    else:
                        st.info("当前项目无文档，将直接使用模型生成。")

                llm = LLMService(model_key)
                resp = llm.generate_cases(prompt, context)

                if resp["status"] == "error":
                    st.error(resp["message"])
                else:
                    st.success(resp["message"])
                    rdm_codes = re.findall(r'[A-Za-z0-9_]+-\d+', prompt)
                    cases = TestCaseService.parse(resp["content"], rdm_codes)
                    if cases:
                        st.session_state.cases = cases
                        st.success(f"解析出 {len(cases)} 个测试用例")
                    else:
                        st.warning("解析失败，显示原始内容")
                        st.code(resp["content"])

        if "cases" in st.session_state:
            self._show_results()

    @staticmethod
    def _show_results():
        cases = st.session_state.cases
        if not cases:
            return
        st.subheader("测试用例")
        df = pd.DataFrame(cases)
        cols = ["RDM单号", "用例ID", "用例名称", "前置条件", "测试步骤", "预期结果"]
        df = df[[c for c in cols if c in df.columns]]
        st.markdown("""
        <style>
        .auto-wrap-table { width: 100%; border-collapse: collapse; }  /* 允许换行 */  /* 长单词断行 */
        .auto-wrap-table th, .auto-wrap-table td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; white-space: normal; word-wrap: break-word; }
        .auto-wrap-table th:nth-child(1), .auto-wrap-table td:nth-child(1) { width: 100px; }   /* RDM单号 */
        .auto-wrap-table th:nth-child(2), .auto-wrap-table td:nth-child(2) { width: 80px; }    /* 用例ID */
        .auto-wrap-table th:nth-child(3), .auto-wrap-table td:nth-child(3) { width: 220px; }   /* 用例名称 */
        .auto-wrap-table th:nth-child(4), .auto-wrap-table td:nth-child(4) { width: 160px; }   /* 前置条件 */
        .auto-wrap-table th:nth-child(5), .auto-wrap-table td:nth-child(5) { width: 320px; }   /* 测试步骤 */
        .auto-wrap-table th:nth-child(6), .auto-wrap-table td:nth-child(6) { width: 320px; }   /* 预期结果 */
        </style>
        """, unsafe_allow_html=True)
        html_table = df.to_html(index=False, classes='auto-wrap-table', escape=False)
        st.markdown(html_table, unsafe_allow_html=True)
        csv = ExportService.to_csv(cases)
        st.download_button(
            "导出 CSV",
            data=csv,
            file_name=f"cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
# ====================== 入口 ======================
def main():
    # 本地开发时确保临时目录存在（实际上不需要创建 OSS 相关目录）
    MainView().render()

if __name__ == "__main__":
    main()