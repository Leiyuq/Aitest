import re
import streamlit as st
from oss2 import ObjectIterator
from config import AppConfig
from oss_client import get_oss_bucket

# ====================== 项目管理（基于 OSS 前缀） ======================
class ProjectManager:
    @staticmethod
    def get_oss_bucket():
        return get_oss_bucket()

    @staticmethod
    def get_all_projects() -> list:
        """通过列出 OSS 前缀获取所有项目名称"""
        bucket = ProjectManager.get_oss_bucket()
        prefix = f"{AppConfig.BASE_ROOT}/"
        try:
            projects = []
            for obj in ObjectIterator(bucket, prefix=prefix, delimiter='/'):
                # obj.key 格式如 "knowledge_projects/项目名/"
                if obj.key.endswith('/'):
                    proj_name = obj.key[len(prefix):-1]
                    if proj_name:
                        projects.append(proj_name)
            return sorted(projects)
        except Exception:
            return []

    @staticmethod
    def create_project(project_name: str) -> bool:
        if not project_name or not project_name.strip():
            return False
        project_name = project_name.strip()
        if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9_-]+$', project_name):
            st.error("项目名称只能包含中文、字母、数字、下划线和中划线")
            return False
        existing = ProjectManager.get_all_projects()
        if project_name in existing:
            return False
        bucket = ProjectManager.get_oss_bucket()
        marker_key = f"{AppConfig.BASE_ROOT}/{project_name}/.project"
        try:
            bucket.put_object(marker_key, b'')
            bucket.put_object(f"{AppConfig.BASE_ROOT}/{project_name}/knowledge_base/.keep", b'')
            bucket.put_object(f"{AppConfig.BASE_ROOT}/{project_name}/vector_store/.keep", b'')
            return True
        except Exception:
            return False

    @staticmethod
    def get_project_path(project_name: str) -> str:
        return f"{AppConfig.BASE_ROOT}/{project_name}"

    @staticmethod
    def get_kb_path(project_name: str) -> str:
        return f"{AppConfig.BASE_ROOT}/{project_name}/knowledge_base"

    @staticmethod
    def get_vector_store_path(project_name: str) -> str:
        return f"{AppConfig.BASE_ROOT}/{project_name}/vector_store"