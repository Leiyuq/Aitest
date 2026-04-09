import streamlit as st
import oss2
# ====================== OSS 客户端 ======================
def get_oss_bucket():
    """从 st.secrets 获取 OSS 配置并返回 Bucket 对象"""
    try:
        auth = oss2.Auth(
            st.secrets["OSS_ACCESS_KEY_ID"],
            st.secrets["OSS_ACCESS_KEY_SECRET"]
        )
        bucket = oss2.Bucket(
            auth,
            st.secrets["OSS_ENDPOINT"],
            st.secrets["OSS_BUCKET_NAME"]
        )
        return bucket
    except Exception as e:
        st.error(f"OSS 配置错误: {e}")
        st.stop()