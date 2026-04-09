import re
from typing import List

# 尝试导入 jieba，如果失败则降级为不使用
try:
    import jieba
    USE_JIEBA = True
except ImportError:
    USE_JIEBA = False
    jieba = None
#===================================通用工具（分词、清理）=======================
def tokenize(text: str) -> List[str]:
    """分词函数，支持中文和英文"""
    text = text.lower()
    if USE_JIEBA and jieba:
        words = jieba.lcut(text)
        # 过滤掉单字符且非字母数字的词
        return [w for w in words if len(w.strip()) > 1 or w.isalnum()]
    else:
        # 降级方案：按中文字符（2个以上）或英文字母（2个以上）匹配
        return re.findall(r'[\u4e00-\u9fa5]{2,}|[a-z]{2,}', text)

def clean_query(query: str) -> str:
    """清理查询字符串，移除日期、编号等噪声"""
    query = re.sub(r'\b[A-Za-z0-9]+[_-][A-Za-z0-9]+\b', '', query)
    query = re.sub(r'\b[A-Z]{2,}[0-9]+\b', '', query)
    query = re.sub(r'\b[A-Z]+-\d+\b', '', query)
    query = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '', query)
    query = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b', '', query)
    query = re.sub(r'\b\d{8}\b', '', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query