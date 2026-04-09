import re
import io
import pandas as pd
from typing import List, Dict

# ====================== 用例解析 ======================
class TestCaseService:
    @staticmethod
    def parse(content: str, rdm_codes: List[str]) -> List[Dict]:
        cases = []
        blocks = re.split(r'\n\s*\n', content.strip())
        rdm_idx = 0
        for block in blocks:
            if not any(k in block for k in ["用例ID", "用例名称", "测试步骤"]):
                continue
            case = {"RDM单号": "", "用例ID": "", "用例名称": "", "前置条件": "", "测试步骤": "", "预期结果": ""}
            patterns = {
                "用例ID": r"用例ID[:：]\s*([^\n]+)",
                "RDM单号": r"RDM单号[:：]\s*([^\n]+)",
                "用例名称": r"用例名称[:：]\s*([^\n]+)",
                "前置条件": r"前置条件[:：]\s*([^\n]+(?:\n(?!用例|rdm|测试|预期)[^\n]+)*)",
                "测试步骤": r"测试步骤[:：]\s*([^\n]+(?:\n(?!用例|rdm|前置|预期)[^\n]+)*)",
                "预期结果": r"预期结果[:：]\s*([^\n]+(?:\n(?!用例|rdm|前置|测试)[^\n]+)*)"
            }
            for field, pat in patterns.items():
                m = re.search(pat, block, re.DOTALL)
                if m:
                    value = m.group(1).strip()
                    if field in ["用例ID", "RDM单号"]:
                        value = re.sub(r'^\[|]$', '', value)
                    case[field] = re.sub(r'\n+', ' ', value)
            if not case["RDM单号"] and rdm_codes:
                case["RDM单号"] = rdm_codes[rdm_idx % len(rdm_codes)]
                rdm_idx += 1
            if case["用例ID"]:
                cases.append(case)
        return cases

# ====================== 用例导出 ======================
class ExportService:
    @staticmethod
    def to_csv(cases: List[Dict]) -> bytes:
        df = pd.DataFrame(cases)
        cols = ["RDM单号", "用例ID", "用例名称", "前置条件", "测试步骤", "预期结果"]
        df = df[[c for c in cols if c in df.columns]]
        buf = io.BytesIO()
        df.to_csv(buf, index=False, encoding='utf-8-sig')
        return buf.getvalue()