from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import os, json, re

# ---- Env setup ----
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2877717 Aware CIN Table Assessment")

# ===== Models =====
class SelectItem(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: str

    @field_validator("used_fields", "suggested_fields")
    @classmethod
    def no_none_elems(cls, v: List[str]) -> List[str]:
        return [x for x in v if x is not None]

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: str
    selects: List[SelectItem] = Field(default_factory=list)

# ===== Table mappings for SAP Note 2877717 =====
OLD_TO_NEW_TABLE_MAP = {
    "J_1IMOCUST": {"new_table": "KNA1", "key_field": "KUNNR"},
    "J_1IMOVEND": {"new_table": "LFA1", "key_field": "LIFNR"},
}
CIN_OLD_FIELDS = {"VEN_CLASS", "J_1IPANNO", "J_1IEXCIVE"}

# ===== Summariser that flags SAP Note 2877717 risks =====
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    """
    Detect CIN table SELECT/UPDATE/MODIFY/INSERT and missing migration to BP master tables.
    """
    tables_count: Dict[str, int] = {}
    total = len(unit.selects)
    flagged = []
    for s in unit.selects:
        tbl_upper = s.table.upper()
        tables_count[tbl_upper] = tables_count.get(tbl_upper, 0) + 1
        if tbl_upper in OLD_TO_NEW_TABLE_MAP:
            mapping = OLD_TO_NEW_TABLE_MAP[tbl_upper]
            flagged.append({
                "table": s.table,
                "target": s.target_name,
                "reason": f"Usage of {tbl_upper} detected. Replace with {mapping['new_table']} per SAP Note 2877717."
            })

    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "stats": {
            "total_statements": total,
            "tables_frequency": tables_count,
            "note_2877717_flags": flagged
        }
    }

# ===== Prompt for SAP Note–specific fix =====
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 2877717. Output strict JSON only."

USER_TEMPLATE = """
You are assessing ABAP code usage in light of SAP Note 2877717 (CIN to BP Master migration).

From S/4HANA 1909 onwards, tables J_1IMOCUST and J_1IMOVEND are obsolete.
- J_1IMOCUST → use KNA1 (Business Partner Customer)
- J_1IMOVEND → use LFA1 (Business Partner Vendor)
- CIN-specific fields VEN_CLASS, J_1IPANNO, J_1IEXCIVE removed or mapped differently.

We provide program/include/unit metadata, and statement analysis.

Your tasks:
1) Produce a concise **assessment** highlighting:
   - Which statements reference legacy tables.
   - Why migration is needed.
   - Potential functional and data impact.
2) Produce an **LLM remediation prompt** to:
   - Scan ABAP code in this unit for J_1IMOCUST/J_1IMOVEND usage.
   - Replace with respective BP Master table (KNA1/LFA1).
   - Adjust SELECT fields: remove obsolete CIN fields, ensure key fields are present.
   - Add `TODO` comments where manual mapping of fields is required.
   - Output strictly in JSON with: original_code, remediated_code, changes[].

Return ONLY strict JSON:
{{
  "assessment": "<concise note 2877717 impact>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

Analysis:
{plan_json}

selects (JSON):
{selects_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL)
parser = JsonOutputParser()
chain = prompt | llm | parser

# ===== LLM Call =====
def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    plan = summarize_selects(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    selects_json = json.dumps([s.model_dump() for s in unit.selects], ensure_ascii=False, indent=2)

    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name,
            "plan_json": plan_json,
            "selects_json": selects_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ===== API =====
@app.post("/assess-cin-migration")
def assess_cin_migration(units: List[Unit]) -> List[Dict[str, Any]]:
    out = []
    for u in units:
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        obj.pop("selects", None)
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}