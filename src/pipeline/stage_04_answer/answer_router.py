from typing import Dict, Any, Optional

class StructuredAnswerTemplates:
    """Provides deterministic template responses for standard syndicate intents."""

    TEMPLATES: Dict[str, str] = {
        "MEMBERSHIP": "للتسجيل والانتساب في نقابة المهندسين الأردنيين، يرجى تقديم وثائق التخرج الرسمية، الهوية الشخصية، ومراجعة قسم التسجيل أو البوابة الإلكترونية.",
        "HEALTH_INSURANCE": "للاستفسار عن خدمات التأمين الصحي والشبكة الطبية، يرجى مراجعة دائرة التأمين الصحي أو تقديم طلب عبر البوابة الإلكترونية للنقابة.",
        "PENSION": "للاشتراك في صندوق التقاعد والضمان الاجتماعي أو الاستعلام عن الرواتب التقاعدية، يرجى مراجعة دائرة التقاعد.",
    }

    @classmethod
    def get_template_answer(cls, intent: str) -> Optional[str]:
        return cls.TEMPLATES.get(intent.upper())


class AnswerRouter:
    """Routes queries to DETERMINISTIC or GENERATION (RAG LLM)."""

    def route(self, query_obj: Dict[str, Any], confidence_res: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "route": "GENERATION",
            "llm_required": True,
            "reason": "dynamic_rag_synthesis"
        }
