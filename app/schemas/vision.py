"""视觉理解生成的可追溯文字；与视觉向量结果分开。"""

from pydantic import BaseModel, ConfigDict, Field


class VisualExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(max_length=16000)
    formulas: list[str] = Field(default_factory=list, max_length=100)
    symbols: list[str] = Field(default_factory=list, max_length=100)
    descriptions: list[str] = Field(default_factory=list, max_length=100)
    uncertain: list[str] = Field(default_factory=list, max_length=100)

    def readable_text(self) -> str:
        parts = [self.text]
        for label, values in (
            ("公式", self.formulas),
            ("符号", self.symbols),
            ("图表说明", self.descriptions),
            ("识别不确定", self.uncertain),
        ):
            if values:
                parts.append(label + "：\n" + "\n".join(values))
        return "\n".join(parts).strip()
