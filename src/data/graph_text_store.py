from typing import Dict
import unicodedata
import numpy as np
import pandas as pd


class TextData:
    def __init__(self, ent_desc: np.ndarray, rel_desc: np.ndarray) -> None:
        self.ent_desc = ent_desc
        self.rel_desc = rel_desc


class TextUtil:
    @staticmethod
    def _simplify_text_data(data: pd.Series):
        data = data.apply(lambda x: x.replace("\\n", " "))
        data = data.apply(lambda x: x.replace("\\t", " "))
        data = data.apply(lambda x: x.replace('\\"', '"'))
        data = data.apply(lambda x: x.replace("\\'", "'"))
        data = data.apply(lambda x: x.replace("\\", ""))
        data = data.apply(TextUtil._remove_accented_chars)
        return data

    @staticmethod
    def _remove_accented_chars(text):
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )


class TextStore:
    def __init__(self, ent_desc_path: str, rel_desc_path: str) -> None:
        self.ent_desc = self.load(ent_desc_path)
        self.rel_desc = self.load(rel_desc_path)

    def load(self, path: str) -> pd.DataFrame:
        desc = pd.read_csv(
            path,
            names=["code", "description"],
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )

        desc["description"] = TextUtil._simplify_text_data(desc["description"])

        return desc

    def _ent_desc_from_mapping(self, ent_mapping: Dict[str, int]) -> np.ndarray:
        ent_desc = np.zeros((len(ent_mapping), 1), dtype=object)
        for ent, idx in ent_mapping.items():
            if ent in self.ent_desc["code"].values:
                ent_desc[idx] = self.ent_desc[self.ent_desc["code"] == ent][
                    "description"
                ].values[0]
            else:
                ent_desc[idx] = f"Unknown entity {idx}"
        return ent_desc

    def _rel_desc_from_mapping(self, rel_mapping: Dict[str, int]) -> np.ndarray:
        rel_desc = np.zeros((len(rel_mapping), 1), dtype=object)
        for rel, idx in rel_mapping.items():
            if rel in self.rel_desc["code"].values:
                rel_desc[idx] = self.rel_desc[self.rel_desc["code"] == rel][
                    "description"
                ].values[0]
            else:
                rel_desc[idx] = f"Unknown relation {idx}"
        return rel_desc

    def desc_from_mapping(
        self, ent_mapping: Dict[str, int], rel_mapping: Dict[str, int]
    ) -> TextData:
        ent_desc = self._ent_desc_from_mapping(ent_mapping)
        rel_desc = self._rel_desc_from_mapping(rel_mapping)

        return TextData(ent_desc, rel_desc)


class GrailTextStore(TextStore):
    def __init__(
        self, ent_desc_path: str, ent_short_desc_path: str, rel_desc_path: str
    ) -> None:
        self.ent_desc = self.load(ent_desc_path)
        self.ent_short_desc = self.load(ent_short_desc_path)
        self.rel_desc = self.load(rel_desc_path)

    def _ent_desc_from_mapping(self, ent_mapping: Dict[str, int]) -> np.ndarray:
        # ent desc 优先使用 ent_short_desc，如果没有则使用 ent_desc
        ent_desc = np.zeros((len(ent_mapping), 1), dtype=object)
        for ent, idx in ent_mapping.items():
            if ent in self.ent_short_desc["code"].values:
                ent_desc[idx] = self.ent_short_desc[self.ent_short_desc["code"] == ent][
                    "description"
                ].values[0]
            elif ent in self.ent_desc["code"].values:
                # 只取第一句话
                ent_desc[idx] = (
                    self.ent_desc[self.ent_desc["code"] == ent]["description"]
                    .values[0]
                    .split(".")[0]
                )
            else:
                ent_desc[idx] = f"Unknown entity {idx}"
        return ent_desc
    
    def desc_from_mapping(
        self, ent_mapping: Dict[str, int], rel_mapping: Dict[str, int]
    ) -> TextData:
        ent_desc = self._ent_desc_from_mapping(ent_mapping)
        rel_desc = self._rel_desc_from_mapping(rel_mapping)

        return TextData(ent_desc, rel_desc)
