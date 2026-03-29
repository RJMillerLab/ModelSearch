import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.utils import _get_models_to_tables_batch_sql, load_modelid_to_csvlist, resolve_table_path


JOB_META_FILENAME = "job_meta.json"


@dataclass
class JobPaths:
    jobs_dir: str
    job_id: str

    @property
    def job_dir(self) -> str:
        return os.path.join(self.jobs_dir, self.job_id)

    @property
    def job_meta_path(self) -> str:
        return os.path.join(self.job_dir, JOB_META_FILENAME)

    @property
    def query2modelcard_path(self) -> str:
        return os.path.join(self.job_dir, "query2modelcard.json")

    def card2tab2card_path(self, search_type: str) -> str:
        return os.path.join(self.job_dir, f"card2tab2card_{search_type}.json")


@dataclass
class JobMeta:
    job_id: str
    query: str
    top_k: int
    model_top_k: int
    table_search_k: int
    table_resources: List[str]
    use_by_type: bool
    timestamp: str
    running_time_seconds: float

    @classmethod
    def load(cls, path: str) -> "JobMeta":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            job_id=data["job_id"],
            query=data["query"],
            top_k=data["top_k"],
            model_top_k=data["model_top_k"],
            table_search_k=data["table_search_k"],
            table_resources=data["table_resources"],
            use_by_type=data["use_by_type"],
            timestamp=data["timestamp"],
            running_time_seconds=data["running_time_seconds"],
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": self.job_id,
                    "query": self.query,
                    "top_k": self.top_k,
                    "model_top_k": self.model_top_k,
                    "table_search_k": self.table_search_k,
                    "table_resources": self.table_resources,
                    "use_by_type": self.use_by_type,
                    "timestamp": self.timestamp,
                    "running_time_seconds": self.running_time_seconds,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "query": self.query,
            "top_k": self.top_k,
            "model_top_k": self.model_top_k,
            "table_search_k": self.table_search_k,
            "table_resources": self.table_resources,
            "use_by_type": self.use_by_type,
            "timestamp": self.timestamp,
            "running_time_seconds": self.running_time_seconds,
        }

    @classmethod
    def save_for_job(cls, *, jobs_dir: str, job_id: str, query: str, top_k: int, model_top_k: int, table_search_k: int, table_resources: List[str], use_by_type: bool, running_time_seconds: float) -> None:
        paths = JobPaths(jobs_dir, job_id)
        cls(
            job_id=job_id,
            query=query,
            top_k=top_k,
            model_top_k=model_top_k,
            table_search_k=table_search_k,
            table_resources=table_resources,
            use_by_type=use_by_type,
            timestamp=datetime.now().isoformat(),
            running_time_seconds=round(running_time_seconds, 3),
        ).save(paths.job_meta_path)


@dataclass
class Query2ModelCardFile:
    query: str
    top_k: int
    job_id: str
    results: Dict[str, Any]

    @classmethod
    def load(cls, path: str) -> "Query2ModelCardFile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            query=data["query"],
            top_k=data["top_k"],
            job_id=data["job_id"],
            results=data["results"],
        )

    @property
    def dense(self) -> List[Any]:
        return self.results["dense"]

    @property
    def sparse(self) -> List[Any]:
        return self.results["sparse"]

    @property
    def hybrid(self) -> List[Any]:
        return self.results["hybrid"]

    def items_for_mode(self, mode: str) -> List[Any]:
        return self.results[mode]

    @property
    def seed_model_id(self) -> str:
        return str(self.dense[0]).strip()

    def neighbor_model_ids(self, mode: str, limit: int) -> List[str]:
        model_ids = []
        for item in self.results[mode]:
            model_id = str(item).strip()
            if model_id == self.seed_model_id:
                continue
            model_ids.append(model_id)
        return list(dict.fromkeys(model_ids))[:limit]

    def build_preview(self, *, query: str, table_resources: List[str], mode: str, max_models: int) -> Dict[str, Any]:
        model_ids = self.neighbor_model_ids(mode, max_models)
        model_to_all_tables = _get_models_to_tables_batch_sql(model_ids, resources=table_resources)
        model_to_table_paths = {mid: [name for name in list(dict.fromkeys(str(x).strip() for x in model_to_all_tables[mid] if str(x).strip())) if resolve_table_path(name)] for mid in model_ids}
        model_to_table_paths = {mid: paths for mid, paths in model_to_table_paths.items() if paths}
        models_with_tables = list(model_to_table_paths.keys())
        table_paths = list(dict.fromkeys(path for paths in model_to_table_paths.values() for path in paths))
        return {
            "query2modelcard_retrieval_mode": mode,
            "models_with_tables": models_with_tables,
            "model_ids": model_ids,
            "model_to_table_paths": model_to_table_paths,
            "table_paths": table_paths,
            "job_context": {
                "query": query,
                "table_search_seed_model_id": self.seed_model_id,
            },
            "stats": {
                "total_model_ids": len(model_ids),
                "models_with_tables": len(models_with_tables),
                "total_unique_tables": len(table_paths),
            },
        }


class Query2Tab2CardFullMap:
    def __init__(self, path: str):
        self.path = path
        self.full_map = self.load(path)
        self.q2c, self.card2tab, self.tab2tab, self.tab2card, self.reranked, self.query, self.seed_models, self.query_tables, self.candidate_pool, self.retrieved_unique, self.model_to_all_table_paths = self.split()

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert frozenset(data.keys()) == frozenset({"query2card_map", "model_rerank_map", "card2tab_map", "tab2tab_map", "tab2card_map"})
        return data

    def split(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str], str, List[str], List[str], List[str], List[str], Dict[str, List[str]]]:
        q2c = self.full_map["query2card_map"]
        card2tab = self.full_map["card2tab_map"]
        tab2tab = self.full_map["tab2tab_map"]
        tab2card = self.full_map["tab2card_map"]
        reranked = self.full_map["model_rerank_map"]
        query = next(iter(q2c.keys()))
        seed_models = [str(x).strip() for x in q2c[query]]
        query_tables = list(dict.fromkeys([str(t) for t in card2tab[seed_models[0]]]))
        candidate_pool = list(dict.fromkeys(x for v in tab2card.values() for x in v))
        retrieved_unique = list(dict.fromkeys(sum(tab2tab.values(), [])))
        model_to_all_table_paths = {mid: load_modelid_to_csvlist(mid, resources=["hugging"]) for mid in reranked}
        return q2c, card2tab, tab2tab, tab2card, reranked, query, seed_models, query_tables, candidate_pool, retrieved_unique, model_to_all_table_paths

    def tab2tab_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for _qt, rts in self.tab2tab.items():
            for rt in rts:
                rt_s = str(rt)
                rows.append({"table": os.path.basename(rt_s), "table_path": rt_s, "models": [str(x).strip() for x in self.tab2card[rt_s]]})
        return rows

    def build_preview(self, *, search_type: str = "", max_models: Optional[int] = None, tables_source: str = "intermediate") -> Dict[str, Any]:
        model_ids = list(self.reranked)
        if max_models is not None:
            model_ids = model_ids[:max(0, int(max_models))]
        tab2tab_trace_rows = self.tab2tab_rows()
        after_model_cap_trace_rows = []
        allowed_models = set(model_ids)
        for row in tab2tab_trace_rows:
            models = [mid for mid in row["models"] if mid in allowed_models]
            if models:
                after_model_cap_trace_rows.append({"table": row["table"], "table_path": row["table_path"], "models": models})
        if tables_source == "all_from_modelcards":
            model_to_table_paths = {mid: [str(p) for p in self.model_to_all_table_paths[mid] if str(p).strip()] for mid in model_ids}
            model_to_table_paths = {mid: paths for mid, paths in model_to_table_paths.items() if paths}
            models_with_tables = list(model_to_table_paths.keys())
            table_paths = list(dict.fromkeys(x for v in model_to_table_paths.values() for x in v))
        else:
            model_to_table_paths = {}
            for row in after_model_cap_trace_rows:
                for mid in row["models"]:
                    model_to_table_paths.setdefault(mid, []).append(row["table_path"])
            model_to_table_paths = {mid: list(dict.fromkeys(paths)) for mid, paths in model_to_table_paths.items() if paths}
            models_with_tables = list(model_to_table_paths.keys())
            table_paths = list(dict.fromkeys(row["table_path"] for row in after_model_cap_trace_rows))
        return {
            "preview_format_version": 1,
            "search_type": search_type,
            "tables_source": tables_source,
            "query_tables": list(self.query_tables),
            "model_ids": list(model_ids),
            "models_with_tables": models_with_tables,
            "model_to_table_paths": model_to_table_paths,
            "table_paths": table_paths,
            "pipeline_trace": {
                "query2modelcard": {"model_ids": list(self.seed_models)},
                "query_dense_rerank": {
                    "applied": True,
                    "model_ids_top_k": list(model_ids),
                    "model_ids_before_dense_rerank": list(self.candidate_pool),
                    "model_ids_after_dense_rerank": list(model_ids),
                },
            },
            "tab2tab_trace_rows": tab2tab_trace_rows,
            "after_model_cap_trace_rows": after_model_cap_trace_rows,
            "retrieved_table_model_rows": after_model_cap_trace_rows,
            "preview_meta": {
                "source": "query2tab2card_full_map",
                "search_type": search_type,
                "tab2tab_trace_rows_source": "tab2tab_map+tab2card_map",
                "tables_source": tables_source,
                "query": self.query,
                "seed_model_ids": list(self.seed_models),
            },
            "job_context": {
                "query": self.query,
                "table_search_seed_model_id": self.seed_models[0],
            },
            "stats": {
                "models_with_tables": len(models_with_tables),
                "total_unique_tables": len(table_paths),
                "tables_source": tables_source,
            },
        }
