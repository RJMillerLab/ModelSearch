"""
Designed new database for storing results.
Not used yet.
"""

import os, pandas as pd, duckdb
from src.config import QUERY_RESULT_PATH, QUERY_TIME_PATH, QUERY_MODEL_ID_PATH, TABLE_RESULT_PATH, LOG_PATH

def _safe_write(df,path):
    tmp=str(path)+".tmp"; df.to_parquet(tmp,index=False); os.replace(tmp,path)

def _load_or_empty(path,cols):
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=cols)

def _upsert_cell(path,key_col,key_val,col,val,cols):
    df=_load_or_empty(path,cols)
    if key_val not in df[key_col].values:
        row={c:None for c in cols}; row[key_col]=key_val; df.loc[len(df)]=row
    df.loc[df[key_col]==key_val,col]=[val]
    _safe_write(df,path)

def _get_cell(path,key_col,key_val,col):
    if not os.path.exists(path): return None
    con=duckdb.connect()
    r=con.execute(f"SELECT {col} FROM read_parquet('{path}') WHERE {key_col}=?", [key_val]).fetchone()
    con.close(); return r[0] if r else None

# ---------------------------
# DF1 query results
# ---------------------------

RESULT_COLS=["query","dense","sparse","hybrid","topk"]

def save_query_result(query,method,result,query_result_path=QUERY_RESULT_PATH):
    _upsert_cell(query_result_path,"query",query,method,result,RESULT_COLS)

def save_query_topk(query,topk,query_result_path=QUERY_RESULT_PATH):
    _upsert_cell(query_result_path,"query",query,"topk",topk,RESULT_COLS)

def get_query_result(query,method,query_result_path=QUERY_RESULT_PATH):
    return _get_cell(query_result_path,"query",query,method)

def get_query_topk(query,query_result_path=QUERY_RESULT_PATH):
    return _get_cell(query_result_path,"query",query,"topk")

# ---------------------------
# DF2 time
# ---------------------------

def save_query_time(timestamp,query,type_,query_time_path=QUERY_TIME_PATH):
    cols=["timestamp","query","type"]
    df=_load_or_empty(QUERY_TIME_PATH,cols)
    df.loc[len(df)]=[timestamp,query,type_]
    _safe_write(df,query_time_path)

def get_latest_query_time(query,type_,query_time_path=QUERY_TIME_PATH):
    if not os.path.exists(query_time_path): return None
    con=duckdb.connect()
    r=con.execute("""
        SELECT timestamp FROM read_parquet(?)
        WHERE query=? AND type=?
        ORDER BY timestamp DESC LIMIT 1
    """,[query_time_path,query,type_]).fetchone()
    con.close(); return r[0] if r else None

# ---------------------------
# DF3 model id mapping
# ---------------------------

MODEL_COLS=["model_id","dense","sparse","hybrid","topk"]

def save_model_query_id(model_id,method,qid,query_model_id_path=QUERY_MODEL_ID_PATH):
    _upsert_cell(query_model_id_path,"model_id",model_id,method,qid,MODEL_COLS)

def save_model_topk(model_id,topk,query_model_id_path=QUERY_MODEL_ID_PATH):
    _upsert_cell(query_model_id_path,"model_id",model_id,"topk",topk,MODEL_COLS)

def get_model_query_id(model_id,method,query_model_id_path=QUERY_MODEL_ID_PATH):
    return _get_cell(query_model_id_path,"model_id",model_id,method)

# ---------------------------
# DF4 table results
# ---------------------------

TABLE_COLS=["csv_path","keyword","single_col","multi_col","unionable"]

def save_table_result(csv_path,method,result,table_result_path=TABLE_RESULT_PATH):
    _upsert_cell(table_result_path,"csv_path",csv_path,method,result,TABLE_COLS)

def get_table_result(csv_path,method,table_result_path=TABLE_RESULT_PATH):
    return _get_cell(table_result_path,"csv_path",csv_path,method)

# ---------------------------
# DF5 logs
# ---------------------------

LOG_COLS=["query","log_path"]

def save_log(query,log_path,log_path=LOG_PATH):
    df=_load_or_empty(log_path,LOG_COLS)
    df.loc[len(df)]=[query,log_path]
    _safe_write(df,log_path)

def get_logs(query,log_path=LOG_PATH):
    if not os.path.exists(log_path): return []
    con=duckdb.connect()
    r=con.execute("""
        SELECT log_path FROM read_parquet(?)
        WHERE query=?
    """,[log_path,query]).fetchall()
    con.close()
    return [x[0] for x in r]

if __name__=="__main__":
    print("===== ResultStore Test Start =====")
    import time
    q="test_query"; model_id="test_model"; csv_path="test_data/test_table.csv"; log_file="test_data/test_run.log"
    print("Testing query result...")
    # save all under test_data/*.parquet
    save_query_result(q,"dense",["a","b","c"],query_result_path="test_data/query_result1.parquet"); save_query_result(q,"sparse",["d","e"],query_result_path="test_data/query_result2.parquet"); save_query_result(q,"hybrid",["f"],query_result_path="test_data/query_result3.parquet")
    print("dense:",get_query_result(q,"dense")); print("sparse:",get_query_result(q,"sparse"))
    print("Testing query topk...")
    save_query_topk(q,50,query_result_path="test_data/query_result1.parquet"); print("topk:",get_query_topk(q))
    print("Testing timestamp...")
    ts=int(time.time()); save_query_time(ts,q,"modelid",query_time_path="test_data/query_time1.parquet"); print("latest timestamp:",get_latest_query_time(q,"modelid"))
    print("Testing model query id...")
    save_model_query_id(model_id,"dense","dense_qid_1",query_model_id_path="test_data/query_model_id1.parquet"); save_model_topk(model_id,100,query_model_id_path="test_data/query_model_id2.parquet")
    print("model dense qid:",get_model_query_id(model_id,"dense"))
    print("Testing table result...")
    save_table_result(csv_path,"keyword",["table1","table2"],table_result_path="test_data/table_result1.parquet"); print("table result:",get_table_result(csv_path,"keyword"))
    print("Testing logs...")
    save_log(q,log_file,log_path="test_data/log1.parquet"); print("logs:",get_logs(q))
    print("===== ResultStore Test Done =====")