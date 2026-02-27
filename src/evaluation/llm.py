"""
LLM integration for evaluation
"""
import os
import json
import re
from typing import Dict, Any, Optional
import pandas as pd


def serialize_table_for_prompt(df: pd.DataFrame, max_rows: int = 50, max_cols: int = 10) -> str:
    """
    Serialize DataFrame to a string format suitable for LLM prompt.
    
    Args:
        df: DataFrame to serialize
        max_rows: Maximum number of rows to include
        max_cols: Maximum number of columns to include
    
    Returns:
        Serialized table as string
    """
    if df is None or df.empty:
        return "Empty table"
    
    # Limit columns
    cols_to_show = list(df.columns[:max_cols])
    df_subset = df[cols_to_show].head(max_rows)
    
    lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"]
    lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    lines.append("")
    lines.append("Sample Data (first few rows):")
    lines.append(df_subset.to_csv(index=False))
    
    if df.shape[0] > max_rows:
        lines.append(f"\n... and {df.shape[0] - max_rows} more rows")
    
    return "\n".join(lines)


def call_llm_api(prompt: str, model: str = "gpt-4", api_key: Optional[str] = None) -> str:
    """
    Call LLM API to get evaluation response using the internal LLM helper module.
    
    Args:
        prompt: The prompt to send
        model: Model name (default: gpt-4, but may map to a backend-compatible name)
        api_key: API key (if None, tries to get from environment)
    
    Returns:
        LLM response text
    """
    # Prefer the internal llm_citationlake helper module when available
    print(f"🔍 Step 1: Importing internal LLM helper module (llm_citationlake)...")
    try:
        from .llm_citationlake import LLM_response, setup_openai
        print(f"✅ Internal LLM helper module imported successfully")
    except ImportError as import_err:
        print(f"⚠️  Internal LLM helper module import failed: {import_err}")
        import traceback
        print(traceback.format_exc())
        # Fallback to direct OpenAI
        try:
            import openai
            # os is already imported at the top of the file
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                print(f"📡 Using direct OpenAI API...")
                client = openai.OpenAI(api_key=api_key)
                # Note: response_format is only supported by certain models
                # Remove it for compatibility with older models
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert data analyst. Provide evaluations in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                    # Removed response_format for compatibility
                )
                return response.choices[0].message.content
            else:
                raise ValueError("OPENAI_API_KEY not found in environment")
        except Exception as e:
            print(f"⚠️  Direct OpenAI API also failed: {e}")
            raise ValueError(f"LLM API not available: {str(e)}")
    
    # The wrapped LLM_response expects specific model names
    # Map user-facing model names to backend-compatible ones
    print(f"🔍 Step 2: Mapping model name...")
    if model.startswith("gpt-4") or model == "gpt-4":
        llm_model = "gpt-4-turbo"  # wrapper-compatible name
    elif model.startswith("gpt-3.5") or model == "gpt-3.5-turbo":
        llm_model = "gpt-3.5-turbo-0125"
    else:
        # Default to gpt-3.5-turbo-0125 which is most compatible
        llm_model = "gpt-3.5-turbo-0125"
    print(f"   Mapped {model} -> {llm_model}")
    
    # Add system message to prompt
    system_message = "You are an expert data analyst. Provide evaluations in valid JSON format."
    full_prompt = f"{system_message}\n\n{prompt}"
    
    print(f"🔍 Step 3: Setting up OpenAI...")
    print(f"   Checking environment variables...")
    print(f"   OPENAI_API_KEY in os.environ: {'OPENAI_API_KEY' in os.environ}")
    print(f"   os.getenv('OPENAI_API_KEY'): {os.getenv('OPENAI_API_KEY') is not None}")
    try:
        setup_openai('', mode='openai')
        print(f"✅ OpenAI setup successful")
    except Exception as setup_err:
        print(f"⚠️  OpenAI setup failed: {setup_err}")
        import traceback
        print(traceback.format_exc())
        raise ValueError(f"OpenAI setup failed: {str(setup_err)}")
    
    print(f"🔍 Step 4: Calling LLM_response with model: {llm_model}")
    try:
        # The wrapped query_openai accepts kwargs and passes them to openai API
        # Note: Don't pass response_format as it's not supported by all models
        response, _ = LLM_response(
            chat_prompt=full_prompt,
            llm_model=llm_model,
            history=[],
            kwargs={"temperature": 0.3},  # Only pass temperature, not response_format
            max_tokens=2000
        )
        print(f"✅ LLM API call successful, response length: {len(response)}")
        return response
    except Exception as call_err:
        print(f"⚠️  LLM_response call failed: {call_err}")
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Full traceback:\n{error_traceback}")
        raise ValueError(f"LLM API call failed: {str(call_err)}")


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response and extract evaluation scores.
    
    Args:
        response_text: Raw response from LLM
    
    Returns:
        Parsed evaluation dictionary
    """
    try:
        # Try to parse as JSON
        if response_text.strip().startswith('{'):
            return json.loads(response_text)
        else:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # Try to find JSON object in the text
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
    except Exception as e:
        print(f"⚠️  Error parsing LLM response: {e}")
        print(f"   Response text: {response_text[:200]}...")
    
    # Fallback: return error response
    return {
        "error": "Failed to parse LLM response",
        "raw_response": response_text[:500]
    }


def evaluate_diversity_with_llm(
    query: str,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table1_source: str = "Table Search Integration",
    table2_source: str = "Model Search Integration",
) -> Dict[str, Any]:
    """
    Evaluate diversity between two tables using LLM.
    
    Args:
        query: Original search query
        table1: First integrated table (DataFrame)
        table2: Second integrated table (DataFrame)
        table1_source: Description of table1 source
        table2_source: Description of table2 source
    
    Returns:
        Dictionary with evaluation results including diversity_score and analysis
    """
    # Serialize tables
    print("📊 Serializing tables for LLM prompt...")
    table1_str = serialize_table_for_prompt(table1)
    table2_str = serialize_table_for_prompt(table2)
    
    # Get prompt
    from .prompt import get_diversity_evaluation_prompt
    prompt = get_diversity_evaluation_prompt(
        query=query,
        table1_serialized=table1_str,
        table2_serialized=table2_str,
        table1_source=table1_source,
        table2_source=table2_source
    )
    
    # Call LLM
    print("📡 Attempting to call LLM API...")
    try:
        response_text = call_llm_api(prompt)
        print(f"✅ Got LLM response, length: {len(response_text)}")
        result = parse_llm_response(response_text)
        print(f"✅ Parsed LLM response, keys: {list(result.keys())}")
        result["success"] = True
        return result
    except ValueError as ve:
        # This is the "LLM API not available" error from call_llm_api
        print(f"⚠️  LLM API ValueError: {ve}")
        raise ValueError(f"LLM API not available: {str(ve)}. Please set OPENAI_API_KEY.")
    except Exception as e:
        # If LLM API fails for any other reason
        print(f"⚠️  LLM API error: {e}")
        import traceback
        print(traceback.format_exc())
        raise Exception(f"LLM API error: {str(e)}. Please check your API configuration.")


def load_fake_response(fake_response_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load fake response from JSON file for testing.
    
    Args:
        fake_response_path: Path to fake response JSON file (default: data/evaluation_fake_response.json)
    
    Returns:
        Fake evaluation response dictionary
    """
    if fake_response_path is None:
        fake_response_path = "data/evaluation_fake_response.json"
    
    if not os.path.exists(fake_response_path):
        # Default fake response: total + sub_scores (relevance, coverage, diversity), strengths/weaknesses, key_differences, evidence
        return {
            "success": True,
            "total_quality_score": {
                "table_search": 85,
                "model_search": 72,
                "winner": "table_search"
            },
            "sub_scores": [
                {"name": "Relevance", "table_search": 88, "model_search": 75, "evidence": "Table Search: 12/15 rows match query terms; Model Search: 5/10."},
                {"name": "Coverage", "table_search": 82, "model_search": 70, "evidence": "Table Search covers 8 distinct model families vs 4 in Model Search."},
                {"name": "Diversity", "table_search": 85, "model_search": 72, "evidence": "Table Search includes 6 table types (benchmark, metadata, etc.); Model Search 2."}
            ],
            "quality_analysis": {
                "table_search": {
                    "strengths": ["More structured data", "Better coverage of technical specs"],
                    "weaknesses": ["Limited to models with table data"]
                },
                "model_search": {
                    "strengths": ["Broader model coverage", "Includes popular models"],
                    "weaknesses": ["Less structured format", "More missing values"]
                }
            },
            "key_differences": [
                "Table Search has higher relevance and diversity for the user question.",
                "Table Search provides more varied table types and structured columns."
            ],
            "evidence_for_differences": "Table Search integrated table has more non-null columns (8 vs 5) and rows mentioning query-related terms (12 vs 5). Diversity: 6 distinct table types in Table Search vs 2 in Model Search.",
            "source": "fake_response"
        }
    
    try:
        with open(fake_response_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data["success"] = True
        data["source"] = "fake_response_file"
        return data
    except Exception as e:
        print(f"⚠️  Error loading fake response: {e}")
        return load_fake_response(None)  # Fallback to default


if __name__ == "__main__":
    # Quick test: serialize + eval (requires valid OPENAI_API_KEY)
    table1 = pd.DataFrame({"model_id": ["codet5-base", "CodeGPT-small-py"], "steps": [500000, 300000]})
    table2 = pd.DataFrame({"model_id": ["gpt2", "bert-base"], "type": ["gen", "encoder"]})
    s1 = serialize_table_for_prompt(table1)
    s2 = serialize_table_for_prompt(table2)
    print("Test evaluation: serialize length", len(s1), len(s2))
    result = evaluate_diversity_with_llm(query="code gen", table1=table1, table2=table2)
    print("Eval result keys:", list(result.keys()), "diversity_score:", result.get("diversity_score"))
