"""
LLM integration for QA (Question Answering) based on integrated tables
"""
import os
import json
import re
from typing import Dict, Any, Optional, List
import pandas as pd
from .prompt import get_qa_prompt


def serialize_table_for_prompt(df: pd.DataFrame, max_rows: int = 100, max_cols: int = 20) -> str:
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
    lines.append("Data:")
    lines.append(df_subset.to_csv(index=False))
    
    if df.shape[0] > max_rows:
        lines.append(f"\n... and {df.shape[0] - max_rows} more rows")
    
    if df.shape[1] > max_cols:
        lines.append(f"\n... and {df.shape[1] - max_cols} more columns")
    
    return "\n".join(lines)


def call_llm_api(prompt: str, model: str = "gpt-4", api_key: Optional[str] = None) -> str:
    """
    Call LLM API to get QA response using CitationLake's LLM module.
    
    Args:
        prompt: The prompt to send
        model: Model name (default: gpt-4)
        api_key: API key (if None, tries to get from environment)
    
    Returns:
        LLM response text
    """
    # Use CitationLake's LLM_response function (absolute import to avoid "beyond top-level package" when qa is top-level)
    print(f"🔍 Importing CitationLake LLM module...")
    try:
        from evaluation.llm_citationlake import LLM_response, setup_openai
        print(f"✅ CitationLake LLM module imported successfully")
    except ImportError as import_err:
        print(f"⚠️  CitationLake LLM module import failed: {import_err}")
        import traceback
        print(traceback.format_exc())
        # Fallback to direct OpenAI
        try:
            import openai
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                print(f"📡 Using direct OpenAI API...")
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert data analyst. Provide answers in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content
            else:
                raise ValueError("OPENAI_API_KEY not found in environment")
        except Exception as e:
            print(f"⚠️  Direct OpenAI API also failed: {e}")
            raise ValueError(f"LLM API not available: {str(e)}")
    
    # CitationLake's LLM_response expects specific model names
    # Map model names to compatible ones
    if model.startswith("gpt-4") or model == "gpt-4":
        llm_model = "gpt-4-turbo"  # CitationLake compatible name
    elif model.startswith("gpt-3.5") or model == "gpt-3.5-turbo":
        llm_model = "gpt-3.5-turbo-0125"
    else:
        # Default to gpt-3.5-turbo-0125 which is most compatible
        llm_model = "gpt-3.5-turbo-0125"
    print(f"   Mapped {model} -> {llm_model}")
    
    # Add system message to prompt
    system_message = "You are an expert data analyst. Provide answers in valid JSON format."
    full_prompt = f"{system_message}\n\n{prompt}"
    
    print(f"🔍 Setting up OpenAI...")
    try:
        setup_openai('', mode='openai')
        print(f"✅ OpenAI setup successful")
    except Exception as setup_err:
        print(f"⚠️  OpenAI setup failed: {setup_err}")
        import traceback
        print(traceback.format_exc())
        raise ValueError(f"OpenAI setup failed: {str(setup_err)}")
    
    print(f"🔍 Calling LLM_response with model: {llm_model}")
    try:
        # CitationLake's query_openai accepts kwargs and passes them to openai API
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
    Parse LLM response text to extract JSON answer.
    
    Args:
        response_text: Raw response text from LLM
    
    Returns:
        Parsed answer dictionary
    """
    # Try to extract JSON from the response
    # Sometimes LLM wraps JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # If no JSON found, return the raw text as answer
            return {
                "answer": response_text,
                "model_ranking": [],
                "summary": {},
                "confidence": "medium",
                "limitations": ["Could not parse structured response from LLM"]
            }
    
    try:
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse JSON: {e}")
        print(f"   Response text: {response_text[:500]}...")
        # Return raw text as answer
        return {
            "answer": response_text,
            "model_ranking": [],
            "summary": {},
            "confidence": "medium",
            "limitations": [f"JSON parsing error: {str(e)}"]
        }


def load_fake_response(fake_response_path: Optional[str] = None, fake_response_content: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load fake QA response for testing.
    
    Args:
        fake_response_path: Path to fake response JSON file
        fake_response_content: Direct fake response content (dict)
    
    Returns:
        Fake answer dictionary
    """
    if fake_response_content:
        return fake_response_content
    
    if fake_response_path and os.path.exists(fake_response_path):
        with open(fake_response_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Default fake response with model ranking
    return {
        "answer": "Based on your query requirements, I have analyzed all models in the integrated table and ranked them by suitability. The top recommendations are models that best match your specific needs, considering factors such as task compatibility, performance metrics, and use case fit.",
        "model_ranking": [
            {
                "rank": 1,
                "model_id": "openai/gpt-4",
                "model_name": "GPT-4",
                "suitability_score": 95,
                "reasons": [
                    "Excellent performance on the specified task",
                    "Strong capabilities matching query requirements",
                    "Proven track record in similar use cases"
                ],
                "strengths": [
                    "High accuracy and performance",
                    "Versatile and adaptable",
                    "Well-documented and supported"
                ],
                "limitations": [
                    "May require significant computational resources"
                ],
                "use_case": "Best suited for production applications requiring high accuracy"
            },
            {
                "rank": 2,
                "model_id": "bert-base-uncased",
                "model_name": "BERT",
                "suitability_score": 85,
                "analysis": "BERT provides good performance on related tasks with efficient resource usage. Suitable for the specified domain based on model card and table data.",
                "supporting_evidence": [
                    {
                        "claim": "Efficient model size",
                        "source": "model_card",
                        "evidence": "Parameters: 110M",
                        "relevance": "Good balance between performance and efficiency"
                    }
                ],
                "reasons": [
                    "Good performance on related tasks",
                    "Efficient and well-optimized",
                    "Suitable for the specified domain"
                ],
                "strengths": [
                    "Efficient inference",
                    "Good balance of performance and resources",
                    "Widely used and tested"
                ],
                "limitations": [
                    "May not match top-tier performance"
                ],
                "key_metrics": {
                    "parameters": "110M"
                },
                "use_case": "Good for applications requiring efficiency and good performance"
            },
            {
                "rank": 3,
                "model_id": "resnet50",
                "model_name": "ResNet",
                "suitability_score": 75,
                "analysis": "ResNet is relevant for the task type with reasonable performance. Suitable as an alternative option.",
                "supporting_evidence": [
                    {
                        "claim": "Computer vision capabilities",
                        "source": "model_card",
                        "evidence": "Model card indicates: 'ResNet-50 is designed for image classification tasks'",
                        "relevance": "Relevant if query involves CV tasks"
                    }
                ],
                "reasons": [
                    "Relevant for the task type",
                    "Reasonable performance",
                    "Suitable as an alternative option"
                ],
                "strengths": [
                    "Proven architecture",
                    "Good for specific use cases"
                ],
                "limitations": [
                    "May not be optimal for the exact requirements"
                ],
                "key_metrics": {
                    "parameters": "25M"
                },
                "use_case": "Alternative option for specific scenarios"
            }
        ],
        "summary": {
            "total_models_analyzed": 3,
            "top_recommendations": ["openai/gpt-4", "bert-base-uncased", "resnet50"],
            "key_criteria_used": [
                "Task compatibility",
                "Performance metrics",
                "Use case fit",
                "Resource requirements"
            ],
            "evidence_sources": {
                "table_cells_used": false,
                "model_cards_used": true,
                "data_quality": "Based on model card information"
            }
        },
        "confidence": "high",
        "limitations": []
    }


def answer_question_with_llm(
    query: str,
    table: pd.DataFrame,
    table_source: str = "Integrated Table",
    qa_mode: str = "card2tab2card",  # "card2card" or "card2tab2card"
    model_ids_to_rank: Optional[List[str]] = None,
    search_results_data: Optional[Dict] = None,  # Full search results for model card access
    use_fake: bool = False,
    fake_response_path: Optional[str] = None,
    fake_response_content: Optional[Dict] = None,
    model: str = "gpt-4"
) -> Dict[str, Any]:
    """
    Answer a question based on an integrated table using LLM.
    
    Args:
        query: User's question or query
        table: Integrated DataFrame
        table_source: Description of where the table came from
        use_fake: Whether to use fake response (for testing)
        fake_response_path: Path to fake response JSON file
        fake_response_content: Direct fake response content (dict)
        model: LLM model to use
    
    Returns:
        Dictionary with answer and metadata
    """
    if use_fake:
        print("📝 Using fake QA response for testing")
        return {
            "success": True,
            "answer": load_fake_response(fake_response_path, fake_response_content),
            "source": "fake"
        }
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set OPENAI_API_KEY in your environment or use fake response mode.")
    
    # Serialize table (may be empty for Card2Card mode)
    table_serialized = serialize_table_for_prompt(table) if not table.empty else "No integrated table available. Analysis will be based on model card information only."
    
    # Extract model card information if available
    model_card_info = {}
    if search_results_data:
        # Try to extract model card information from search results
        # This is a placeholder - actual model card extraction would need to access CitationLake or parquet files
        model_card_info = {}  # Will be populated if model card access is implemented
    
    # Generate prompt with model IDs to rank and QA mode
    prompt = get_qa_prompt(
        query=query,
        table_serialized=table_serialized,
        table_source=table_source,
        qa_mode=qa_mode,
        model_ids_to_rank=model_ids_to_rank,
        model_card_info=model_card_info
    )
    
    # Call LLM
    try:
        response_text = call_llm_api(prompt, model=model, api_key=api_key)
        answer = parse_llm_response(response_text)
        
        return {
            "success": True,
            "answer": answer,
            "source": "llm",
            "model": model
        }
    except ValueError as ve:
        raise ValueError(f"LLM API not available: {str(ve)}. Please set OPENAI_API_KEY or use fake response mode.")
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"❌ Error in answer_question_with_llm: {str(e)}")
        print(f"Traceback:\n{error_traceback}")
        raise ValueError(f"QA failed: {str(e)}")


if __name__ == "__main__":
    # Quick test: QA with fake
    df = pd.DataFrame({"model": ["GPT-4", "BERT"], "type": ["LLM", "NLP"]})
    result = answer_question_with_llm(query="What types?", table=df, table_source="Test", use_fake=True)
    print("Test QA: success", result.get("success"), "source", result.get("source"))

