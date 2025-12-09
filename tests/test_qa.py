"""
Test script for QA module
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.qa.llm import answer_question_with_llm, serialize_table_for_prompt
from src.qa.prompt import get_qa_prompt


def test_qa_with_fake():
    """Test QA with fake response"""
    print("=" * 60)
    print("Testing QA with fake response")
    print("=" * 60)
    
    # Create a sample table
    sample_data = {
        "model_name": ["GPT-4", "BERT", "ResNet", "YOLO"],
        "type": ["LLM", "NLP", "CV", "CV"],
        "parameters": ["1.7T", "110M", "25M", "65M"],
        "accuracy": [0.95, 0.92, 0.88, 0.85]
    }
    df = pd.DataFrame(sample_data)
    
    query = "What are the different types of models in this dataset?"
    
    print(f"\nQuery: {query}")
    print(f"\nTable shape: {df.shape}")
    print(f"Table columns: {df.columns.tolist()}")
    print(f"\nTable preview:")
    print(df.head())
    
    # Test with fake response
    result = answer_question_with_llm(
        query=query,
        table=df,
        table_source="Test Table",
        use_fake=True
    )
    
    print(f"\n✅ QA Result:")
    print(f"   Success: {result['success']}")
    print(f"   Source: {result['source']}")
    print(f"\n   Answer:")
    print(f"   {result['answer']['answer']}")
    print(f"\n   Key Findings:")
    for finding in result['answer'].get('key_findings', []):
        print(f"   - {finding}")
    
    return result


def test_qa_prompt():
    """Test QA prompt generation"""
    print("\n" + "=" * 60)
    print("Testing QA prompt generation")
    print("=" * 60)
    
    query = "What models are best for natural language processing?"
    table_serialized = "Shape: 10 rows × 5 columns\nColumns: model_name, type, parameters, accuracy, task\n\nSample data..."
    
    prompt = get_qa_prompt(query, table_serialized, "Integrated Table")
    
    print(f"\nQuery: {query}")
    print(f"\nGenerated prompt (first 500 chars):")
    print(prompt[:500] + "...")
    
    return prompt


def test_serialize_table():
    """Test table serialization"""
    print("\n" + "=" * 60)
    print("Testing table serialization")
    print("=" * 60)
    
    sample_data = {
        "model_name": ["GPT-4", "BERT", "ResNet", "YOLO", "Transformer"],
        "type": ["LLM", "NLP", "CV", "CV", "NLP"],
        "parameters": ["1.7T", "110M", "25M", "65M", "170M"],
        "accuracy": [0.95, 0.92, 0.88, 0.85, 0.90]
    }
    df = pd.DataFrame(sample_data)
    
    serialized = serialize_table_for_prompt(df, max_rows=3, max_cols=3)
    
    print(f"\nOriginal table shape: {df.shape}")
    print(f"\nSerialized table:")
    print(serialized)
    
    return serialized


def test_qa_with_real_llm():
    """Test QA with real LLM (requires OPENAI_API_KEY)"""
    print("\n" + "=" * 60)
    print("Testing QA with real LLM")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found. Skipping real LLM test.")
        return None
    
    # Create a sample table
    sample_data = {
        "model_name": ["GPT-4", "BERT", "ResNet"],
        "type": ["LLM", "NLP", "CV"],
        "parameters": ["1.7T", "110M", "25M"]
    }
    df = pd.DataFrame(sample_data)
    
    query = "What are the different types of models?"
    
    print(f"\nQuery: {query}")
    print(f"Table shape: {df.shape}")
    
    try:
        result = answer_question_with_llm(
            query=query,
            table=df,
            table_source="Test Table",
            use_fake=False,
            model="gpt-3.5-turbo-0125"  # Use cheaper model for testing
        )
        
        print(f"\n✅ QA Result:")
        print(f"   Success: {result['success']}")
        print(f"   Source: {result['source']}")
        print(f"\n   Answer:")
        print(f"   {result['answer']['answer']}")
        
        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


if __name__ == "__main__":
    print("🧪 QA Module Test Suite")
    print("=" * 60)
    
    # Test 1: Fake response
    test_qa_with_fake()
    
    # Test 2: Prompt generation
    test_qa_prompt()
    
    # Test 3: Table serialization
    test_serialize_table()
    
    # Test 4: Real LLM (optional, requires API key)
    # Uncomment to test with real LLM
    # test_qa_with_real_llm()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

