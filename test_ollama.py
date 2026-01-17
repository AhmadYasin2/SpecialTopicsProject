"""
Test script to verify Ollama configuration
Run with: python test_ollama.py
"""
import sys
import requests
from config import settings

print("\n" + "="*60)
print("AutoPRISMA - Ollama Configuration Test")
print("="*60 + "\n")

# Test 1: Check configuration
print("[1/4] Checking configuration...")
print(f"      Provider: {settings.llm_provider}")
print(f"      Model: {settings.llm_model}")
print(f"      Base URL: {settings.ollama_base_url}")
print(f"      ✓ Configuration loaded\n")

# Test 2: Check Ollama server
print("[2/4] Checking Ollama server...")
try:
    response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
    if response.status_code == 200:
        print(f"      ✓ Ollama server is running\n")
    else:
        print(f"      ✗ Ollama server returned status {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"      ✗ Cannot connect to Ollama server: {e}")
    print(f"      Please start Ollama with: ollama serve")
    sys.exit(1)

# Test 3: Check if model is available
print("[3/4] Checking if model is available...")
try:
    response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
    data = response.json()
    models = [model.get('name', '') for model in data.get('models', [])]
    
    if settings.llm_model in models or any(settings.llm_model in m for m in models):
        print(f"      ✓ Model '{settings.llm_model}' is available\n")
    else:
        print(f"      ✗ Model '{settings.llm_model}' not found")
        print(f"      Available models: {', '.join(models)}")
        print(f"      Please pull the model with: ollama pull {settings.llm_model}")
        sys.exit(1)
except Exception as e:
    print(f"      ✗ Error checking models: {e}")
    sys.exit(1)

# Test 4: Test LLM connection
print("[4/4] Testing LLM connection...")
try:
    from langchain_ollama import ChatOllama
    
    llm = ChatOllama(
        model=settings.llm_model,
        temperature=0.1,
        base_url=settings.ollama_base_url
    )
    
    # Simple test prompt
    response = llm.invoke("Say 'Hello' in one word.")
    print(f"      ✓ LLM connection successful")
    print(f"      Response: {response.content}\n")
    
except ImportError:
    print(f"      ✗ langchain-ollama not installed")
    print(f"      Please install with: pip install langchain-ollama")
    sys.exit(1)
except Exception as e:
    print(f"      ✗ Error testing LLM: {e}")
    print(f"      This may be normal if the model is still loading")
    sys.exit(1)

# Success
print("="*60)
print("✓ All tests passed! System is ready to use.")
print("="*60)
print("\nYou can now run:")
print('  python cli.py "Your research question"')
print("\n")
