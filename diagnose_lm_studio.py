#!/usr/bin/env python3
"""
LM Studio Connection Diagnostic Script

This script helps diagnose connection issues with LM Studio
and provides troubleshooting steps.
"""
import requests
import time
import json

def test_lm_studio_connection(url="http://127.0.0.1:1234"):
    """Test LM Studio connection with detailed diagnostics."""
    print("🔍 LM Studio Connection Diagnostics")
    print("=" * 40)
    
    # Test 1: Basic connectivity
    print("\n1. Testing basic connectivity...")
    try:
        response = requests.get(f"{url}/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ LM Studio is running and accessible")
            models_data = response.json()
            
            if models_data.get("data"):
                print(f"📋 Available models: {len(models_data['data'])}")
                for model in models_data["data"]:
                    print(f"   - {model.get('id', 'Unknown')}")
            else:
                print("⚠️  LM Studio is running but no models are loaded")
                return False
        else:
            print(f"❌ LM Studio responded with status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectRefused:
        print("❌ Connection refused - LM Studio is not running")
        print("\n💡 Solutions:")
        print("   1. Start LM Studio application")
        print("   2. Load a model in LM Studio")
        print("   3. Ensure the local server is started in LM Studio")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ Connection timeout - LM Studio may be starting up or overloaded")
        print("\n💡 Solutions:")
        print("   1. Wait for LM Studio to fully load the model")
        print("   2. Try again in a few minutes")
        print("   3. Check if your computer has enough RAM for the model")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False
    
    # Test 2: Simple chat completion
    print("\n2. Testing chat completion...")
    try:
        test_payload = {
            "model": "local-model",  # LM Studio typically uses this
            "messages": [
                {"role": "user", "content": "Hello, can you respond with just 'OK'?"}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        start_time = time.time()
        response = requests.post(
            f"{url}/v1/chat/completions", 
            json=test_payload,
            timeout=30  # 30 second timeout for model response
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Chat completion successful ({response_time:.1f}s)")
            data = response.json()
            if data.get("choices"):
                content = data["choices"][0]["message"]["content"]
                print(f"   Model response: '{content.strip()}'")
            return True
        else:
            print(f"❌ Chat completion failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Chat completion timeout (>30 seconds)")
        print("\n💡 This suggests:")
        print("   1. Model is too large for your hardware")
        print("   2. Model is still loading")
        print("   3. System is under heavy load")
        return False
        
    except Exception as e:
        print(f"❌ Chat completion error: {str(e)}")
        return False

def check_system_resources():
    """Check system resources that might affect LM Studio."""
    print("\n3. Checking system resources...")
    
    try:
        import psutil
        
        # Check RAM
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        used_percent = memory.percent
        
        print(f"💾 Memory Status:")
        print(f"   Total RAM: {total_gb:.1f} GB")
        print(f"   Available: {available_gb:.1f} GB ({100-used_percent:.1f}% free)")
        print(f"   Used: {used_percent:.1f}%")
        
        if available_gb < 4:
            print("⚠️  Warning: Low available memory (< 4GB)")
            print("   Large models may not run properly")
        
        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"🖥️  CPU Usage: {cpu_percent:.1f}%")
        
        if cpu_percent > 90:
            print("⚠️  Warning: High CPU usage")
            print("   This may slow down model responses")
        
    except ImportError:
        print("⚠️  Cannot check system resources (psutil not installed)")

def test_model_specific(model_names):
    """Test specific models if provided."""
    url = "http://127.0.0.1:1234"
    
    print(f"\n4. Testing specific models...")
    
    for model_name in model_names:
        print(f"\n🤖 Testing model: {model_name}")
        
        try:
            test_payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "What is 2+2? Answer with just the number."}
                ],
                "max_tokens": 5,
                "temperature": 0.1
            }
            
            start_time = time.time()
            response = requests.post(
                f"{url}/v1/chat/completions",
                json=test_payload,
                timeout=20
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                print(f"   ✅ Success ({response_time:.1f}s)")
                data = response.json()
                if data.get("choices"):
                    content = data["choices"][0]["message"]["content"]
                    print(f"   Response: '{content.strip()}'")
            else:
                print(f"   ❌ Failed (Status: {response.status_code})")
                
        except requests.exceptions.Timeout:
            print(f"   ❌ Timeout (>20s) - Model may be too slow")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def provide_solutions():
    """Provide troubleshooting solutions."""
    print("\n" + "="*50)
    print("🛠️  TROUBLESHOOTING SOLUTIONS")
    print("="*50)
    
    print("\n🚫 If LM Studio is not running:")
    print("   1. Open LM Studio application")
    print("   2. Go to 'Local Server' tab")
    print("   3. Load one of your models:")
    print("      - claude-3.7-sonnet-reasoning-gemma3-12B")
    print("      - codellama-34b-instruct")
    print("   4. Click 'Start Server'")
    
    print("\n⏱️  If getting timeouts:")
    print("   1. Wait for model to fully load (can take 5-10 minutes)")
    print("   2. Try a smaller model first")
    print("   3. Close other applications to free up RAM")
    print("   4. Use a lower quantization model (Q4 instead of Q8)")
    
    print("\n🔧 Model-specific tips:")
    print("   📊 For your models:")
    print("   - claude-3.7-sonnet-reasoning-gemma3-12B (12B params)")
    print("     Recommended: 16GB+ RAM, Q4_K_M quantization")
    print("   - codellama-34b-instruct (34B params)")
    print("     Recommended: 32GB+ RAM, Q3_K_M quantization")
    
    print("\n🎯 Testing sequence:")
    print("   1. Start with the smaller model (claude-3.7-sonnet)")
    print("   2. Test basic connectivity first")
    print("   3. Then run the full model comparison")

def main():
    """Main diagnostic function."""
    # Test basic connection
    if test_lm_studio_connection():
        print("\n✅ Basic LM Studio connection successful!")
        
        # Check system resources
        check_system_resources()
        
        # Test specific models
        models_to_test = [
            "claude-3.7-sonnet-reasoning-gemma3-12B",
            "codellama-34b-instruct",
            "local-model"  # Generic LM Studio model name
        ]
        test_model_specific(models_to_test)
        
        print("\n🎉 Diagnostics complete! You can now run:")
        print("   python test_llm_models.py")
        
    else:
        print("\n❌ LM Studio connection failed!")
        provide_solutions()

if __name__ == "__main__":
    main() 