# Big Data Migrator - Troubleshooting Guide

**Version**: 2.0.0  
**Updated**: January 25, 2025

This guide helps resolve common issues with the Big Data Migrator system.

---

## 🚨 **Critical Issues**

### **Issue: Multi-LLM Consensus Returns "None"**

**Symptoms:**
- Frontend shows empty consensus response
- API returns `consensus_response: null`
- Individual LLM responses work but no consensus

**Solution:**
```bash
# 1. Check API route configuration
curl http://localhost:8000/llm/conversations/{id}/messages/multi

# 2. Verify consensus logic fix
grep -n "consensus_response" app/api/routes.py

# 3. Restart API server
python start_api.py
```

**Root Cause:** Missing consensus response assignment in API routes  
**Fix Applied:** Updated routes.py to properly set consensus_response field

---

### **Issue: LLM Asking for Data Instead of Analyzing Files**

**Symptoms:**
- Response: "Please share the specific numbers from both datasets"
- LLM not accessing uploaded file data
- Generic responses without actual calculations

**Solution:**
```bash
# 1. Check if data files are loaded
curl http://localhost:8000/debug/conversation-debug/{conversation_id}

# 2. Verify smart processing is enabled
grep "enable_smart_processing=True" app/api/routes.py

# 3. Force data reload by sending a message
# (Safety check will automatically reload data)
```

**Root Cause:** Data files not loaded into conversation system  
**Fix Applied:** Added safety check to reload data when messages are sent

---

### **Issue: 80% Timeout Rate for Multi-file Analysis**

**Symptoms:**
- Requests timeout after 30 seconds
- Multi-file queries consistently fail
- Single file queries work fine

**Solution:**
```bash
# 1. Verify extended timeouts
grep "LOCAL_LLM_TIMEOUT" .env

# 2. Check memory-based extensions
curl http://localhost:8000/debug/system-performance

# 3. Monitor processing in real-time
curl http://localhost:8000/debug/current-processing
```

**Root Cause:** Fixed 30-second timeouts insufficient for complex analysis  
**Fix Applied:** Extended to 300s base, up to 600s with high memory

---

## 🔧 **Startup Issues**

### **Issue: Port Already in Use**

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Or use different port
export API_PORT=8001
python start_api.py
```

---

### **Issue: Module Import Errors**

**Error:**
```
ModuleNotFoundError: No module named 'app'
```

**Solution:**
```bash
# Ensure you're in project directory
cd Big-Data-Migrator

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use absolute imports
python -m app.api.routes
```

---

### **Issue: Environment File Not Found**

**Error:**
```
FileNotFoundError: .env file not found
```

**Solution:**
```bash
# Copy example environment file
cp env_example.txt .env

# Edit with your settings
nano .env

# Verify file exists
ls -la .env
```

---

## 📊 **Performance Issues**

### **Issue: High Memory Usage**

**Symptoms:**
- System becomes slow
- Out of memory errors
- File processing fails

**Diagnosis:**
```bash
# Check memory usage
free -h

# Monitor specific process
top -p $(pgrep -f "start_api.py")

# Check debug endpoint
curl http://localhost:8000/debug/system-performance
```

**Solutions:**
```bash
# 1. Reduce chunk size
echo "CHUNK_SIZE=25000" >> .env

# 2. Lower memory threshold
echo "MEMORY_THRESHOLD_PERCENT=60" >> .env

# 3. Restart services
pkill -f python
python start_api.py
```

---

### **Issue: Slow Response Times**

**Symptoms:**
- API responses take >10 seconds
- Frontend becomes unresponsive
- Timeouts in browser

**Diagnosis:**
```bash
# Check API response time
time curl http://localhost:8000/health

# Monitor system resources
htop

# Check for active processes
curl http://localhost:8000/debug/current-processing
```

**Solutions:**
```bash
# 1. Enable smart processing (if not already)
echo "ENABLE_SMART_PROCESSING=true" >> .env

# 2. Optimize LLM timeouts
echo "LOCAL_LLM_TIMEOUT=120" >> .env

# 3. Use single LLM for simple queries
# (System automatically handles this)
```

---

## 🧠 **LLM Integration Issues**

### **Issue: LM Studio Connection Failed**

**Error:**
```
ConnectionError: Cannot connect to LM Studio
```

**Solution:**
```bash
# 1. Check if LM Studio is running
curl http://localhost:1234/v1/models

# 2. Verify URL in environment
grep LOCAL_LLM_URL .env

# 3. Test connection
python -c "
import requests
try:
    response = requests.get('http://localhost:1234/v1/models')
    print('✅ LM Studio connected')
except:
    print('❌ LM Studio not accessible')
"
```

---

### **Issue: Anthropic Rate Limiting**

**Error:**
```
RateLimitError: Rate limit exceeded
```

**Solution:**
```bash
# 1. Check rate limit status
curl -H "x-api-key: $ANTHROPIC_API_KEY" \
     https://api.anthropic.com/v1/usage

# 2. Reduce Anthropic usage
echo "PRIMARY_LLM=local" >> .env

# 3. Enable intelligent provider selection
# (Already implemented - uses rate limit detection)
```

---

### **Issue: OpenAI API Key Invalid**

**Error:**
```
AuthenticationError: Invalid API key
```

**Solution:**
```bash
# 1. Verify API key format
echo $OPENAI_API_KEY | wc -c  # Should be ~50 characters

# 2. Test key directly
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# 3. Disable if not needed
echo "ENABLE_ONLINE_FALLBACK=false" >> .env
```

---

## 📁 **File Processing Issues**

### **Issue: CSV Headers Not Detected**

**Symptoms:**
- Columns named "Unnamed: 0", "Unnamed: 1"
- Smart processing fails to find amount columns
- Incorrect data analysis

**Solution:**
```bash
# 1. Check CSV structure
head -n 5 your_file.csv

# 2. Test header detection
python -c "
import pandas as pd
from app.processors.smart_query_processor import SmartQueryProcessor
processor = SmartQueryProcessor()
df = pd.read_csv('your_file.csv')
fixed_df = processor._fix_csv_headers(df)
print('Original columns:', list(df.columns))
print('Fixed columns:', list(fixed_df.columns))
"
```

---

### **Issue: File Upload Fails**

**Error:**
```
HTTP 413: Request Entity Too Large
```

**Solution:**
```bash
# 1. Check file size
ls -lh your_file.csv

# 2. Increase size limit
echo "MAX_FILE_SIZE_MB=1000" >> .env

# 3. Check available memory
curl http://localhost:8000/memory-status
```

---

### **Issue: Data Not Loading in Conversation**

**Symptoms:**
- Conversation created successfully
- Debug shows 0 active data files
- LLM can't access file data

**Diagnosis:**
```bash
# Check conversation debug info
curl http://localhost:8000/debug/conversation-debug/{conversation_id}
```

**Solution:**
```bash
# 1. Verify file paths are absolute
python -c "
import os
file_path = 'temp/your_file.csv'
abs_path = os.path.abspath(file_path)
print(f'Absolute path: {abs_path}')
print(f'File exists: {os.path.exists(abs_path)}')
"

# 2. Force data reload by sending message
# (Safety check will trigger automatically)

# 3. Check file permissions
ls -la temp/your_file.csv
```

---

## 🔍 **Debug & Monitoring**

### **Issue: Debug Endpoints Not Working**

**Error:**
```
404 Not Found: /debug/system-performance
```

**Solution:**
```bash
# 1. Verify debug routes are included
grep "debug_router" app/api/routes.py

# 2. Check if endpoint exists
curl http://localhost:8000/docs

# 3. Restart API server
python start_api.py
```

---

### **Issue: Frontend Debug Monitor Blank**

**Symptoms:**
- Debug panel shows no information
- System status not updating
- No error messages shown

**Solution:**
```bash
# 1. Check API connection from frontend
curl http://localhost:8000/debug/system-performance

# 2. Verify frontend debug component
grep -r "debug_monitor" app/frontend/

# 3. Restart frontend
python start_frontend.py
```

---

## 🌐 **Network & Connectivity**

### **Issue: CORS Errors in Browser**

**Error:**
```
Access to fetch at 'http://localhost:8000' blocked by CORS policy
```

**Solution:**
```python
# Already configured in routes.py:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### **Issue: Cannot Access Frontend/API Externally**

**Symptoms:**
- Works on localhost but not external IP
- Connection refused from other machines

**Solution:**
```bash
# 1. Bind to all interfaces
echo "API_HOST=0.0.0.0" >> .env
echo "FRONTEND_HOST=0.0.0.0" >> .env

# 2. Check firewall
sudo ufw status

# 3. Open required ports
sudo ufw allow 8000
sudo ufw allow 8501
```

---

## 🧪 **Testing & Validation**

### **Issue: Tests Failing**

**Error:**
```
AssertionError: Expected smart processing but got False
```

**Solution:**
```bash
# 1. Run specific test
python test_specific_query.py

# 2. Check test environment
python -c "
import os
print('Current directory:', os.getcwd())
print('Files exist:', os.path.exists('temp/'))
"

# 3. Verify system is running
curl http://localhost:8000/health
```

---

## 📋 **Quick Diagnostic Commands**

### **System Health Check**
```bash
#!/bin/bash
echo "🔍 Big Data Migrator System Diagnosis"
echo "======================================"

# API Health
echo "1. API Status:"
curl -s http://localhost:8000/health || echo "❌ API not responding"

# Frontend Health  
echo -e "\n2. Frontend Status:"
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 || echo "❌ Frontend not responding"

# Memory Status
echo -e "\n3. Memory Status:"
curl -s http://localhost:8000/memory-status | grep -o '"memory_usage_percent":[0-9]*' || echo "❌ Cannot get memory status"

# Recent Errors
echo -e "\n4. Recent Errors:"
curl -s http://localhost:8000/debug/recent-errors | head -5 || echo "❌ Cannot get error status"

# Smart Processing Status
echo -e "\n5. Smart Processing:"
echo "✅ Enabled in routes.py" && grep -q "enable_smart_processing=True" app/api/routes.py || echo "❌ Not enabled"
```

### **File Processing Test**
```bash
#!/bin/bash
# Test file processing capability
echo "🧪 Testing File Processing"
echo "========================="

# Check sample files
echo "Sample files:"
ls -la temp/*.csv 2>/dev/null || echo "❌ No sample CSV files found"

# Test pandas import
python -c "
try:
    import pandas as pd
    print('✅ Pandas working')
except ImportError:
    print('❌ Pandas not available')
"

# Test smart processor
python -c "
try:
    from app.processors.smart_query_processor import SmartQueryProcessor
    processor = SmartQueryProcessor()
    print('✅ Smart Query Processor available')
except Exception as e:
    print(f'❌ Smart processor error: {e}')
"
```

---

## 🆘 **Emergency Recovery**

### **Complete System Reset**
```bash
#!/bin/bash
echo "🚨 Emergency System Reset"
echo "========================"

# Stop all processes
pkill -f "python.*start_"

# Clear temporary files
rm -rf __pycache__/
rm -rf app/__pycache__/
rm -rf .pytest_cache/

# Reset environment
cp env_example.txt .env

# Reinstall dependencies
pip install -r requirements.txt

# Start fresh
python start_api.py &
sleep 5
python start_frontend.py &

echo "✅ System reset complete"
```

---

## 📞 **Getting Help**

### **When to Contact Support**
- System completely unresponsive after reset
- Data corruption or loss
- Security-related issues
- Performance degradation >50%

### **Information to Provide**
1. **System Information:**
   ```bash
   python --version
   pip list | grep -E "(fastapi|streamlit|pandas)"
   free -h
   df -h
   ```

2. **Error Logs:**
   ```bash
   tail -50 logs/app.log
   curl http://localhost:8000/debug/recent-errors
   ```

3. **Configuration:**
   ```bash
   cat .env | grep -v "API_KEY"  # Hide sensitive keys
   ```

---

## ✅ **Recovery Checklist**

- [ ] **Identify the specific error message**
- [ ] **Check this troubleshooting guide**
- [ ] **Verify system requirements (RAM, disk space)**
- [ ] **Confirm all services are running**
- [ ] **Test with sample data**
- [ ] **Check network connectivity**
- [ ] **Review recent changes**
- [ ] **Try emergency recovery if needed**
- [ ] **Document the solution for future reference**

---

**💡 Most issues are resolved by restarting services and checking configuration. The system is designed to be resilient and self-recovering.** 