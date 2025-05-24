# Big Data Migrator - Deployment Guide

**Version**: 2.0.0  
**Updated**: January 25, 2025

This guide covers deploying the Big Data Migrator system for production use.

---

## 🚀 **Quick Production Deployment**

### **Prerequisites**
- Python 3.8+ 
- 4GB+ RAM (8GB+ recommended)
- 50GB+ storage space
- SSL certificate (for HTTPS)
- Domain name (optional)

### **One-Command Setup**
```bash
# Clone and setup
git clone https://github.com/yourusername/Big-Data-Migrator
cd Big-Data-Migrator
chmod +x setup_production.sh
./setup_production.sh
```

---

## 🔧 **Manual Production Setup**

### **1. System Preparation**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.8 python3-pip python3-venv nginx supervisor -y

# Create application user
sudo useradd -m -s /bin/bash bigdata
sudo usermod -aG sudo bigdata
```

### **2. Application Setup**
```bash
# Switch to application user
sudo su - bigdata

# Clone repository
git clone https://github.com/yourusername/Big-Data-Migrator
cd Big-Data-Migrator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create production config
cp env_example.txt .env
```

### **3. Environment Configuration**
```bash
# Edit .env file
nano .env
```

**Production .env Template:**
```env
# Core Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_PORT=8501
WORKERS=4

# Security
SECRET_KEY=your-secure-secret-key-here
ALLOWED_HOSTS=yourdomain.com,localhost

# LLM Configuration
ENABLE_MULTI_LLM=true
PRIMARY_LLM=local
LOCAL_LLM_URL=http://127.0.0.1:1234/v1
LOCAL_LLM_MODEL=claude-3.7-sonnet-reasoning-gemma3-12b
LOCAL_LLM_TIMEOUT=300

# Optional: External LLM Providers
ENABLE_ANTHROPIC=true
ANTHROPIC_API_KEY=your-anthropic-key
ANTHROPIC_TIMEOUT=300

ENABLE_ONLINE_FALLBACK=true
OPENAI_API_KEY=your-openai-key
OPENAI_TIMEOUT=300

# Memory Management
MAX_FILE_SIZE_MB=500
MEMORY_THRESHOLD_PERCENT=70
CHUNK_SIZE=50000

# Database (Optional)
DATABASE_URL=postgresql://user:pass@localhost/bigdata

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

---

## 🐳 **Docker Deployment**

### **Option 1: Docker Compose (Recommended)**
```yaml
# docker-compose.yml
version: '3.8'
services:
  bigdata-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./uploads:/app/uploads
      - ./conversations:/app/conversations
    restart: unless-stopped
    
  bigdata-frontend:
    build: 
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    depends_on:
      - bigdata-api
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - bigdata-api
      - bigdata-frontend
    restart: unless-stopped
```

### **Build and Deploy**
```bash
# Build and start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

---

## ☁️ **Cloud Deployment**

### **AWS Deployment**

#### **EC2 Setup**
```bash
# Launch EC2 instance (t3.large or larger recommended)
# Connect via SSH
ssh -i your-key.pem ubuntu@your-ec2-ip

# Follow manual setup steps above
# Configure security groups for ports 80, 443, 8000, 8501
```

#### **Application Load Balancer**
```json
{
  "listeners": [
    {
      "port": 80,
      "protocol": "HTTP",
      "targets": [
        {"port": 8501, "health_check": "/health"}
      ]
    },
    {
      "port": 8000,
      "protocol": "HTTP", 
      "targets": [
        {"port": 8000, "health_check": "/health"}
      ]
    }
  ]
}
```

### **Azure Deployment**

#### **Container Instances**
```bash
# Create resource group
az group create --name bigdata-rg --location eastus

# Deploy container
az container create \
  --resource-group bigdata-rg \
  --name bigdata-migrator \
  --image youracr.azurecr.io/bigdata:latest \
  --ports 8000 8501 \
  --memory 4 \
  --cpu 2
```

### **Google Cloud Platform**

#### **Cloud Run**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/bigdata
gcloud run deploy --image gcr.io/PROJECT-ID/bigdata --platform managed
```

---

## 🔒 **Security Configuration**

### **SSL/TLS Setup**
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### **Nginx Configuration**
```nginx
# /etc/nginx/sites-available/bigdata
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Frontend
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support for Streamlit
    location /_stcore/stream {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

### **Firewall Setup**
```bash
# UFW configuration
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw deny 8000  # API only via nginx
sudo ufw deny 8501  # Frontend only via nginx
```

---

## 📊 **Process Management**

### **Supervisor Configuration**
```ini
# /etc/supervisor/conf.d/bigdata-api.conf
[program:bigdata-api]
command=/home/bigdata/Big-Data-Migrator/venv/bin/python start_api.py
directory=/home/bigdata/Big-Data-Migrator
user=bigdata
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/supervisor/bigdata-api.log

# /etc/supervisor/conf.d/bigdata-frontend.conf
[program:bigdata-frontend]
command=/home/bigdata/Big-Data-Migrator/venv/bin/python start_frontend.py
directory=/home/bigdata/Big-Data-Migrator
user=bigdata
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/supervisor/bigdata-frontend.log
```

### **Supervisor Commands**
```bash
# Reload configuration
sudo supervisorctl reread
sudo supervisorctl update

# Start services
sudo supervisorctl start bigdata-api
sudo supervisorctl start bigdata-frontend

# Check status
sudo supervisorctl status

# View logs
sudo supervisorctl tail -f bigdata-api
```

---

## 📈 **Monitoring & Logging**

### **System Monitoring**
```bash
# Install monitoring tools
sudo apt install htop iotop nethogs

# Setup log rotation
sudo nano /etc/logrotate.d/bigdata
```

**Log Rotation Configuration:**
```
/var/log/supervisor/bigdata-*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    copytruncate
}
```

### **Application Monitoring**
```python
# Health check endpoint
curl http://localhost:8000/health

# System performance
curl http://localhost:8000/debug/system-performance

# Recent errors
curl http://localhost:8000/debug/recent-errors
```

---

## 🔧 **Performance Optimization**

### **System Optimization**
```bash
# Increase file limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.somaxconn = 65536" >> /etc/sysctl.conf
echo "vm.max_map_count = 262144" >> /etc/sysctl.conf
sysctl -p
```

### **Application Optimization**
```env
# .env optimizations for production
WORKERS=4                    # CPU cores * 2
MAX_CONCURRENT_REQUESTS=100
CONNECTION_POOL_SIZE=20
MEMORY_THRESHOLD_PERCENT=70
CHUNK_SIZE=50000
```

---

## 🚨 **Backup & Recovery**

### **Automated Backup Script**
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/bigdata"
APP_DIR="/home/bigdata/Big-Data-Migrator"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup conversations and uploads
tar -czf $BACKUP_DIR/conversations_$DATE.tar.gz $APP_DIR/conversations/
tar -czf $BACKUP_DIR/uploads_$DATE.tar.gz $APP_DIR/uploads/

# Backup configuration
cp $APP_DIR/.env $BACKUP_DIR/env_$DATE.backup

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.backup" -mtime +30 -delete
```

### **Cron Setup**
```bash
# Daily backup at 2 AM
sudo crontab -e
# Add: 0 2 * * * /home/bigdata/backup.sh
```

---

## 🔍 **Health Checks**

### **System Health Script**
```bash
#!/bin/bash
# health_check.sh

# Check API health
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ $API_STATUS -eq 200 ]; then
    echo "✅ API: Healthy"
else
    echo "❌ API: Unhealthy ($API_STATUS)"
fi

# Check Frontend health
FRONTEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501)
if [ $FRONTEND_STATUS -eq 200 ]; then
    echo "✅ Frontend: Healthy"
else
    echo "❌ Frontend: Unhealthy ($FRONTEND_STATUS)"
fi

# Check memory usage
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ $MEMORY_USAGE -lt 80 ]; then
    echo "✅ Memory: ${MEMORY_USAGE}% used"
else
    echo "⚠️ Memory: ${MEMORY_USAGE}% used (High)"
fi

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 80 ]; then
    echo "✅ Disk: ${DISK_USAGE}% used"
else
    echo "⚠️ Disk: ${DISK_USAGE}% used (High)"
fi
```

---

## 📚 **Troubleshooting**

### **Common Issues**

1. **Port Already in Use**
```bash
# Find and kill process
sudo lsof -i :8000
sudo kill -9 PID
```

2. **Memory Issues**
```bash
# Check memory usage
free -h
# Restart services
sudo supervisorctl restart bigdata-api
```

3. **Permission Issues**
```bash
# Fix permissions
sudo chown -R bigdata:bigdata /home/bigdata/Big-Data-Migrator
sudo chmod +x start_*.py
```

### **Log Locations**
- **Application Logs**: `/var/log/supervisor/bigdata-*.log`
- **Nginx Logs**: `/var/log/nginx/access.log`, `/var/log/nginx/error.log`
- **System Logs**: `/var/log/syslog`

---

## ✅ **Deployment Checklist**

- [ ] **System Requirements**: RAM, Storage, Network
- [ ] **Dependencies**: Python, Nginx, Supervisor installed
- [ ] **Application**: Code deployed and configured
- [ ] **Environment**: Production .env configured
- [ ] **Security**: SSL certificate, firewall, user permissions
- [ ] **Process Management**: Supervisor configured and running
- [ ] **Monitoring**: Health checks and logging configured
- [ ] **Backup**: Automated backup script scheduled
- [ ] **Testing**: End-to-end functionality verified
- [ ] **Documentation**: Access URLs and credentials documented

---

## 🚀 **Go Live**

### **Final Steps**
1. **Test all endpoints** with production data
2. **Verify SSL certificate** and HTTPS redirect
3. **Check monitoring** and alerting systems
4. **Backup current state** before going live
5. **Update DNS** to point to production server
6. **Monitor system** for first 24 hours

### **Production URLs**
- **Frontend**: https://yourdomain.com
- **API**: https://yourdomain.com/api
- **Health Check**: https://yourdomain.com/api/health
- **Debug Info**: https://yourdomain.com/api/debug/system-performance

---

**🎉 Your Big Data Migrator is now ready for production use!** 