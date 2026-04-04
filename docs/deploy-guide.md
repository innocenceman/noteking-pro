# NoteKing Pro 部署教程

## 方案 A：一键部署到 Railway（最简单，推荐新手）

### 1. 注册 Railway
1. 访问 [railway.app](https://railway.app)
2. 用 GitHub 账号注册（免费，500小时/月）

### 2. 一键部署
1. Fork 本仓库到你的 GitHub
2. 在 Railway 控制台点击 "New Project" → "Deploy from GitHub repo"
3. 选择你 fork 的 `noteking` 仓库
4. 设置环境变量:
   - `NOTEKING_LLM_API_KEY` = 你的 API Key
   - `NOTEKING_LLM_BASE_URL` = `https://api.minimax.chat/v1` (MiniMax 中国区)
   - `NOTEKING_LLM_MODEL` = `MiniMax-M2.7`
5. 点击 Deploy

### 3. 访问
Railway 会自动分配一个域名，如 `noteking-xxx.up.railway.app`

---

## 方案 B：Docker 部署到云服务器（推荐正式运营）

### 1. 购买云服务器

**腾讯云** (推荐新用户):
- 访问 [cloud.tencent.com](https://cloud.tencent.com)
- 新用户优惠：2核4G 约 50-100 元/月
- 选择 Ubuntu 22.04 系统
- 开放安全组端口：22 (SSH), 80 (HTTP), 443 (HTTPS), 8000 (API)

**阿里云**:
- 访问 [aliyun.com](https://www.aliyun.com)
- 新用户优惠类似
- ECS 选择 2核4G 即可

### 2. 连接服务器

```bash
ssh root@你的服务器IP
```

### 3. 安装 Docker

```bash
curl -fsSL https://get.docker.com | sh
systemctl enable docker && systemctl start docker
docker compose version  # 确认安装成功
```

### 4. 部署 NoteKing Pro

```bash
# 克隆代码
git clone https://github.com/bcefghj/noteking.git
cd noteking

# 配置环境变量
cp .env.example .env
nano .env
# 填入你的 API Key 等配置

# 一键启动
docker compose up -d

# 查看日志
docker compose logs -f
```

### 5. 配置域名和 HTTPS

```bash
# 安装 Nginx
apt install -y nginx certbot python3-certbot-nginx

# 配置 Nginx
cat > /etc/nginx/sites-available/noteking << 'EOF'
server {
    listen 80;
    server_name 你的域名.com;

    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
    }
}
EOF

ln -sf /etc/nginx/sites-available/noteking /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# 配置 HTTPS (需要先将域名 DNS 解析到服务器IP)
certbot --nginx -d 你的域名.com
```

### 6. 更新维护

```bash
cd noteking
git pull
docker compose down
docker compose up -d --build
```

---

## 方案 C：带 GPU 的专业部署

适用于需要最佳 ASR 质量的场景（FunASR + pyannote 本地推理）。

### 服务器选择
- 腾讯云 GN7 或 阿里云 GPU 实例
- 至少 NVIDIA T4 (16GB 显存)
- 推荐 4核16G + T4 GPU

### 部署

```bash
git clone https://github.com/bcefghj/noteking.git
cd noteking
cp .env.example .env
nano .env  # 配置 API Key 和 HF_TOKEN

# 使用 GPU 版本
docker compose -f docker-compose.gpu.yml up -d
```

---

## 个人本地使用

### macOS / Linux

```bash
# 安装依赖
brew install ffmpeg  # macOS
# apt install ffmpeg  # Ubuntu

git clone https://github.com/bcefghj/noteking.git
cd noteking

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装
pip install -e ".[meeting,asr]"

# 配置
noteking setup --api-key "你的API_KEY" --base-url "https://api.minimax.chat/v1" --model "MiniMax-M2.7"

# 使用
noteking process meeting.mp4 -t meeting_minutes -c "产品周会"
noteking run "https://www.bilibili.com/video/BV1xx" -t detailed
```

### Windows

```powershell
# 安装 FFmpeg: https://www.gyan.dev/ffmpeg/builds/
# 安装 Python 3.11+: https://www.python.org/downloads/

git clone https://github.com/bcefghj/noteking.git
cd noteking
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[meeting,asr]"
noteking setup --api-key "你的API_KEY"
```

---

## MiniMax API 配置说明

如果使用 MiniMax M2.7:

```bash
NOTEKING_LLM_API_KEY=sk-cp-你的key
NOTEKING_LLM_BASE_URL=https://api.minimax.chat/v1
NOTEKING_LLM_MODEL=MiniMax-M2.7
```

## 常见问题

**Q: 处理大文件超时怎么办？**
A: 增加 Nginx 的 `proxy_read_timeout` 和 `client_max_body_size`

**Q: 说话人分离不准确？**
A: 1) 指定 `--speakers` 数量  2) 安装 pyannote.audio 并设置 HF_TOKEN

**Q: 中文识别不准？**
A: 安装 `funasr` (`pip install funasr modelscope`)，会自动使用 Paraformer-zh

**Q: 如何节省 LLM 调用成本？**
A: ASR/降噪/分离全在本地，只有生成笔记才调用 LLM。一次会议约 1-2 次调用。
