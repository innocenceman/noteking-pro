# Docker 一键部署教程

最简单的部署方式，适合所有用户。

## 前置要求

- 安装 [Docker](https://docs.docker.com/get-docker/)
- 安装 [Docker Compose](https://docs.docker.com/compose/install/)

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/bcefghj/noteking.git
cd noteking

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 LLM API Key

# 3. 启动服务
docker compose up -d

# 4. 访问 Web 界面
# 打开浏览器访问 http://localhost:3000
```

## 环境变量配置

编辑 `.env` 文件：

```env
# 必填: LLM API Key (OpenAI / DeepSeek / 通义千问 等)
NOTEKING_LLM_API_KEY=sk-your-api-key-here

# 可选: 使用国产模型 (推荐 DeepSeek，性价比高)
NOTEKING_LLM_BASE_URL=https://api.deepseek.com
NOTEKING_LLM_MODEL=deepseek-chat

# 可选: 代理 (中国用户访问 YouTube 需要)
NOTEKING_PROXY=socks5://127.0.0.1:7890

# 可选: B站登录 (获取高清和会员内容)
BILIBILI_SESSDATA=your_sessdata_here
```

## GPU 加速版本 (本地语音识别)

如果你有 NVIDIA GPU，可以使用 GPU 版本加速本地语音识别：

```bash
docker compose -f docker-compose.gpu.yml up -d
```

## 常用命令

```bash
# 查看日志
docker compose logs -f

# 停止服务
docker compose down

# 更新到最新版
git pull && docker compose up -d --build

# 重启服务
docker compose restart
```

## 端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| Web 前端 | 3000 | 浏览器访问 |
| API 后端 | 8000 | REST API |

## 常见问题

**Q: Docker 镜像下载太慢？**
A: 配置 Docker 镜像加速器，参考 https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors

**Q: 如何修改端口？**
A: 编辑 `docker-compose.yml` 中的 `ports` 映射

**Q: 需要多少磁盘空间？**
A: 基础镜像约 2GB，加上缓存数据预留 5GB 以上
