# Vercel 部署教程 (免费方案)

适合个人使用，利用 Vercel 免费额度部署前端，后端使用 Railway 或本地运行。

## 方案说明

- **前端**: 部署到 Vercel (免费)
- **后端**: 本地运行 或 Railway (免费额度)
- **适合**: 个人使用，不需要公网后端

## 部署前端到 Vercel

### 方式一：一键部署

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/bcefghj/noteking&root-directory=web)

### 方式二：手动部署

1. Fork 项目到你的 GitHub
2. 打开 https://vercel.com/ 并用 GitHub 登录
3. Import 你 fork 的仓库
4. Root Directory 设置为 `web`
5. 在 Environment Variables 中添加：
   - `NEXT_PUBLIC_API_URL` = 你的后端 API 地址
6. Deploy

## 运行后端 (本地)

```bash
# 安装依赖
pip install -e ".[api]"

# 启动 API 服务
python -m api.main
# API 运行在 http://localhost:8000
```

## 注意事项

- Vercel 免费版有 100GB/月带宽限制
- Serverless Functions 有 10 秒超时限制 (不适合长视频处理)
- 推荐方案: Vercel 前端 + 本地/VPS 后端
