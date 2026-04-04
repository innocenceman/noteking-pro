# NoteKing 小白入门指南

完全不懂编程也能使用 NoteKing！以下是从零开始的最简单方法。

## 你需要准备什么

1. **一台电脑** (Windows / Mac / Linux 都可以)
2. **一个 LLM API Key** (用来让 AI 生成笔记)

### 如何获取 API Key？

**推荐使用 DeepSeek (便宜好用)：**
1. 打开 https://platform.deepseek.com/
2. 注册账号
3. 进入「API Keys」页面
4. 创建一个新的 API Key
5. 充值 10 元 (可以用很久)
6. 记下你的 API Key (sk-xxx 开头)

**也可以使用 OpenAI：**
1. 打开 https://platform.openai.com/
2. 注册并充值
3. 创建 API Key

## 最简单的使用方法

### 方法一：Docker (推荐，最简单)

1. 安装 Docker Desktop:
   - Mac: https://www.docker.com/products/docker-desktop/
   - Windows: https://www.docker.com/products/docker-desktop/

2. 打开终端 (Mac: 搜索"终端", Windows: 搜索"PowerShell")

3. 运行以下命令:
```bash
git clone https://github.com/bcefghj/noteking.git
cd noteking
echo "NOTEKING_LLM_API_KEY=你的API密钥" > .env
docker compose up -d
```

4. 打开浏览器，访问 http://localhost:3000

5. 粘贴视频链接，选择笔记模板，点击生成！

### 方法二：pip 安装 (需要 Python)

1. 安装 Python 3.11+: https://www.python.org/downloads/
2. 安装 ffmpeg:
   - Mac: `brew install ffmpeg`
   - Windows: 下载 https://ffmpeg.org/download.html
3. 安装 NoteKing:
```bash
pip install noteking
```
4. 配置:
```bash
noteking setup
# 按提示输入 API Key
```
5. 使用:
```bash
noteking run "https://www.bilibili.com/video/BV1xx411c79H" -t detailed
```

### 方法三：用 OpenClaw (小龙虾)

如果你已经在用 OpenClaw，直接说：
> 请帮我安装 NoteKing 视频笔记技能

然后就可以：
> 帮我总结这个视频 https://www.bilibili.com/video/BVxxx

## 13 种模板怎么选？

| 你想要... | 用这个模板 | 命令 |
|-----------|-----------|------|
| 快速了解视频讲了什么 | 简要总结 | `-t brief` |
| 系统学习视频内容 | 详细笔记 | `-t detailed` |
| 画出知识结构 | 思维导图 | `-t mindmap` |
| 做闪卡背诵 | 闪卡 | `-t flashcard` |
| 自测掌握程度 | 测验题 | `-t quiz` |
| 准备考试 | 考试笔记 | `-t exam` |
| 跟着教程学操作 | 教程步骤 | `-t tutorial` |
| 了解新闻资讯 | 新闻速览 | `-t news` |
| 听播客/访谈 | 播客摘要 | `-t podcast` |
| 发小红书 | 小红书笔记 | `-t xhs_note` |
| 打印学术讲义 | LaTeX PDF | `-t latex_pdf` |
| 自定义格式 | 自定义 | `-t custom` |

## 常见问题

**Q: 处理一个视频要多久？**
A: 通常 30 秒到 2 分钟，取决于视频长度和网速。

**Q: 支持哪些平台？**
A: B站、YouTube、抖音、小红书、快手、TikTok、Twitter 等 30+ 平台。

**Q: YouTube 视频打不开？**
A: 需要配置代理，参考 [YouTube 代理配置教程](./youtube-proxy.md)。

**Q: 可以处理整个课程吗？**
A: 可以！直接粘贴播放列表或合集链接，NoteKing 会批量处理所有视频。

**Q: API Key 会泄露吗？**
A: 不会。NoteKing 是开源的，所有处理在你本地进行，API Key 只保存在你的设备上。
