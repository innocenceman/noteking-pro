# OpenClaw / Claude Code 安装教程

NoteKing 可以作为 Skill 安装到 OpenClaw (小龙虾)、Claude Code、Codex 中使用。

## 方式一：OpenClaw 安装

### 自动安装 (推荐)
对 OpenClaw 说：
> 请帮我安装 NoteKing 视频笔记技能

或在终端执行：
```bash
npx skills add bcefghj/noteking
```

### 手动安装
1. 克隆仓库：
```bash
git clone https://github.com/bcefghj/noteking.git ~/.openclaw/skills/noteking
```

2. 安装依赖：
```bash
cd ~/.openclaw/skills/noteking
pip install -e .
```

3. 在 OpenClaw 配置文件中注册 Skill

## 方式二：Claude Code 安装

```bash
/install-skill bcefghj/noteking
```

或手动：
```bash
git clone https://github.com/bcefghj/noteking.git ~/.claude/skills/noteking
```

## 方式三：MCP Server 安装

在 MCP 客户端配置中添加：

```json
{
  "mcpServers": {
    "noteking": {
      "command": "npx",
      "args": ["-y", "@noteking/mcp-server"],
      "env": {
        "NOTEKING_API": "http://127.0.0.1:8000"
      }
    }
  }
}
```

需要先启动 NoteKing API 服务器：
```bash
cd /path/to/noteking
pip install -e ".[api]"
python -m api.main
```

## 使用方法

安装后，直接对 AI 说：

- "帮我总结这个视频: https://www.bilibili.com/video/BVxxx"
- "把这个YouTube视频做成思维导图: https://youtu.be/xxx"
- "帮我把这个课程做成闪卡: https://www.bilibili.com/video/BVxxx"
- "这个视频的核心内容是什么?"

AI 会自动调用 NoteKing 处理视频并返回结果。

## 环境变量

确保以下环境变量已设置：

```bash
export NOTEKING_LLM_API_KEY=sk-your-api-key
export NOTEKING_LLM_MODEL=gpt-4o-mini  # 或 deepseek-chat
```
