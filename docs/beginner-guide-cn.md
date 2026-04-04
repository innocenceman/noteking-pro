# NoteKing 保姆级入门教程（中文版）

> 完全不懂编程也能使用！每一步都有详细解释，跟着做就行。

---

## 目录

1. [你需要准备什么](#1-你需要准备什么)
2. [获取大模型 API Key](#2-获取大模型-api-key)
3. [方法一：Docker 一键部署（推荐小白）](#3-方法一docker-一键部署推荐小白)
4. [方法二：pip 安装（灵活但稍复杂）](#4-方法二pip-安装灵活但稍复杂)
5. [方法三：OpenClaw 小龙虾（对话式使用）](#5-方法三openclaw-小龙虾对话式使用)
6. [生成你的第一个笔记](#6-生成你的第一个笔记)
7. [生成 LaTeX PDF 讲义](#7-生成-latex-pdf-讲义)
8. [处理整个课程合集](#8-处理整个课程合集)
9. [13 种模板怎么选](#9-13-种模板怎么选)
10. [常见问题排查](#10-常见问题排查)

---

## 1. 你需要准备什么

| 必须 | 说明 |
|------|------|
| 一台电脑 | Windows / Mac / Linux 都行 |
| 一个大模型 API Key | 让 AI 帮你生成笔记（下面教你怎么获取） |

| 可选 | 说明 |
|------|------|
| Docker Desktop | 用方法一部署最简单 |
| Python 3.11+ | 用方法二需要 |
| TinyTeX | 想要生成专业 LaTeX PDF 讲义需要 |

---

## 2. 获取大模型 API Key

NoteKing 需要一个大模型 API 来生成笔记。以下推荐几家，选一家就行。

### 推荐一：MiniMax（国产，性价比高）

1. 打开 MiniMax 开放平台：https://www.minimaxi.com/
2. 点击右上角「注册/登录」
3. 注册后进入控制台
4. 在左侧菜单找到「API Keys」
5. 点击「创建 API Key」
6. **复制并保存** 生成的 Key（格式类似 `sk-cp-xxx...`）
7. 在「套餐」页面选择购买：
   - **Starter 月度套餐**：约 30 元/月，600 次调用，够个人使用
   - **Max 月度套餐**：约 200 元/月，4500 次调用，适合重度用户
8. 推荐使用 `MiniMax-M2.7` 模型

> 💡 **记下这三个信息，后面要用**：
> - API Key：`sk-cp-xxx...`
> - API 地址：`https://api.minimax.chat/v1`
> - 模型名：`MiniMax-M2.7`

### 推荐二：DeepSeek（国产，便宜好用）

1. 打开 DeepSeek 开放平台：https://platform.deepseek.com/
2. 注册账号（支持手机号注册）
3. 进入「API Keys」页面
4. 点击「创建 API Key」
5. 复制并保存 Key（格式类似 `sk-xxx...`）
6. 充值 10 元（能用很久）

> 💡 **记下这三个信息**：
> - API Key：`sk-xxx...`
> - API 地址：`https://api.deepseek.com/v1`
> - 模型名：`deepseek-chat`

### 推荐三：OpenAI（效果最好，但贵且需要翻墙）

1. 打开 https://platform.openai.com/
2. 注册账号（需要海外手机号）
3. 创建 API Key
4. 充值

> 💡 **记下这三个信息**：
> - API Key：`sk-xxx...`
> - API 地址：`https://api.openai.com/v1`
> - 模型名：`gpt-4o` 或 `gpt-4o-mini`

---

## 3. 方法一：Docker 一键部署（推荐小白）

> 这是最简单的方法，不需要懂编程，装好 Docker 后三行命令搞定。

### 第一步：安装 Docker Desktop

**Mac 用户：**

1. 打开浏览器，访问 https://www.docker.com/products/docker-desktop/
2. 点击「Download for Mac」
3. 下载完成后，双击 `.dmg` 文件
4. 把 Docker 图标拖到「Applications」文件夹
5. 在「启动台（Launchpad）」里找到 Docker，点击打开
6. 首次打开可能需要输入电脑密码授权
7. 等待 Docker 启动完成（顶部菜单栏会出现一个鲸鱼图标🐳）

**Windows 用户：**

1. 打开浏览器，访问 https://www.docker.com/products/docker-desktop/
2. 点击「Download for Windows」
3. 下载完成后，双击 `.exe` 文件安装
4. 安装过程中如果提示启用 WSL 2，点击「确定」
5. 安装完成后重启电脑
6. 重启后 Docker Desktop 会自动启动（任务栏右下角会出现鲸鱼图标🐳）

> ⚠️ **Windows 注意**：如果提示需要安装 WSL 2，打开 PowerShell（管理员模式），运行：
> ```powershell
> wsl --install
> ```
> 然后重启电脑。

### 第二步：打开终端

**Mac：**
- 按 `Command + 空格`，输入「终端」，回车
- 或者在「启动台」→「其他」→「终端」

**Windows：**
- 按 `Win + X`，选择「Windows PowerShell」
- 或者在搜索栏输入「PowerShell」，点击打开

### 第三步：运行以下命令

在终端里，**一行一行** 复制粘贴以下命令，每输入一行按回车：

```bash
git clone https://github.com/bcefghj/noteking.git
```

> 这会下载 NoteKing 的代码到你的电脑。等待下载完成。

```bash
cd noteking
```

> 进入 NoteKing 目录。

```bash
echo "NOTEKING_LLM_API_KEY=这里替换成你的API密钥" > .env
echo "NOTEKING_LLM_BASE_URL=https://api.minimax.chat/v1" >> .env
echo "NOTEKING_LLM_MODEL=MiniMax-M2.7" >> .env
```

> ⚠️ **重要**：把 `这里替换成你的API密钥` 替换成你在第 2 步获取的真实 API Key！
> 如果你用的不是 MiniMax，把后面两行的地址和模型名也改成对应的。

```bash
docker compose up -d
```

> 这会启动 NoteKing 服务。首次运行需要下载一些东西，可能要等 3-5 分钟。

### 第四步：使用

1. 打开浏览器
2. 在地址栏输入 `http://localhost:3000`
3. 你会看到 NoteKing 的网页界面
4. 粘贴一个视频链接（比如 B 站的）
5. 选择模板（比如「详细笔记」）
6. 点击「生成」
7. 等待几十秒，你的笔记就生成了！

### 停止和重启

```bash
# 停止
docker compose down

# 重新启动
docker compose up -d
```

---

## 4. 方法二：pip 安装（灵活但稍复杂）

> 适合有一点编程基础的用户，或者想要更灵活地使用 NoteKing 的人。

### 第一步：安装 Python

**Mac：**

1. 打开终端
2. 输入以下命令安装 Homebrew（如果还没装过）：
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. 安装 Python：
   ```bash
   brew install python@3.11
   ```
4. 验证安装：
   ```bash
   python3 --version
   ```
   应该显示 `Python 3.11.x` 或更高版本。

**Windows：**

1. 打开 https://www.python.org/downloads/
2. 点击「Download Python 3.12.x」
3. 运行安装程序
4. ⚠️ **重要**：勾选 「Add Python to PATH」（把 Python 加入系统路径）
5. 点击「Install Now」
6. 安装完成后，打开 PowerShell，输入：
   ```powershell
   python --version
   ```
   应该显示 `Python 3.12.x`

### 第二步：安装 ffmpeg

**Mac：**
```bash
brew install ffmpeg
```

**Windows：**
1. 打开 https://ffmpeg.org/download.html
2. 选择 Windows 版本，下载
3. 解压到一个文件夹，比如 `C:\ffmpeg`
4. 把 `C:\ffmpeg\bin` 加入系统 PATH 环境变量：
   - 右键「此电脑」→「属性」→「高级系统设置」→「环境变量」
   - 在「系统变量」里找到 `Path`，双击
   - 点击「新建」，输入 `C:\ffmpeg\bin`
   - 点击「确定」保存

### 第三步：安装 NoteKing 依赖

打开终端，运行：

```bash
pip install yt-dlp openai httpx pillow imagehash scenedetect opencv-python-headless
```

> 等待安装完成，可能要 1-2 分钟。

### 第四步：下载 NoteKing 代码

```bash
git clone https://github.com/bcefghj/noteking.git
cd noteking
```

### 第五步：配置 API Key

**Mac / Linux：**
```bash
export NOTEKING_LLM_API_KEY="你的API密钥"
export NOTEKING_LLM_BASE_URL="https://api.minimax.chat/v1"
export NOTEKING_LLM_MODEL="MiniMax-M2.7"
```

**Windows PowerShell：**
```powershell
$env:NOTEKING_LLM_API_KEY = "你的API密钥"
$env:NOTEKING_LLM_BASE_URL = "https://api.minimax.chat/v1"
$env:NOTEKING_LLM_MODEL = "MiniMax-M2.7"
```

> ⚠️ 每次打开新的终端窗口都要重新设置。想永久保存可以写入 `.bashrc` / `.zshrc` 或系统环境变量。

### 第六步：使用

```bash
# 生成详细笔记
python -m noteking.cli run "https://www.bilibili.com/video/BV1T2k6BaEeC?p=7" --template detailed

# 生成简要总结
python -m noteking.cli run "https://www.bilibili.com/video/BV1T2k6BaEeC?p=7" --template brief

# 生成测验题
python -m noteking.cli run "https://www.bilibili.com/video/BV1T2k6BaEeC?p=7" --template quiz
```

---

## 5. 方法三：OpenClaw 小龙虾（对话式使用）

> 如果你已经在用 OpenClaw，这是最省事的方法。

1. 打开 OpenClaw
2. 对它说：
   > 请帮我安装 NoteKing 视频笔记技能
3. 安装完成后，直接说：
   > 帮我总结这个视频 https://www.bilibili.com/video/BV1T2k6BaEeC
4. 还可以指定模板：
   > 帮我用考试复习模板总结这个视频 https://www.bilibili.com/video/BV1T2k6BaEeC

---

## 6. 生成你的第一个笔记

假设你已经通过上面任意一种方法安装好了 NoteKing，现在来生成你的第一个笔记。

### 示例：总结一个 B 站视频

1. 找到你想总结的 B 站视频，复制它的链接
   - 比如：`https://www.bilibili.com/video/BV1T2k6BaEeC?p=7`

2. 在终端运行：
   ```bash
   python -m noteking.cli run "https://www.bilibili.com/video/BV1T2k6BaEeC?p=7" --template detailed
   ```

3. 等待 30 秒到 2 分钟（取决于视频长度和 API 速度）

4. 生成的笔记会保存在当前目录，终端会显示输出文件路径

### 示例：总结一个 YouTube 视频

```bash
# 如果在国内需要代理
export NOTEKING_PROXY="http://127.0.0.1:7890"

python -m noteking.cli run "https://www.youtube.com/watch?v=xxxxx" --template brief
```

---

## 7. 生成 LaTeX PDF 讲义

> 这是 NoteKing 最强大的功能——生成带关键帧截图的专业 PDF 讲义。

### 第一步：安装 TinyTeX

TinyTeX 是一个很小的 LaTeX 发行版，只需要几分钟就能装好。

**Mac / Linux：**
```bash
curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
```

装完后安装需要的 LaTeX 包：
```bash
# 把 TinyTeX 加入 PATH（Mac 示例）
export PATH=$PATH:~/Library/TinyTeX/bin/universal-darwin

# 安装必需的 LaTeX 包
tlmgr install ctex tcolorbox listings booktabs float fancyhdr xcolor enumitem etoolbox environ trimspaces adjustbox collectbox caption hyperref geometry graphicx fontspec xunicode xltxtra
```

**Windows：**
```powershell
Invoke-WebRequest -Uri "https://yihui.org/tinytex/install-bin-windows.bat" -OutFile "install-tinytex.bat"
.\install-tinytex.bat
```

### 第二步：验证安装

```bash
xelatex --version
```

应该输出 `XeTeX 3.x.x` 之类的版本信息。如果提示"command not found"：
- Mac：运行 `export PATH=$PATH:~/Library/TinyTeX/bin/universal-darwin`
- Windows：重新打开 PowerShell

### 第三步：生成 PDF 讲义

```bash
python -m noteking.cli run "https://www.bilibili.com/video/BV1T2k6BaEeC?p=7" --template latex_pdf
```

生成的 PDF 会包含：
- 封面页
- 自动目录
- 关键帧截图
- 高亮知识框（重点/背景/注意）
- 代码高亮
- 数学公式
- 页眉页脚

---

## 8. 处理整个课程合集

NoteKing 支持一键处理整个 B 站合集或 YouTube 播放列表。

### B 站合集

```bash
# 直接给合集链接，NoteKing 会自动识别所有分P
python -m noteking.cli run "https://www.bilibili.com/video/BV1T2k6BaEeC" --template detailed --batch
```

### YouTube 播放列表

```bash
python -m noteking.cli run "https://www.youtube.com/playlist?list=PLxxxx" --template detailed --batch
```

### 控制并发数

```bash
# 同时处理 3 个视频（默认）
python -m noteking.cli run "..." --batch --workers 3

# 单线程处理（慢但省 API 额度）
python -m noteking.cli run "..." --batch --workers 1
```

---

## 9. 13 种模板怎么选

| 你的需求 | 推荐模板 | 命令参数 |
|----------|----------|----------|
| 快速了解视频讲了什么 | 简要总结 | `-t brief` |
| 系统学习视频内容 | 详细笔记 | `-t detailed` |
| 画出知识结构图 | 思维导图 | `-t mindmap` |
| 做闪卡背诵（导入 Anki） | 闪卡 | `-t flashcard` |
| 自测掌握程度 | 测验题 | `-t quiz` |
| 按时间线整理知识点 | 时间线 | `-t timeline` |
| 准备考试（公式速查+真题） | 考试复习 | `-t exam` |
| 跟着教程学操作 | 教程步骤 | `-t tutorial` |
| 了解新闻资讯 | 新闻速览 | `-t news` |
| 整理播客/访谈内容 | 播客摘要 | `-t podcast` |
| 发小红书分享 | 小红书笔记 | `-t xhs_note` |
| 打印专业讲义（带截图） | LaTeX PDF | `-t latex_pdf` |
| 完全自定义格式 | 自定义 | `-t custom` |

---

## 10. 常见问题排查

### Q: `git` 命令提示 "command not found"

**Mac：** 打开终端，运行 `xcode-select --install`，会弹出安装 Command Line Tools 的提示，点击安装。

**Windows：** 去 https://git-scm.com/download/win 下载安装 Git。

---

### Q: `pip` 命令提示 "command not found"

**Mac：** 试试 `pip3` 而不是 `pip`。

**Windows：** 重新安装 Python，确保勾选了 "Add Python to PATH"。

---

### Q: Docker 启动失败

1. 确保 Docker Desktop 已经打开（看看任务栏有没有鲸鱼图标🐳）
2. Windows 用户：确保已经安装了 WSL 2
3. 试试重启 Docker Desktop

---

### Q: API 报错 "Invalid API Key"

1. 检查 API Key 是否复制完整（不要多复制空格）
2. 检查 API 地址是否正确
3. 检查 API 套餐是否还有余额

---

### Q: YouTube 视频无法下载

在国内需要配置代理：

```bash
export NOTEKING_PROXY="http://127.0.0.1:7890"
```

把 `7890` 换成你自己的代理端口。详见 [YouTube 代理配置教程](youtube-proxy.md)。

---

### Q: LaTeX PDF 编译失败，提示缺少 xxx.sty

运行以下命令安装缺少的包（把 `xxx` 换成提示缺少的包名）：

```bash
tlmgr install xxx
```

常见需要安装的包：
```bash
tlmgr install ctex tcolorbox listings booktabs float fancyhdr xcolor enumitem etoolbox environ trimspaces adjustbox collectbox caption
```

---

### Q: 生成的笔记内容不够好 / 太短

1. 换一个更好的模型（如 `gpt-4o`）
2. 确保视频有字幕（有字幕的效果远好于纯视觉模式）
3. 对于 B 站视频，配置 SESSDATA 可以获取更高质量的音频

---

### Q: 处理速度很慢

1. 检查你的 API 套餐是否有限速
2. MiniMax Max 套餐比 Starter 快得多
3. 批量处理时适当降低并发数（`--workers 1`）

---

> 💬 还有其他问题？在 [GitHub Issues](https://github.com/bcefghj/noteking/issues) 提问，或小红书私信 **bcefghj**。
