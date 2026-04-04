"""Tutorial step-by-step template."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class TutorialTemplate(BaseTemplate):
    name = "tutorial"
    display_name = "教程步骤"
    description = "Step-by-step tutorial notes extracted from how-to videos"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = _truncate_transcript(ctx.transcript)
        return f"""请根据以下视频内容，提取并整理成步骤化教程笔记。

视频标题: {ctx.meta.title}
作者: {ctx.meta.uploader}

视频内容转录:
{transcript}

要求:
1. 格式为:

## 🛠️ 教程: {ctx.meta.title}

### 前置条件
- 需要准备的工具/软件/知识

### 步骤 1: [标题]
**目标:** 这一步要达到什么效果
**操作:**
1. 具体操作 1
2. 具体操作 2
**代码/命令:** (如果有)
```
具体代码或命令
```
**注意事项:** 常见问题和解决方案

### 步骤 2: [标题]
...

### 常见问题 FAQ
- Q: 问题 1?
  A: 解答 1

### 总结
- 完成后应该达到的效果

2. 提取所有代码片段、命令行操作
3. 标注每个步骤的预期耗时
4. 标注难度等级

请直接输出教程笔记:"""
