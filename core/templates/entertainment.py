"""Entertainment template: highlights, quotes, and rating for casual content."""

from .base import BaseTemplate, TemplateContext, _truncate_transcript


class EntertainmentTemplate(BaseTemplate):
    name = "entertainment"
    display_name = "娱乐内容"
    description = "Entertainment highlights with quotes, key moments, and rating"

    def build_prompt(self, ctx: TemplateContext) -> str:
        transcript = ctx.extra.get("diarized_transcript", "") or _truncate_transcript(ctx.transcript)
        user_context = ctx.extra.get("context", "")
        context_info = f"\n描述: {user_context}" if user_context else ""

        return f"""请根据以下娱乐/休闲内容的录音转录稿，生成一份精彩内容回顾。

标题: {ctx.meta.title}
时长: {ctx.meta.duration:.0f} 秒{context_info}

录音转录稿:
{transcript}

请按以下格式输出:

# 内容回顾: [标题]

## 一句话推荐
[用一句吸引人的话概括这个内容]

## 推荐指数: ⭐⭐⭐⭐ (1-5星)

## 适合人群
[什么样的人会喜欢这个内容]

## 内容概要
[3-5句话介绍内容梗概，不含剧透]

## 精彩时刻 🎬
1. **[时间戳] [精彩瞬间标题]**: [描述]
2. **[时间戳] [精彩瞬间标题]**: [描述]

## 金句收集 💬
- > "[金句1]"
- > "[金句2]"
- > "[金句3]"

## 笑点/泪点/高能预警
- [描述令人印象深刻的瞬间]

## 关键信息
[提取的有用信息或知识点，如果有]

## 个人观后感
[基于内容生成的简短评价]

要求:
1. 风格轻松活泼
2. 突出精彩瞬间
3. 保留最有趣的原话
4. 不过度剧透
5. 用中文输出"""
