#!/usr/bin/env node
/**
 * NoteKing Pro MCP Server
 *
 * Provides 14+ tools for video/recording/meeting to notes conversion.
 * Supports STDIO and HTTP/SSE transport modes.
 * Compatible with Cursor, Claude Desktop, OpenClaw, Codex.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { execSync } from "child_process";

const server = new McpServer({
  name: "noteking",
  version: "2.0.0",
});

function callCore(args: string[]): string {
  try {
    const result = execSync(
      `python -m cli ${args.map((a) => `"${a}"`).join(" ")}`,
      {
        encoding: "utf-8",
        timeout: 600_000,
        cwd: process.env.NOTEKING_DIR || ".",
        maxBuffer: 50 * 1024 * 1024,
      }
    );
    return result;
  } catch (err: any) {
    return `Error: ${err.message || err}`;
  }
}

function callAPI(endpoint: string, body: Record<string, any>): string {
  const apiBase = process.env.NOTEKING_API || "http://127.0.0.1:8000";
  try {
    const result = execSync(
      `curl -s -X POST "${apiBase}${endpoint}" -H "Content-Type: application/json" -d '${JSON.stringify(body)}'`,
      { encoding: "utf-8", timeout: 600_000 }
    );
    return result;
  } catch (err: any) {
    return `Error: ${err.message || err}`;
  }
}

// ---------- Template list ----------

const TEMPLATES = [
  "brief", "detailed", "mindmap", "flashcard", "quiz", "timeline",
  "exam", "tutorial", "news", "podcast", "xhs_note", "latex_pdf", "custom",
  "meeting_minutes", "lecture_notes", "interview", "brainstorm",
  "news_digest", "exam_prep", "cornell_notes", "podcast_shownotes",
  "entertainment", "smart_summary",
];

const SCENES = [
  "meeting", "lecture", "interview", "brainstorm",
  "news", "exam", "entertainment", "custom",
];

// ---------- Original Tools ----------

server.tool(
  "summarize_video",
  "Convert a video/blog URL into structured notes using the specified template",
  {
    url: z.string().describe("Video URL or local file path"),
    template: z.enum(TEMPLATES as [string, ...string[]]).default("detailed").describe("Output template"),
    custom_prompt: z.string().optional().describe("Custom prompt for 'custom' template"),
  },
  async ({ url, template, custom_prompt }) => {
    const body: Record<string, any> = { url, template };
    if (custom_prompt) body.custom_prompt = custom_prompt;
    const result = callAPI("/api/v1/summarize", body);
    try {
      const parsed = JSON.parse(result);
      return {
        content: [{ type: "text" as const, text: `# ${parsed.title || "Video Notes"}\n\n${parsed.content || result}` }],
      };
    } catch {
      return { content: [{ type: "text" as const, text: result }] };
    }
  }
);

server.tool(
  "batch_summarize",
  "Process a playlist/collection/series and generate notes for all videos",
  {
    url: z.string().describe("Playlist or collection URL"),
    template: z.enum(TEMPLATES as [string, ...string[]]).default("detailed").describe("Output template"),
  },
  async ({ url, template }) => {
    const result = callAPI("/api/v1/batch", { url, template });
    try {
      const parsed = JSON.parse(result);
      return {
        content: [{ type: "text" as const, text: `# Batch Results (${parsed.completed}/${parsed.total})\n\n${parsed.content || result}` }],
      };
    } catch {
      return { content: [{ type: "text" as const, text: result }] };
    }
  }
);

server.tool(
  "get_transcript",
  "Extract subtitles/transcript text from a video",
  { url: z.string().describe("Video URL") },
  async ({ url }) => {
    const result = callAPI(`/api/v1/transcript?url=${encodeURIComponent(url)}`, {});
    try {
      const parsed = JSON.parse(result);
      return { content: [{ type: "text" as const, text: parsed.transcript || result }] };
    } catch {
      return { content: [{ type: "text" as const, text: result }] };
    }
  }
);

server.tool(
  "get_video_info",
  "Get video metadata (title, duration, chapters, subtitle availability)",
  { url: z.string().describe("Video URL") },
  async ({ url }) => {
    try {
      const result = execSync(
        `curl -s "${process.env.NOTEKING_API || "http://127.0.0.1:8000"}/api/v1/info?url=${encodeURIComponent(url)}"`,
        { encoding: "utf-8", timeout: 60_000 }
      );
      return { content: [{ type: "text" as const, text: result }] };
    } catch (err: any) {
      return { content: [{ type: "text" as const, text: `Error: ${err.message}` }] };
    }
  }
);

server.tool(
  "search_in_transcript",
  "Search for specific text within a video transcript",
  {
    url: z.string().describe("Video URL"),
    query: z.string().describe("Search query"),
  },
  async ({ url, query }) => {
    const transcriptResult = callAPI(`/api/v1/transcript?url=${encodeURIComponent(url)}`, {});
    try {
      const parsed = JSON.parse(transcriptResult);
      const text = parsed.transcript || "";
      const lines = text.split("\n");
      const matches = lines.filter((l: string) => l.toLowerCase().includes(query.toLowerCase()));
      return {
        content: [{
          type: "text" as const,
          text: matches.length ? `Found ${matches.length} matches:\n\n${matches.join("\n")}` : `No matches found for "${query}"`,
        }],
      };
    } catch {
      return { content: [{ type: "text" as const, text: transcriptResult }] };
    }
  }
);

server.tool(
  "answer_from_video",
  "Answer a question based on video content",
  {
    url: z.string().describe("Video URL"),
    question: z.string().describe("Question to answer"),
  },
  async ({ url, question }) => {
    const result = callAPI("/api/v1/summarize", {
      url, template: "custom",
      custom_prompt: `请根据视频内容回答以下问题:\n\n${question}\n\n要求: 引用视频中的具体内容来支持你的回答。`,
    });
    try {
      const parsed = JSON.parse(result);
      return { content: [{ type: "text" as const, text: parsed.content || result }] };
    } catch {
      return { content: [{ type: "text" as const, text: result }] };
    }
  }
);

// ---------- New Recording Tools ----------

server.tool(
  "process_recording",
  "Process a local audio/video recording into structured notes with speaker diarization, noise reduction, and scene-specific templates. Supports meeting minutes, lecture notes, interview records, and more.",
  {
    file_path: z.string().describe("Path to the local audio/video file"),
    template: z.enum(TEMPLATES as [string, ...string[]]).default("meeting_minutes").describe("Output template"),
    context: z.string().optional().describe("Content description (e.g. 'AI开源圆桌会议')"),
    scene: z.enum(SCENES as [string, ...string[]]).optional().describe("Scene type"),
    num_speakers: z.number().optional().describe("Number of speakers (auto-detect if not set)"),
    denoise_level: z.number().min(0).max(3).default(1).describe("Noise reduction level (0=none, 1=light, 2=medium, 3=heavy)"),
    output_formats: z.string().default("markdown").describe("Comma-separated output formats: markdown,pdf,srt,vtt,transcript,json"),
  },
  async ({ file_path, template, context, scene, num_speakers, denoise_level, output_formats }) => {
    const args = ["process", file_path, "-t", template, "--denoise", String(denoise_level), "--format", output_formats];
    if (context) args.push("-c", context);
    if (scene) args.push("-s", scene);
    if (num_speakers) args.push("--speakers", String(num_speakers));
    const result = callCore(args);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

server.tool(
  "transcribe_file",
  "Transcribe a local audio/video file to text using the best available ASR engine",
  {
    file_path: z.string().describe("Path to the local audio/video file"),
    language: z.string().optional().describe("Language code (auto-detect if not set)"),
  },
  async ({ file_path, language }) => {
    const args = ["transcribe", file_path];
    if (language) args.push("--lang", language);
    const result = callCore(args);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

server.tool(
  "denoise_audio",
  "Apply noise reduction to an audio file",
  {
    file_path: z.string().describe("Path to the audio file"),
    level: z.number().min(1).max(3).default(2).describe("Noise reduction level (1=light, 2=medium, 3=heavy)"),
  },
  async ({ file_path, level }) => {
    const result = callCore(["denoise", file_path, "--level", String(level)]);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

server.tool(
  "merge_media_files",
  "Merge multiple audio/video files into one",
  {
    file_paths: z.array(z.string()).describe("Array of file paths to merge"),
    output_path: z.string().describe("Output file path"),
  },
  async ({ file_paths, output_path }) => {
    const args = ["merge", ...file_paths, "-o", output_path];
    const result = callCore(args);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

server.tool(
  "list_templates",
  "List all available output templates with descriptions",
  {},
  async () => {
    const templateList = TEMPLATES.map((t) => `- **${t}**`).join("\n");
    return {
      content: [{
        type: "text" as const,
        text: `# Available Templates (${TEMPLATES.length})\n\n${templateList}\n\nUse with summarize_video or process_recording tool's template parameter.`,
      }],
    };
  }
);

server.tool(
  "list_scenes",
  "List available scene types for recording processing",
  {},
  async () => {
    const sceneDescs: Record<string, string> = {
      meeting: "会议 — 参会人、议题、决议、行动项",
      lecture: "课堂/讲座 — 知识点、公式、习题",
      interview: "访谈 — Q&A、观点、立场分析",
      brainstorm: "灵感记录 — 想法、思维导图、行动建议",
      news: "新闻/播客 — 5W1H、引用、背景",
      exam: "考试复习 — 闪卡、模拟题、要点清单",
      entertainment: "娱乐 — 高光、金句、推荐指数",
      custom: "自定义 — 用户提供prompt",
    };
    const text = Object.entries(sceneDescs).map(([k, v]) => `- **${k}**: ${v}`).join("\n");
    return { content: [{ type: "text" as const, text: `# Available Scenes\n\n${text}` }] };
  }
);

// ---------- Start ----------

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("NoteKing Pro MCP Server v2.0.0 running on STDIO");
}

main().catch(console.error);
