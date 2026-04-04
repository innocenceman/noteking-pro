"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";

const VIDEO_TEMPLATES = [
  { name: "brief", label: "简要总结", icon: "⚡" },
  { name: "detailed", label: "详细学习笔记", icon: "📝" },
  { name: "mindmap", label: "思维导图", icon: "🧠" },
  { name: "flashcard", label: "闪卡 (Anki)", icon: "🃏" },
  { name: "quiz", label: "测验题", icon: "❓" },
  { name: "timeline", label: "时间线笔记", icon: "⏱️" },
  { name: "exam", label: "考试复习笔记", icon: "📚" },
  { name: "tutorial", label: "教程步骤", icon: "🛠️" },
  { name: "news", label: "新闻速览", icon: "📰" },
  { name: "podcast", label: "播客摘要", icon: "🎙️" },
  { name: "xhs_note", label: "小红书笔记", icon: "📕" },
  { name: "latex_pdf", label: "LaTeX PDF", icon: "📄" },
  { name: "custom", label: "自定义", icon: "✏️" },
];

const RECORDING_TEMPLATES = [
  { name: "meeting_minutes", label: "会议纪要", icon: "📋" },
  { name: "lecture_notes", label: "课堂笔记", icon: "🎓" },
  { name: "interview", label: "访谈记录", icon: "🎤" },
  { name: "brainstorm", label: "灵感记录", icon: "💡" },
  { name: "news_digest", label: "新闻摘要", icon: "📰" },
  { name: "exam_prep", label: "考试复习", icon: "📝" },
  { name: "cornell_notes", label: "康奈尔笔记", icon: "📓" },
  { name: "podcast_shownotes", label: "播客笔记", icon: "🎙️" },
  { name: "entertainment", label: "娱乐内容", icon: "🎬" },
  { name: "smart_summary", label: "智能摘要", icon: "🤖" },
];

const ALL_TEMPLATES = [...VIDEO_TEMPLATES, ...RECORDING_TEMPLATES];

const SCENES = [
  { name: "meeting", label: "会议", icon: "🤝", desc: "参会人、议题、决议、行动项" },
  { name: "lecture", label: "课堂", icon: "🎓", desc: "知识点、公式、习题" },
  { name: "interview", label: "访谈", icon: "🎤", desc: "Q&A、观点、立场" },
  { name: "brainstorm", label: "灵感", icon: "💡", desc: "想法、思维导图" },
  { name: "news", label: "新闻", icon: "📰", desc: "5W1H、引用、背景" },
  { name: "exam", label: "考试", icon: "📝", desc: "闪卡、模拟题" },
  { name: "entertainment", label: "娱乐", icon: "🎬", desc: "高光、金句" },
];

const API_BASE =
  typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://api:8000";

const HISTORY_KEY = "noteking_history";
const MAX_HISTORY = 50;

type Result = {
  title: string;
  content: string;
  template: string;
  source: string;
  platform: string;
  duration: number;
  num_speakers?: number;
  output_files?: Record<string, string>;
};

type HistoryItem = Result & { id: string; url: string; timestamp: number; contentPreview: string };

function loadHistory(): HistoryItem[] {
  if (typeof window === "undefined") return [];
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; }
}
function saveHistory(items: HistoryItem[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, MAX_HISTORY)));
}

type TabMode = "video" | "recording";

export default function Home() {
  const [mode, setMode] = useState<TabMode>("video");
  const [url, setUrl] = useState("");
  const [template, setTemplate] = useState("detailed");
  const [customPrompt, setCustomPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [darkMode, setDarkMode] = useState(false);
  const [streamContent, setStreamContent] = useState("");
  const [streamStage, setStreamStage] = useState("");
  const [streamTitle, setStreamTitle] = useState("");
  const [remaining, setRemaining] = useState<number | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [copied, setCopied] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Recording mode state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [scene, setScene] = useState("meeting");
  const [context, setContext] = useState("");
  const [numSpeakers, setNumSpeakers] = useState<number | undefined>();
  const [denoiseLevel, setDenoiseLevel] = useState(1);
  const [outputFormats, setOutputFormats] = useState(["markdown"]);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { setHistory(loadHistory()); }, []);

  useEffect(() => {
    if (mode === "video") setTemplate("detailed");
    else setTemplate("meeting_minutes");
  }, [mode]);

  const handleVideoSubmit = useCallback(async () => {
    if (!url.trim() || loading) return;
    setLoading(true); setError(""); setResult(null);
    setStreamContent(""); setStreamStage("正在连接..."); setStreamTitle("");
    abortRef.current = new AbortController();
    try {
      const resp = await fetch(`${API_BASE}/api/v1/summarize/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim(), template, custom_prompt: customPrompt }),
        signal: abortRef.current.signal,
      });
      if (resp.status === 429) {
        const e = await resp.json().catch(() => ({}));
        setRemaining(0); throw new Error(e.detail || "今日免费次数已用完");
      }
      if (!resp.ok) { const e = await resp.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${resp.status}`); }
      const lh = resp.headers.get("X-RateLimit-Remaining");
      if (lh !== null) setRemaining(parseInt(lh, 10));
      const reader = resp.body?.getReader();
      if (!reader) throw new Error("No response body");
      const decoder = new TextDecoder(); let buffer = ""; let accumulated = "";
      while (true) {
        const { done, value } = await reader.read(); if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n"); buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const ev = JSON.parse(line.slice(6).trim());
            if (ev.stage === "info") { setStreamStage(ev.message || ""); if (ev.title) setStreamTitle(ev.title); }
            else if (ev.stage === "subtitle") setStreamStage(ev.message || "提取字幕...");
            else if (ev.stage === "generating") { setStreamStage("AI 正在生成笔记..."); if (ev.content) { accumulated += ev.content; setStreamContent(accumulated); } }
            else if (ev.stage === "done") {
              const r: Result = { title: ev.title || streamTitle, content: ev.content || accumulated, template: ev.template || template, source: ev.source || "", platform: ev.platform || "", duration: ev.duration || 0 };
              setResult(r);
              const items = loadHistory();
              items.unshift({ ...r, id: Date.now().toString(36), url: url.trim(), timestamp: Date.now(), contentPreview: r.content.slice(0, 120).replace(/\n/g, " ") });
              saveHistory(items); setHistory(loadHistory());
            }
            else if (ev.stage === "error") throw new Error(ev.message);
          } catch (pe: any) { if (pe.message && !pe.message.includes("JSON")) throw pe; }
        }
      }
    } catch (e: any) { if (e.name !== "AbortError") setError(e.message || "生成失败"); }
    finally { setLoading(false); setStreamStage(""); abortRef.current = null; }
  }, [url, template, customPrompt, loading, streamTitle]);

  const handleRecordingSubmit = useCallback(async () => {
    if (!uploadFile || loading) return;
    setLoading(true); setError(""); setResult(null);
    setStreamContent(""); setStreamStage("上传文件...");
    try {
      // Upload file
      const formData = new FormData();
      formData.append("file", uploadFile);
      const uploadResp = await fetch(`${API_BASE}/api/v1/recording/upload`, { method: "POST", body: formData });
      if (!uploadResp.ok) throw new Error("上传失败");
      const { file_id } = await uploadResp.json();

      // Process
      setStreamStage("处理中...");
      const processForm = new FormData();
      processForm.append("file_id", file_id);
      processForm.append("template", template);
      if (context) processForm.append("context", context);
      processForm.append("scene", scene);
      if (numSpeakers) processForm.append("num_speakers", String(numSpeakers));
      processForm.append("denoise_level", String(denoiseLevel));
      processForm.append("output_formats", outputFormats.join(","));

      const resp = await fetch(`${API_BASE}/api/v1/recording/process`, { method: "POST", body: processForm });
      if (!resp.ok) { const e = await resp.json().catch(() => ({})); throw new Error(e.detail || "处理失败"); }
      const data = await resp.json();
      setResult({
        title: data.title || uploadFile.name,
        content: data.content || "",
        template: data.template || template,
        source: "recording",
        platform: "local",
        duration: data.duration || 0,
        num_speakers: data.num_speakers,
        output_files: data.output_files,
      });
      setStreamStage("完成!");
    } catch (e: any) { setError(e.message || "处理失败"); }
    finally { setLoading(false); setStreamStage(""); }
  }, [uploadFile, template, context, scene, numSpeakers, denoiseLevel, outputFormats, loading]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) setUploadFile(file);
  };

  const handleCancel = () => { abortRef.current?.abort(); setLoading(false); setStreamStage(""); };

  const handleCopy = () => {
    const text = result?.content || streamContent;
    if (text) { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000); }
  };

  const handleDownload = () => {
    const content = result?.content; if (!content) return;
    const isLatex = template === "latex_pdf";
    const ext = isLatex ? ".tex" : ".md";
    const mime = isLatex ? "application/x-tex" : "text/markdown";
    const blob = new Blob([content], { type: mime });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = `${result?.title || "notes"}_${template}${ext}`; a.click();
  };

  const displayContent = result?.content || streamContent;
  const currentTemplates = mode === "video" ? VIDEO_TEMPLATES : RECORDING_TEMPLATES;

  return (
    <div className={darkMode ? "dark" : ""}>
      <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)]">
        {/* Header */}
        <header className="border-b border-[var(--border)] bg-[var(--bg-secondary)]">
          <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-3xl">👑</span>
              <div>
                <h1 className="text-xl font-bold">NoteKing Pro 笔记之王</h1>
                <p className="text-xs text-[var(--text-secondary)]">
                  全网最强视频/录音处理工具 | 23种模板 | 说话人分离 | 降噪增强
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {remaining !== null && (
                <span className="text-xs text-[var(--text-secondary)] px-2 py-1 rounded border border-[var(--border)]">
                  今日剩余 {remaining} 次
                </span>
              )}
              <button onClick={() => { setShowHistory(!showHistory); setResult(null); }}
                className="px-3 py-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-primary)] transition text-sm">
                📋 历史
              </button>
              <button onClick={() => setDarkMode(!darkMode)}
                className="px-3 py-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-primary)] transition text-sm">
                {darkMode ? "☀️" : "🌙"}
              </button>
            </div>
          </div>
        </header>

        <main className="max-w-6xl mx-auto px-4 py-8">
          {/* Mode Tabs */}
          <div className="flex gap-1 mb-6 bg-[var(--bg-secondary)] rounded-xl p-1 border border-[var(--border)] w-fit">
            <button onClick={() => setMode("video")}
              className={`px-5 py-2.5 rounded-lg text-sm font-medium transition ${mode === "video" ? "bg-[var(--accent)] text-white" : "hover:bg-[var(--bg-primary)]"}`}>
              🌐 在线视频
            </button>
            <button onClick={() => setMode("recording")}
              className={`px-5 py-2.5 rounded-lg text-sm font-medium transition ${mode === "recording" ? "bg-[var(--accent)] text-white" : "hover:bg-[var(--bg-primary)]"}`}>
              🎙️ 本地录音/视频
            </button>
          </div>

          {/* Input Area */}
          <div className="bg-[var(--bg-secondary)] rounded-2xl p-6 border border-[var(--border)] mb-8">
            {mode === "video" ? (
              /* Video URL Input */
              <div className="flex gap-3 mb-4">
                <input type="text" value={url} onChange={(e) => setUrl(e.target.value)}
                  placeholder="粘贴视频链接... (B站、YouTube、抖音、小红书等)"
                  className="flex-1 px-4 py-3 rounded-xl bg-[var(--bg-primary)] border border-[var(--border)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                  onKeyDown={(e) => e.key === "Enter" && handleVideoSubmit()} />
                {loading ? (
                  <button onClick={handleCancel} className="px-6 py-3 bg-red-500 text-white rounded-xl font-medium hover:opacity-90 transition whitespace-nowrap">取消</button>
                ) : (
                  <button onClick={handleVideoSubmit} disabled={!url.trim()}
                    className="px-6 py-3 bg-[var(--accent)] text-white rounded-xl font-medium hover:opacity-90 transition disabled:opacity-50 whitespace-nowrap">
                    生成笔记
                  </button>
                )}
              </div>
            ) : (
              /* Recording Upload */
              <>
                <div className={`border-2 border-dashed rounded-xl p-8 text-center mb-4 transition cursor-pointer ${dragOver ? "border-[var(--accent)] bg-[var(--accent)]/5" : "border-[var(--border)]"}`}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}>
                  <input ref={fileInputRef} type="file" className="hidden"
                    accept="video/*,audio/*,.mp4,.mp3,.wav,.m4a,.flac,.mkv,.avi,.mov,.ogg,.aac"
                    onChange={(e) => e.target.files?.[0] && setUploadFile(e.target.files[0])} />
                  {uploadFile ? (
                    <div>
                      <p className="text-lg font-medium">📎 {uploadFile.name}</p>
                      <p className="text-sm text-[var(--text-secondary)] mt-1">
                        {(uploadFile.size / 1024 / 1024).toFixed(1)} MB | 点击更换文件
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="text-lg mb-2">🎙️ 拖放录音/视频文件到这里</p>
                      <p className="text-sm text-[var(--text-secondary)]">
                        支持 MP4, MP3, WAV, M4A, FLAC, MKV, AVI, MOV, OGG, AAC
                      </p>
                    </div>
                  )}
                </div>

                {/* Scene Selection */}
                <div className="mb-4">
                  <label className="text-sm font-medium mb-2 block">场景类型:</label>
                  <div className="flex flex-wrap gap-2">
                    {SCENES.map((s) => (
                      <button key={s.name} onClick={() => setScene(s.name)}
                        className={`px-3 py-2 rounded-lg text-sm transition ${scene === s.name ? "bg-[var(--accent)] text-white" : "bg-[var(--bg-primary)] border border-[var(--border)] hover:border-[var(--accent)]"}`}>
                        {s.icon} {s.label}
                        <span className="text-xs opacity-70 ml-1">({s.desc})</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Context */}
                <div className="mb-4">
                  <label className="text-sm font-medium mb-2 block">内容描述 (可选):</label>
                  <input type="text" value={context} onChange={(e) => setContext(e.target.value)}
                    placeholder="例如: AI开源项目圆桌讨论, 数学课第3讲, 产品周会..."
                    className="w-full px-4 py-2.5 rounded-xl bg-[var(--bg-primary)] border border-[var(--border)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] text-sm" />
                </div>

                {/* Advanced Options */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div>
                    <label className="text-sm font-medium mb-1 block">说话人数量:</label>
                    <input type="number" min={1} max={20}
                      value={numSpeakers || ""}
                      onChange={(e) => setNumSpeakers(e.target.value ? parseInt(e.target.value) : undefined)}
                      placeholder="自动检测"
                      className="w-full px-3 py-2 rounded-lg bg-[var(--bg-primary)] border border-[var(--border)] text-sm" />
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-1 block">降噪强度:</label>
                    <select value={denoiseLevel} onChange={(e) => setDenoiseLevel(parseInt(e.target.value))}
                      className="w-full px-3 py-2 rounded-lg bg-[var(--bg-primary)] border border-[var(--border)] text-sm">
                      <option value={0}>无降噪</option>
                      <option value={1}>轻度 (推荐)</option>
                      <option value={2}>中度</option>
                      <option value={3}>重度 (嘈杂环境)</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-1 block">输出格式:</label>
                    <div className="flex flex-wrap gap-1">
                      {["markdown", "srt", "vtt", "json"].map(fmt => (
                        <button key={fmt} onClick={() => {
                          setOutputFormats(prev => prev.includes(fmt) ? prev.filter(f => f !== fmt) : [...prev, fmt]);
                        }}
                          className={`px-2 py-1 rounded text-xs ${outputFormats.includes(fmt) ? "bg-[var(--accent)] text-white" : "bg-[var(--bg-primary)] border border-[var(--border)]"}`}>
                          {fmt}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                <button onClick={handleRecordingSubmit} disabled={!uploadFile || loading}
                  className="w-full py-3 bg-[var(--accent)] text-white rounded-xl font-medium hover:opacity-90 transition disabled:opacity-50">
                  {loading ? "处理中..." : "开始处理"}
                </button>
              </>
            )}

            {/* Template Selection */}
            <div className="mt-4">
              <div className="flex flex-wrap gap-2">
                {currentTemplates.map((t) => (
                  <button key={t.name} onClick={() => setTemplate(t.name)}
                    className={`px-3 py-1.5 rounded-lg text-sm transition ${template === t.name ? "bg-[var(--accent)] text-white" : "bg-[var(--bg-primary)] border border-[var(--border)] hover:border-[var(--accent)]"}`}>
                    {t.icon} {t.label}
                  </button>
                ))}
              </div>
            </div>

            {template === "custom" && (
              <textarea value={customPrompt} onChange={(e) => setCustomPrompt(e.target.value)}
                placeholder="输入自定义 Prompt..."
                className="w-full mt-3 px-4 py-3 rounded-xl bg-[var(--bg-primary)] border border-[var(--border)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] min-h-[80px] text-sm" />
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 rounded-xl p-4 mb-6">
              {error}
            </div>
          )}

          {/* Streaming progress */}
          {loading && (
            <div className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] mb-6">
              <div className="px-6 py-4 border-b border-[var(--border)] flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
                <span className="text-sm font-medium">{streamStage}</span>
                {streamTitle && <span className="text-xs text-[var(--text-secondary)] ml-auto truncate max-w-[300px]">{streamTitle}</span>}
              </div>
              {streamContent && (
                <div className="p-6 note-content prose prose-slate dark:prose-invert max-w-none max-h-[60vh] overflow-y-auto">
                  <ReactMarkdown>{streamContent}</ReactMarkdown>
                </div>
              )}
            </div>
          )}

          {/* Result */}
          {result && !loading && (
            <div className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)]">
              <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)]">
                <div>
                  <h2 className="font-semibold text-lg">{result.title}</h2>
                  <div className="flex gap-3 text-xs text-[var(--text-secondary)] mt-1">
                    <span>来源: {result.source}</span>
                    {result.platform && <span>平台: {result.platform}</span>}
                    {result.duration > 0 && <span>时长: {Math.round(result.duration / 60)} 分钟</span>}
                    {result.num_speakers && <span>说话人: {result.num_speakers}</span>}
                  </div>
                </div>
                <div className="flex gap-2">
                  <button onClick={handleCopy}
                    className="px-3 py-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-primary)] transition text-sm">
                    {copied ? "已复制!" : "复制"}
                  </button>
                  <button onClick={handleDownload}
                    className="px-3 py-1.5 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-primary)] transition text-sm">
                    下载 {template === "latex_pdf" ? ".tex" : ".md"}
                  </button>
                </div>
              </div>
              <div className="p-6 note-content prose prose-slate dark:prose-invert max-w-none">
                <ReactMarkdown>{result.content}</ReactMarkdown>
              </div>
            </div>
          )}

          {/* History */}
          {showHistory && !loading && !result && (
            <div className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)]">
              <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)]">
                <h2 className="font-semibold">历史记录 ({history.length})</h2>
                {history.length > 0 && (
                  <button onClick={() => { localStorage.removeItem(HISTORY_KEY); setHistory([]); }}
                    className="text-xs text-red-500 hover:underline">清空</button>
                )}
              </div>
              {history.length === 0 ? (
                <div className="p-8 text-center text-[var(--text-secondary)]">暂无记录</div>
              ) : (
                <div className="divide-y divide-[var(--border)]">
                  {history.map((item) => (
                    <button key={item.id} onClick={() => { setResult(item); setTemplate(item.template); setShowHistory(false); }}
                      className="w-full text-left px-6 py-4 hover:bg-[var(--bg-primary)] transition">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-sm truncate max-w-[70%]">{item.title}</span>
                        <span className="text-xs text-[var(--text-secondary)]">
                          {new Date(item.timestamp).toLocaleDateString("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                        </span>
                      </div>
                      <p className="text-xs text-[var(--text-secondary)] line-clamp-1">{item.contentPreview}</p>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Features */}
          {!result && !loading && !showHistory && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
              <FC icon="🌐" title="30+ 平台" desc="B站、YouTube、抖音、小红书、TikTok 等" />
              <FC icon="📋" title="23 种模板" desc="笔记、会议纪要、思维导图、闪卡、考试复习等" />
              <FC icon="🎙️" title="录音处理" desc="本地视频/录音文件上传，支持多文件合并" />
              <FC icon="🗣️" title="说话人分离" desc="自动识别多人对话，标注谁说了什么" />
              <FC icon="🔇" title="降噪增强" desc="三级降噪，嘈杂环境也能清晰转录" />
              <FC icon="📄" title="LaTeX PDF" desc="自动生成带目录、公式的精美 PDF 讲义" />
              <FC icon="🌍" title="多语言" desc="中英混合、50+ 语言自动检测" />
              <FC icon="📦" title="批量处理" desc="支持整个播放列表和多段录音合并" />
              <FC icon="⚡" title="多格式输出" desc="Markdown、SRT字幕、思维导图、JSON等" />
            </div>
          )}
        </main>

        <footer className="border-t border-[var(--border)] mt-16">
          <div className="max-w-6xl mx-auto px-4 py-6 flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="text-sm text-[var(--text-secondary)]">
              NoteKing Pro 笔记之王 - 全网最强视频/录音处理工具 | 开源免费
            </div>
            <a href="https://github.com/bcefghj/noteking" target="_blank"
              className="text-[var(--accent)] hover:underline flex items-center gap-1 text-sm">
              <svg className="w-4 h-4" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
              GitHub
            </a>
          </div>
        </footer>
      </div>
    </div>
  );
}

function FC({ icon, title, desc }: { icon: string; title: string; desc: string }) {
  return (
    <div className="bg-[var(--bg-secondary)] rounded-xl p-5 border border-[var(--border)]">
      <span className="text-2xl">{icon}</span>
      <h3 className="font-semibold mt-2">{title}</h3>
      <p className="text-sm text-[var(--text-secondary)] mt-1">{desc}</p>
    </div>
  );
}
