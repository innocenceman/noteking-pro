import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NoteKing Pro - 全网最强视频/录音处理工具",
  description:
    "全网最强视频/录音处理工具。支持30+平台视频、本地录音/视频上传、说话人分离、降噪增强、23种输出模板。",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)]">
        {children}
      </body>
    </html>
  );
}
