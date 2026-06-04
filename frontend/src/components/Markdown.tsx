/**
 * Markdown — renders assistant answers with lightweight, theme-matched
 * markdown styling (lists, bold, inline code, paragraphs). Citations like
 * [fao_gap.pdf, p.25] are plain text and pass through untouched.
 */

import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";

const components: Components = {
    p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
    ul: ({ children }) => <ul className="list-disc pl-5 mb-2 space-y-1">{children}</ul>,
    ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 space-y-1">{children}</ol>,
    li: ({ children }) => <li className="leading-relaxed">{children}</li>,
    strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
    em: ({ children }) => <em className="italic">{children}</em>,
    code: ({ children }) => (
        <code className="px-1.5 py-0.5 rounded bg-muted text-brand-700 text-[13px] font-mono border border-line">{children}</code>
    ),
    a: ({ href, children }) => (
        <a href={href} target="_blank" rel="noreferrer" className="text-brand-700 underline underline-offset-2 hover:text-brand-600">{children}</a>
    ),
    h1: ({ children }) => <h1 className="text-lg font-bold text-ink mb-2 mt-1">{children}</h1>,
    h2: ({ children }) => <h2 className="text-base font-bold text-ink mb-2 mt-1">{children}</h2>,
    h3: ({ children }) => <h3 className="text-sm font-semibold text-ink mb-1 mt-1">{children}</h3>,
    blockquote: ({ children }) => (
        <blockquote className="border-l-2 border-brand-300 pl-3 italic text-ink-2 my-2">{children}</blockquote>
    ),
};

export default function Markdown({ children }: { children: string }) {
    return (
        <div className="text-[15px] text-ink-2">
            <ReactMarkdown components={components}>{children}</ReactMarkdown>
        </div>
    );
}
