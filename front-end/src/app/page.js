'use client'
import { useState, useRef, useEffect } from "react";

// ---------------------------------------------------------------------------
// Inline markdown renderer — handles **bold** and [link](url) in reply text.
// Animation links are rendered separately as structured cards, never in here.
// ---------------------------------------------------------------------------

function renderInlineMarkdown(text) {
  const COMBINED_RE = /(\*\*(.+?)\*\*)|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g;
  const result = [];
  let lastIndex = 0;
  let match;
  let key = 0;
  while ((match = COMBINED_RE.exec(text)) !== null) {
    if (match.index > lastIndex) result.push(text.slice(lastIndex, match.index));
    if (match[1]) {
      result.push(<strong key={key++}>{match[2]}</strong>);
    } else {
      result.push(
        <a key={key++} href={match[4]} target="_blank" rel="noopener noreferrer"
           className="text-blue-600 underline hover:text-blue-800">
          {match[3]}
        </a>
      );
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) result.push(text.slice(lastIndex));
  // Wrap in a fragment so React 19 hydration never sees a raw array
  return <>{result}</>;
}

function MessageContent({ content, isUser }) {
  if (isUser) return <span className="whitespace-pre-wrap">{content}</span>;
  return <span className="whitespace-pre-wrap">{renderInlineMarkdown(content)}</span>;
}

function AnimationCards({ animations }) {
  if (!animations || animations.length === 0) return null;
  return (
    <div className="mt-2 max-w-xl space-y-2">
      {animations.map((anim, i) => (
        <div key={i} className="rounded-xl border border-blue-200 bg-blue-50 p-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-blue-400 mb-1">
            📹 Recommended Video
          </p>
          <p className="text-sm font-semibold text-blue-900 leading-snug mb-2">
            {anim.title}
          </p>
          <a
            href={anim.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-sm font-semibold text-blue-600 hover:text-blue-800 underline underline-offset-2"
          >
            Watch on Vimeo →
          </a>
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  const [mounted, setMounted] = useState(false);
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi! How can I help you today?" }
  ]);
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [city, setCity] = useState("Columbus");
  const [fitbitConnected, setFitbitConnected] = useState(false);
  const [userDocId, setUserDocId] = useState(null);
  const [showChunks, setShowChunks] = useState(false);
  const [chunksByMessage, setChunksByMessage] = useState({});
  const [animationsByMessage, setAnimationsByMessage] = useState({});
  const [expandedChunks, setExpandedChunks] = useState({});

  const bottomRef = useRef(null);

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

  // Mount guard — prevents SSR/CSR hydration mismatch.
  // The server renders a blank shell; the full UI only renders on the client.
  useEffect(() => setMounted(true), []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("fitbit") === "connected") {
      setFitbitConnected(true);
      const uid = params.get("uid");
      if (uid) setUserDocId(uid);
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Fitbit connected successfully! I can now use your activity data to give more personalized recommendations." }
      ]);
      window.history.replaceState({}, document.title, "/");
    }
  }, []);

  if (!mounted) return null;

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setInput("");

    // Snapshot length before state update so msgIndex is stable
    const nextMsgIndex = messages.length + 1;

    setMessages(prev => [...prev, { role: "user", content: userText }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/endpoint`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          message: userText,
          history,
          city,
          show_chunks: showChunks,
          ...(userDocId && { user_doc_id: userDocId }),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.error || `Server error (${response.status})`);
      }

      const data = await response.json();

      if (data.reply) {
        setMessages(prev => [...prev, { role: "assistant", content: data.reply }]);
        setHistory(data.history);

        if (data.animations && data.animations.length > 0) {
          setAnimationsByMessage(prev => ({ ...prev, [nextMsgIndex]: data.animations }));
        }

        if (data.rag_debug) {
          setChunksByMessage(prev => ({ ...prev, [nextMsgIndex]: data.rag_debug }));
        }
      } else {
        setMessages(prev => [
          ...prev,
          { role: "assistant", content: data.error || "Something went wrong. Please try again." },
        ]);
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Connection error. Is the backend running?" },
      ]);
    }

    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const connectFitbit = () => {
    window.location.href = `${API_BASE_URL}/authorize`;
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 font-sans">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4 flex items-center justify-between shadow-sm">
        <div>
          <h1 className="text-lg font-semibold text-gray-800">Health Chatbot</h1>
        </div>
        <div className="flex items-center gap-3">
          <input
            className="text-sm border rounded px-2 py-1 w-32 text-gray-600"
            value={city}
            onChange={(e) => setCity(e.target.value)}
            placeholder="City for weather"
          />
          <button
            onClick={() => setShowChunks(prev => !prev)}
            className={`text-sm px-3 py-1 rounded transition ${
              showChunks
                ? "bg-amber-100 text-amber-700"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {showChunks ? "Chunks: ON" : "Chunks: OFF"}
          </button>
          <button
            onClick={connectFitbit}
            className={`text-sm px-3 py-1 rounded transition ${
              fitbitConnected
                ? "bg-green-100 text-green-700"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {fitbitConnected ? "Fitbit Connected" : "Connect Fitbit"}
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div className="flex flex-col">
              {/* Message bubble */}
              <div
                className={`max-w-xl px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-blue-500 text-white rounded-br-sm"
                    : "bg-white text-gray-800 border rounded-bl-sm shadow-sm"
                }`}
              >
                <MessageContent content={msg.content} isUser={msg.role === "user"} />
              </div>

              {/* Animation cards — rendered below bubble, outside it */}
              {msg.role === "assistant" && (
                <AnimationCards animations={animationsByMessage[i]} />
              )}

              {/* RAG debug panel */}
              {msg.role === "assistant" && chunksByMessage[i] && (
                <div className="mt-1 max-w-xl">
                  <button
                    onClick={() => setExpandedChunks(prev => ({ ...prev, [i]: !prev[i] }))}
                    className="text-xs text-amber-600 hover:text-amber-800 underline"
                  >
                    {expandedChunks[i]
                      ? "Hide chunks"
                      : `Show chunks (${chunksByMessage[i].context_chunks_count}/${chunksByMessage[i].total_candidates} used)`}
                  </button>
                  {expandedChunks[i] && (
                    <div className="mt-2 space-y-2">
                      {/* Summary row */}
                      <div className="bg-gray-100 border border-gray-300 rounded-lg p-3 text-xs text-gray-600 space-y-1">
                        <p>
                          <span className="font-semibold">RAG query:</span>{" "}
                          {chunksByMessage[i].rag_query}
                        </p>
                        <p>
                          <span className="font-semibold">Used in context:</span>{" "}
                          {chunksByMessage[i].context_chunks_count} of{" "}
                          {chunksByMessage[i].total_candidates} candidates
                          &nbsp;·&nbsp;
                          <span className="font-semibold">Context threshold:</span>{" "}
                          {chunksByMessage[i].distance_threshold}
                          &nbsp;·&nbsp;
                          <span className="font-semibold">Animation threshold:</span>{" "}
                          {chunksByMessage[i].animation_threshold}
                        </p>
                        {chunksByMessage[i].animations_surfaced?.length > 0 && (
                          <p>
                            <span className="font-semibold">Animations surfaced:</span>{" "}
                            {chunksByMessage[i].animations_surfaced
                              .map(a => a.title)
                              .join(", ")}
                          </p>
                        )}
                        {chunksByMessage[i].error && (
                          <p className="text-red-600 font-semibold">
                            ⚠ RAG error: {chunksByMessage[i].error}
                          </p>
                        )}
                        {chunksByMessage[i].context_sent_to_llm && (
                          <details className="mt-1">
                            <summary className="cursor-pointer font-semibold text-gray-700 hover:text-gray-900">
                              Raw context sent to LLM
                            </summary>
                            <pre className="mt-1 whitespace-pre-wrap wrap-break-word text-gray-600 bg-white border border-gray-200 rounded p-2 max-h-64 overflow-y-auto">
                              {chunksByMessage[i].context_sent_to_llm}
                            </pre>
                          </details>
                        )}
                      </div>
                      {/* Individual chunks */}
                      {chunksByMessage[i].chunks.map((chunk, ci) => {
                        const used = chunk.used_in_context;
                        return (
                          <div
                            key={ci}
                            className={`border rounded-lg p-3 text-xs ${
                              used
                                ? "bg-amber-50 border-amber-300 text-gray-700"
                                : "bg-gray-50 border-gray-200 text-gray-400"
                            }`}
                          >
                            <div className="flex justify-between items-start gap-2 mb-1 flex-wrap">
                              <div className="flex flex-col gap-0.5">
                                <span className={`font-semibold ${used ? "text-amber-800" : "text-gray-400"}`}>
                                  {chunk.metadata?.source || chunk.id}
                                </span>
                                {chunk.metadata?.section_title && (
                                  <span className={used ? "text-amber-700" : "text-gray-400"}>
                                    § {chunk.metadata.section_title}
                                  </span>
                                )}
                              </div>
                              <div className="flex flex-col items-end gap-0.5 shrink-0">
                                <span className={used ? "text-amber-600" : "text-gray-400"}>
                                  dist: {chunk.distance}
                                </span>
                                <span className={`font-semibold ${used ? "text-green-600" : "text-gray-400"}`}>
                                  {used ? "✓ used" : "✗ filtered"}
                                </span>
                              </div>
                            </div>
                            {chunk.metadata?.animation_url && (
                              <div className="mb-1">
                                <span className="inline-block bg-blue-100 text-blue-700 rounded px-1.5 py-0.5">
                                  📹 has animation
                                </span>
                              </div>
                            )}
                            <p className="whitespace-pre-wrap leading-relaxed">{chunk.text}</p>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border rounded-2xl rounded-bl-sm px-4 py-3 text-sm text-gray-400 shadow-sm">
              Thinking...
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="bg-white border-t px-4 py-4 flex gap-3 items-end">
        <textarea
          className="flex-1 border text-black px-5 py-2 rounded-xl text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-300"
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about exercise, goals, or your health..."
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="bg-blue-500 text-white px-5 py-2 rounded-xl text-sm hover:bg-blue-600 transition disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  );
}
