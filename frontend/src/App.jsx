import React, { useEffect, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

const HISTORY_KEY = "unc_ai_history_v1";
const SYSTEM_PROMPT_KEY = "unc_ai_system_prompt_v1";
const PERSONA_KEY = "unc_ai_persona_v1";

function getPersonaSuffix(persona) {
  switch (persona) {
    case "coder":
      return " You are a senior software engineer. Prefer concise, working code and explain trade-offs when useful.";
    case "teacher":
      return " You are a calm, structured teacher. Break concepts into clear, sequential steps.";
    case "creative":
      return " You are a creative collaborator. Suggest bold, unusual ideas but remain practical when needed.";
    default:
      return "";
  }
}

export default function App() {
  const [history, setHistory] = useState([]);
  const [message, setMessage] = useState("");
  const [model, setModel] = useState(
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
  );
  const [temperature, setTemperature] = useState(0.7);
  const [persona, setPersona] = useState("default");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [status, setStatus] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [useSearch, setUseSearch] = useState(false);
  const [useStreaming, setUseStreaming] = useState(true);

  const [imagePrompt, setImagePrompt] = useState("");
  const [imageStatus, setImageStatus] = useState("");
  const [imageUrl, setImageUrl] = useState("");

  const [searchQuery, setSearchQuery] = useState("");
  const [searchStatus, setSearchStatus] = useState("");
  const [searchResults, setSearchResults] = useState([]);

  const chatRef = useRef(null);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(HISTORY_KEY);
      if (raw) {
        setHistory(JSON.parse(raw));
      }
    } catch (e) {
      console.warn("Could not load history", e);
    }

    const savedPrompt = window.localStorage.getItem(SYSTEM_PROMPT_KEY);
    if (savedPrompt) {
      setSystemPrompt(savedPrompt);
    } else {
      setSystemPrompt(
        "You are an advanced assistant owned by the user. Be direct, helpful, and honest. Avoid anything illegal or genuinely harmful."
      );
    }

    const savedPersona = window.localStorage.getItem(PERSONA_KEY);
    if (savedPersona) {
      setPersona(savedPersona);
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch (e) {
      console.warn("Could not save history", e);
    }
  }, [history]);

  useEffect(() => {
    try {
      window.localStorage.setItem(SYSTEM_PROMPT_KEY, systemPrompt);
    } catch (e) {
      console.warn("Could not save system prompt", e);
    }
  }, [systemPrompt]);

  useEffect(() => {
    try {
      window.localStorage.setItem(PERSONA_KEY, persona);
    } catch (e) {
      console.warn("Could not save persona", e);
    }
  }, [persona]);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [history]);

  async function handleSend() {
    const trimmed = message.trim();
    if (!trimmed || isSending) return;

    const newHistory = [...history, { role: "user", content: trimmed }];
    setHistory(newHistory);
    setMessage("");
    setIsSending(true);
    setStatus("Thinking…");

    if (useStreaming) {
      await sendStreaming(trimmed, newHistory);
    } else {
      await sendNonStreaming(trimmed);
    }

    setIsSending(false);
    setStatus("");
  }

  async function sendNonStreaming(trimmed) {
    const tempSafe =
      isNaN(Number(temperature)) || temperature === null
        ? 0.7
        : Number(temperature);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          history,
          model,
          temperature: tempSafe,
          system_prompt: systemPrompt + getPersonaSuffix(persona),
          use_search: useSearch,
        }),
      });

      if (!res.ok) {
        let errText = res.statusText;
        try {
          const err = await res.json();
          errText = err.detail || errText;
        } catch (_) {}
        setHistory((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${errText}` },
        ]);
        return;
      }

      const data = await res.json();
      const updatedHistory = data.history || [];
      setHistory(updatedHistory);

      if (data.search_snippets && Array.isArray(data.search_snippets)) {
        const joined = data.search_snippets.join("\n\n");
        setHistory((prev) => [
          ...prev,
          { role: "search", content: `Web search context:\n${joined}` },
        ]);
      }
    } catch (e) {
      setHistory((prev) => [
        ...prev,
        { role: "assistant", content: `Network error: ${e}` },
      ]);
    }
  }

  async function sendStreaming(trimmed, historyWithUser) {
    const tempSafe =
      isNaN(Number(temperature)) || temperature === null
        ? 0.7
        : Number(temperature);

    const body = {
      message: trimmed,
      history,
      model,
      temperature: tempSafe,
      system_prompt: systemPrompt + getPersonaSuffix(persona),
      use_search: useSearch,
    };

    try {
      const res = await fetch(`${API_BASE}/chat-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok || !res.body) {
        let errText = res.statusText;
        try {
          const err = await res.json();
          errText = err.detail || errText;
        } catch (_) {}
        setHistory((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${errText}` },
        ]);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let done = false;

      setHistory([...historyWithUser, { role: "assistant", content: "" }]);
      let aiIndex = historyWithUser.length;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (done || !value) break;

        const chunk = decoder.decode(value, { stream: true });
        if (!chunk) continue;

        setHistory((prev) => {
          const updated = [...prev];
          if (!updated[aiIndex]) return updated;
          updated[aiIndex] = {
            ...updated[aiIndex],
            content: (updated[aiIndex].content || "") + chunk,
          };
          return updated;
        });
      }
    } catch (e) {
      setHistory((prev) => [
        ...prev,
        { role: "assistant", content: `Network error: ${e}` },
      ]);
    }
  }

  function handleClear() {
    setHistory([]);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function resetSystemPrompt() {
    const base =
      "You are an advanced assistant owned by the user. Be direct, helpful, and honest. Avoid anything illegal or genuinely harmful.";
    setSystemPrompt(base);
  }

  async function handleGenerateImage() {
    const trimmed = imagePrompt.trim();
    if (!trimmed) return;

    setImageStatus("Generating…");
    setImageUrl("");

    try {
      const res = await fetch(`${API_BASE}/generate-image`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: trimmed }),
      });

      if (!res.ok) {
        let errText = res.statusText;
        try {
          const err = await res.json();
          errText = err.detail || errText;
        } catch (_) {}
        setImageStatus(`Error: ${errText}`);
        return;
      }

      const data = await res.json();
      setImageUrl(data.url);
      setImageStatus("");
    } catch (e) {
      setImageStatus(`Network error: ${e}`);
    }
  }

  async function handleSearch() {
    const trimmed = searchQuery.trim();
    if (!trimmed) return;

    setSearchStatus("Searching…");
    setSearchResults([]);

    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmed, max_results: 5 }),
      });

      if (!res.ok) {
        let errText = res.statusText;
        try {
          const err = await res.json();
          errText = err.detail || errText;
        } catch (_) {}
        setSearchStatus(`Error: ${errText}`);
        return;
      }

      const data = await res.json();
      setSearchResults(data.results || []);
      setSearchStatus("");
    } catch (e) {
      setSearchStatus(`Network error: ${e}`);
    }
  }

  return (
    <div className="app-shell">
      <h1>Uncensored AI Chat</h1>
      <p className="subtitle">React frontend · FastAPI backend · Together + tools.</p>

      <div className="card">
        <div className="settings-row">
          <label>
            Persona:
            <select value={persona} onChange={(e) => setPersona(e.target.value)}>
              <option value="default">Default assistant</option>
              <option value="coder">Senior engineer</option>
              <option value="teacher">Tutor / explainer</option>
              <option value="creative">Creative partner</option>
            </select>
          </label>

          <label>
            Model:
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo">
                Llama 3.1 70B Turbo
              </option>
              <option value="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo">
                Llama 3.1 8B Turbo (cheaper)
              </option>
            </select>
          </label>

          <label>
            Temperature:
            <input
              type="number"
              min="0"
              max="1.5"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              style={{ width: 60 }}
            />
          </label>
        </div>

        <label style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
          System prompt (controls tone/behaviour):
          <textarea
            className="system-prompt"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="You are an advanced uncensored assistant owned by the user..."
          />
        </label>
        <div className="system-controls">
          <span>Saved locally in your browser only.</span>
          <button className="button-pill" onClick={resetSystemPrompt}>
            Reset
          </button>
        </div>

        <div className="chat-box" ref={chatRef}>
          {history.map((msg, idx) => {
            if (msg.role === "user") {
              return (
                <div key={idx} className="msg msg-user">
                  You: {msg.content}
                </div>
              );
            }
            if (msg.role === "assistant") {
              return (
                <div key={idx} className="msg msg-ai">
                  AI: {msg.content}
                </div>
              );
            }
            if (msg.role === "search") {
              return (
                <div key={idx} className="msg msg-search">
                  {msg.content}
                </div>
              );
            }
            return null;
          })}
        </div>

        <div className="chat-controls">
          <span className="status-text">{status}</span>
          <button className="button-pill" onClick={handleClear}>
            Clear chat
          </button>
        </div>

        <div className="input-row">
          <input
            type="text"
            placeholder="Type a message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isSending}
          />
          <button onClick={handleSend} disabled={isSending}>
            Send
          </button>
        </div>
        <div className="helper-row">
          <label>
            <input
              type="checkbox"
              checked={useSearch}
              onChange={(e) => setUseSearch(e.target.checked)}
            />
            Use web search for this message
          </label>
          <label>
            <input
              type="checkbox"
              checked={useStreaming}
              onChange={(e) => setUseStreaming(e.target.checked)}
            />
            Stream response
          </label>
        </div>
      </div>

      <div className="section">
        <h2>Image Generation</h2>
        <div className="image-section">
          <input
            type="text"
            placeholder="Describe an image to generate..."
            value={imagePrompt}
            onChange={(e) => setImagePrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleGenerateImage();
            }}
          />
          <div className="image-actions">
            <button onClick={handleGenerateImage}>Generate Image</button>
            <span className="status-text">{imageStatus}</span>
          </div>
          <div className="image-result">
            {imageUrl && <img src={imageUrl} alt={imagePrompt} />}
          </div>
        </div>
      </div>

      <div className="section">
        <h2>Web Search Tool</h2>
        <div className="search-section">
          <input
            type="text"
            placeholder="Search the web..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSearch();
            }}
          />
          <div className="search-actions">
            <button onClick={handleSearch}>Search</button>
            <span className="status-text">{searchStatus}</span>
          </div>
          <div style={{ fontSize: "0.9rem" }}>
            {searchResults.length > 0 && (
              <ul>
                {searchResults.map((r, idx) => (
                  <li key={idx}>{r}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}


