//! ACP v2 `session/update`s as the v1 updates the timeline folds.
//!
//! v2 reports tool calls as upserting `tool_call_update`s (plus appended
//! `tool_call_content_chunk`s), command output on display terminals, whole
//! messages next to their chunks, plans by id, and compaction as an entity of its
//! own. The normalizer keeps what it needs per session to turn each into zero or
//! more v1 updates: the tool calls it has announced and their content, the text
//! of each message so far, and each terminal's output.

use serde_json::{Map, Value, json};
use std::collections::{BTreeSet, HashMap};

/// Terminal output kept per terminal; longer output keeps its tail.
const MAX_TERMINAL: usize = 64 * 1024;

#[derive(Default)]
pub(crate) struct Normalizer {
    sessions: HashMap<String, SessionState>,
}

#[derive(Default)]
struct SessionState {
    /// v2 content of each tool call announced so far.
    tools: HashMap<String, Vec<Value>>,
    /// Text so far of each message, by `(chunk kind, messageId)`.
    messages: HashMap<(String, String), String>,
    terminals: HashMap<String, Terminal>,
    /// The plan shown as the v1 `plan`.
    plan: Option<String>,
    /// Summary so far of each compaction.
    compactions: HashMap<String, String>,
}

#[derive(Default)]
struct Terminal {
    output: String,
    /// Tool calls whose content shows this terminal.
    tools: BTreeSet<String>,
}

impl Normalizer {
    /// The v1 updates for v2 update `update` of session `session`.
    pub fn normalize(&mut self, session: &str, update: &Value) -> Vec<Value> {
        let kind = update.get("sessionUpdate").and_then(Value::as_str).unwrap_or_default();
        let s = self.sessions.entry(session.to_owned()).or_default();
        match kind {
            // The echo of a prompt: the client shows its own.
            "user_message" => Vec::new(),
            "agent_message_chunk" | "agent_thought_chunk" | "user_message_chunk" => {
                if let (Some(id), Some(text)) = (str_of(update, "messageId"), update.pointer("/content/text"))
                    && let Some(text) = text.as_str()
                {
                    s.messages.entry((kind.to_owned(), id.to_owned())).or_default().push_str(text);
                }
                vec![update.clone()]
            }
            "agent_message" | "agent_thought" => {
                let chunk = format!("{kind}_chunk");
                let id = str_of(update, "messageId").unwrap_or_default().to_owned();
                let text = message_text(update.get("content"));
                let seen = s.messages.entry((chunk.clone(), id)).or_default();
                // A message replaces what its chunks said; the timeline only appends, so
                // what the chunks did not say yet goes as one more chunk.
                let new = match text.strip_prefix(seen.as_str()) {
                    Some(rest) => rest.to_owned(),
                    None => text.clone(),
                };
                *seen = text;
                if new.is_empty() {
                    return Vec::new();
                }
                vec![json!({ "sessionUpdate": chunk, "content": { "type": "text", "text": new } })]
            }
            "tool_call_update" => {
                let Some(id) = str_of(update, "toolCallId").map(str::to_owned) else { return Vec::new() };
                let first = !s.tools.contains_key(&id);
                let content = s.tools.entry(id.clone()).or_default();
                let mut out = Map::new();
                for (k, v) in update.as_object().into_iter().flatten() {
                    // `null` clears a field in v2; the timeline keeps what it has.
                    if !v.is_null() && k != "content" {
                        out.insert(k.clone(), v.clone());
                    }
                }
                let has_content = update.get("content").is_some();
                if has_content {
                    *content = update.get("content").and_then(Value::as_array).cloned().unwrap_or_default();
                }
                if first {
                    out.insert("sessionUpdate".into(), json!("tool_call"));
                    if !out.contains_key("title") {
                        out.insert("title".into(), json!(str_of(update, "name").unwrap_or_default()));
                    }
                }
                if has_content || first {
                    s.link_terminals(&id);
                    out.insert("content".into(), Value::Array(s.v1_content(&id)));
                }
                vec![Value::Object(out)]
            }
            "tool_call_content_chunk" => {
                let (Some(id), Some(item)) = (str_of(update, "toolCallId"), update.get("content")) else {
                    return Vec::new();
                };
                let first = !s.tools.contains_key(id);
                s.tools.entry(id.to_owned()).or_default().push(item.clone());
                s.link_terminals(id);
                let mut out =
                    json!({ "sessionUpdate": "tool_call_update", "toolCallId": id, "content": s.v1_content(id) });
                if first {
                    out["sessionUpdate"] = json!("tool_call");
                    out["title"] = json!("");
                }
                vec![out]
            }
            "terminal_update" | "terminal_output_chunk" => {
                let Some(id) = str_of(update, "terminalId") else { return Vec::new() };
                let terminal = s.terminals.entry(id.to_owned()).or_default();
                let data = match kind {
                    "terminal_update" => update.pointer("/output/data"),
                    _ => update.get("data"),
                };
                let Some(data) = data.and_then(Value::as_str) else { return Vec::new() };
                let text = String::from_utf8_lossy(&base64(data)).into_owned();
                if kind == "terminal_update" {
                    terminal.output = text;
                } else {
                    terminal.output.push_str(&text);
                }
                tail(&mut terminal.output);
                let tools: Vec<String> = terminal.tools.iter().cloned().collect();
                tools
                    .into_iter()
                    .map(|t| json!({ "sessionUpdate": "tool_call_update", "toolCallId": t, "content": s.v1_content(&t) }))
                    .collect()
            }
            "plan_update" => {
                let Some(plan) = update.get("plan") else { return Vec::new() };
                // Markdown and file plans have no v1 form; the tool call that made them shows them.
                if plan.get("type").and_then(Value::as_str) != Some("items") {
                    return Vec::new();
                }
                s.plan = str_of(plan, "planId").map(str::to_owned);
                let entries = plan.get("entries").cloned().unwrap_or_else(|| json!([]));
                vec![json!({ "sessionUpdate": "plan", "entries": entries })]
            }
            "plan_removed" => {
                if s.plan.is_none() || s.plan.as_deref() != str_of(update, "planId") {
                    return Vec::new();
                }
                s.plan = None;
                vec![json!({ "sessionUpdate": "plan", "entries": [] })]
            }
            // The agent compacting its context, as v1 agents report it: a tool call.
            "compaction_update" | "compaction_summary_chunk" => {
                let Some(id) = str_of(update, "compactionId") else { return Vec::new() };
                let tool = format!("compaction:{id}");
                let first = !s.compactions.contains_key(id);
                let summary = s.compactions.entry(id.to_owned()).or_default();
                let mut out = json!({ "sessionUpdate": "tool_call_update", "toolCallId": tool });
                if kind == "compaction_summary_chunk" {
                    if let Some(text) = update.pointer("/content/text").and_then(Value::as_str) {
                        summary.push_str(text);
                    }
                } else {
                    if let Some(blocks) = update.get("summary").filter(|v| v.is_array()) {
                        *summary = message_text(Some(blocks));
                    }
                    let status = match str_of(update, "status") {
                        Some("completed") => "completed",
                        Some("failed" | "cancelled") => "failed",
                        _ => "in_progress",
                    };
                    out["status"] = json!(status);
                    if let Some(error) = str_of(update, "error") {
                        summary.push_str(error);
                    }
                }
                if !summary.is_empty() {
                    out["content"] = json!([{ "type": "content", "content": { "type": "text", "text": summary } }]);
                }
                if first {
                    out["sessionUpdate"] = json!("tool_call");
                    out["title"] = json!("Compact conversation");
                    out["kind"] = json!("other");
                }
                vec![out]
            }
            // The turn's state: what ends a prompt (see `Dialect`), not part of the conversation.
            "state_update" => Vec::new(),
            // Same shape in both: config options (`configId`), commands (`input.type`),
            // titles, usage, subagents, session messages, notices; and what v1 doesn't know.
            _ => vec![update.clone()],
        }
    }
}

impl SessionState {
    /// Notes which terminals tool call `id` shows.
    fn link_terminals(&mut self, id: &str) {
        let content = self.tools.get(id).map(Vec::as_slice).unwrap_or_default();
        for item in content {
            if item.get("type").and_then(Value::as_str) == Some("terminal")
                && let Some(t) = str_of(item, "terminalId")
            {
                self.terminals.entry(t.to_owned()).or_default().tools.insert(id.to_owned());
            }
        }
    }

    /// Tool call `id`'s content in v1 form: a terminal as its output so far, a diff per changed file.
    fn v1_content(&self, id: &str) -> Vec<Value> {
        let content = self.tools.get(id).map(Vec::as_slice).unwrap_or_default();
        v1_content(content, &|t| self.terminals.get(t).map(|t| t.output.as_str()))
    }
}

/// v2 tool call content in v1 form. `terminal` gives a terminal's output so far.
pub(crate) fn v1_content<'a>(content: &[Value], terminal: &dyn Fn(&str) -> Option<&'a str>) -> Vec<Value> {
    let mut out = Vec::new();
    for item in content {
        match item.get("type").and_then(Value::as_str) {
            Some("terminal") => {
                if let Some(text) = str_of(item, "terminalId").and_then(terminal).filter(|t| !t.is_empty()) {
                    out.push(json!({ "type": "content", "content": { "type": "text", "text": text } }));
                }
            }
            Some("diff") if item.get("changes").is_some() => out.extend(v1_diffs(item)),
            _ => out.push(item.clone()),
        }
    }
    out
}

/// A v2 diff (`changes` and an optional git patch) as v1 diffs, one per changed file. v1
/// diffs carry texts: each is the patch's hunks for the file, context plus removed lines
/// before and context plus added lines after, so diffing them counts the patch's lines.
fn v1_diffs(diff: &Value) -> Vec<Value> {
    let patch = diff.pointer("/patch/text").and_then(Value::as_str).unwrap_or_default();
    let files = patch_files(patch);
    diff.get("changes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|change| {
            let path = str_of(change, "path")?;
            // Patch paths can be relative to the repository.
            let same = |p: &str| !p.is_empty() && (p == path || path.ends_with(&format!("/{p}")));
            let (old, new) =
                files.iter().find(|f| same(&f.0)).map(|f| (f.1.as_str(), f.2.as_str())).unwrap_or_default();
            Some(json!({ "type": "diff", "path": path, "oldText": old, "newText": new }))
        })
        .collect()
}

/// `(path, old text, new text)` of each file in a git patch, from its hunks.
fn patch_files(patch: &str) -> Vec<(String, String, String)> {
    let mut files: Vec<(String, String, String)> = Vec::new();
    let mut in_hunk = false;
    for line in patch.lines() {
        if line.starts_with("diff --git ") {
            files.push((String::new(), String::new(), String::new()));
            in_hunk = false;
            continue;
        }
        if !in_hunk && let Some(path) = line.strip_prefix("+++ ").or_else(|| line.strip_prefix("--- ")) {
            let path = path.trim_end();
            let path = path.strip_prefix("b/").or_else(|| path.strip_prefix("a/")).unwrap_or(path);
            if files.is_empty() {
                files.push(Default::default());
            }
            if path != "/dev/null"
                && let Some(f) = files.last_mut()
            {
                f.0 = path.to_owned();
            }
            continue;
        }
        if line.starts_with("@@") {
            in_hunk = true;
            continue;
        }
        let Some(f) = files.last_mut().filter(|_| in_hunk) else { continue };
        let (old, new) = (&mut f.1, &mut f.2);
        match line.as_bytes().first() {
            Some(b'-') => push_line(old, &line[1..]),
            Some(b'+') => push_line(new, &line[1..]),
            Some(b' ') => {
                push_line(old, &line[1..]);
                push_line(new, &line[1..]);
            }
            _ => {}
        }
    }
    files
}

fn push_line(text: &mut String, line: &str) {
    text.push_str(line);
    text.push('\n');
}

/// The text of a message's content blocks.
fn message_text(content: Option<&Value>) -> String {
    let blocks = content.and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
    blocks.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).collect()
}

fn str_of<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

fn tail(text: &mut String) {
    if text.len() > MAX_TERMINAL {
        let mut cut = text.len() - MAX_TERMINAL;
        while !text.is_char_boundary(cut) {
            cut += 1;
        }
        text.replace_range(..cut, "");
    }
}

/// Standard base64 (padding optional); invalid characters are skipped.
pub(crate) fn base64(data: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for b in data.bytes() {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => continue,
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    out
}
