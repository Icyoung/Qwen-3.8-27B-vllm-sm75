"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.callModel = callModel;
exports.createInitialMessages = createInitialMessages;
const benchmark_1 = require("../lib/benchmark");
const DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 30;
function normalizeBaseUrl(baseUrl) {
    return baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl;
}
function normalizeContent(content) {
    if (typeof content === "string") {
        return content;
    }
    if (Array.isArray(content)) {
        return content
            .map((part) => (part?.type === "text" ? part.text ?? "" : ""))
            .join("")
            .trim();
    }
    return "";
}
function normalizeToolCalls(message) {
    return (message?.tool_calls?.map((call, index) => ({
        id: call.id ?? `tool_call_${index + 1}`,
        type: "function",
        function: {
            name: call.function?.name ?? "unknown_tool",
            arguments: typeof call.function?.arguments === "string"
                ? call.function.arguments
                : JSON.stringify(call.function?.arguments ?? {})
        }
    })) ?? []);
}
function toPythonValue(value) {
    if (typeof value === "string") {
        return `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
    }
    if (typeof value === "boolean") {
        return value ? "True" : "False";
    }
    if (value === null || value === undefined) {
        return "None";
    }
    if (typeof value === "number") {
        return String(value);
    }
    return JSON.stringify(value);
}
function serializeToolCallsToLfm(toolCalls) {
    const calls = toolCalls
        .map((toolCall) => {
        let args = {};
        try {
            const parsed = JSON.parse(toolCall.function.arguments || "{}");
            if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
                args = parsed;
            }
        }
        catch {
            args = {};
        }
        const argsString = Object.entries(args)
            .map(([key, value]) => `${key}=${toPythonValue(value)}`)
            .join(", ");
        return `${toolCall.function.name}(${argsString})`;
    })
        .join(", ");
    return `<|tool_call_start|>[${calls}]<|tool_call_end|>`;
}
function buildLfmMessages(messages) {
    const toolList = benchmark_1.UNIVERSAL_TOOLS.map((tool) => tool.function);
    const injection = `\n\nList of tools: ${JSON.stringify(toolList)}\n`;
    return messages.map((message) => {
        if (message.role === "system") {
            return { ...message, content: message.content + injection };
        }
        if (message.role === "assistant" && message.tool_calls && message.tool_calls.length > 0) {
            const lfmBlock = serializeToolCallsToLfm(message.tool_calls);
            return {
                role: "assistant",
                content: message.content ? `${message.content}\n${lfmBlock}` : lfmBlock
            };
        }
        return message;
    });
}
function splitTopLevel(input, delimiter) {
    const parts = [];
    let current = "";
    let inString = false;
    let stringChar = "";
    let parenDepth = 0;
    let bracketDepth = 0;
    let braceDepth = 0;
    for (let index = 0; index < input.length; index += 1) {
        const ch = input[index];
        if (inString) {
            if (ch === "\\" && index + 1 < input.length) {
                current += ch + input[index + 1];
                index += 1;
            }
            else if (ch === stringChar) {
                inString = false;
                current += ch;
            }
            else {
                current += ch;
            }
            continue;
        }
        if (ch === '"' || ch === "'") {
            inString = true;
            stringChar = ch;
            current += ch;
            continue;
        }
        if (ch === "(") {
            parenDepth += 1;
        }
        else if (ch === ")" && parenDepth > 0) {
            parenDepth -= 1;
        }
        else if (ch === "[") {
            bracketDepth += 1;
        }
        else if (ch === "]" && bracketDepth > 0) {
            bracketDepth -= 1;
        }
        else if (ch === "{") {
            braceDepth += 1;
        }
        else if (ch === "}" && braceDepth > 0) {
            braceDepth -= 1;
        }
        else if (ch === delimiter && parenDepth === 0 && bracketDepth === 0 && braceDepth === 0) {
            const part = current.trim();
            if (part) {
                parts.push(part);
            }
            current = "";
            continue;
        }
        current += ch;
    }
    if (current.trim()) {
        parts.push(current.trim());
    }
    return parts;
}
function findTopLevelChar(input, target) {
    let inString = false;
    let stringChar = "";
    let parenDepth = 0;
    let bracketDepth = 0;
    let braceDepth = 0;
    for (let index = 0; index < input.length; index += 1) {
        const ch = input[index];
        if (inString) {
            if (ch === "\\" && index + 1 < input.length) {
                index += 1;
            }
            else if (ch === stringChar) {
                inString = false;
            }
            continue;
        }
        if (ch === '"' || ch === "'") {
            inString = true;
            stringChar = ch;
            continue;
        }
        if (ch === "(") {
            parenDepth += 1;
            continue;
        }
        if (ch === ")" && parenDepth > 0) {
            parenDepth -= 1;
            continue;
        }
        if (ch === "[") {
            bracketDepth += 1;
            continue;
        }
        if (ch === "]" && bracketDepth > 0) {
            bracketDepth -= 1;
            continue;
        }
        if (ch === "{") {
            braceDepth += 1;
            continue;
        }
        if (ch === "}" && braceDepth > 0) {
            braceDepth -= 1;
            continue;
        }
        if (ch === target && parenDepth === 0 && bracketDepth === 0 && braceDepth === 0) {
            return index;
        }
    }
    return -1;
}
function parsePythonValue(raw) {
    const trimmed = raw.trim();
    if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
        return trimmed
            .slice(1, -1)
            .replace(/\\n/g, "\n")
            .replace(/\\t/g, "\t")
            .replace(/\\"/g, '"')
            .replace(/\\'/g, "'")
            .replace(/\\\\/g, "\\");
    }
    if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
        const inner = trimmed.slice(1, -1).trim();
        return inner ? splitTopLevel(inner, ",").map((part) => parsePythonValue(part)) : [];
    }
    if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
        const inner = trimmed.slice(1, -1).trim();
        const objectValue = {};
        if (!inner) {
            return objectValue;
        }
        for (const entry of splitTopLevel(inner, ",")) {
            const separatorIndex = findTopLevelChar(entry, ":");
            if (separatorIndex === -1) {
                continue;
            }
            const keyValue = parsePythonValue(entry.slice(0, separatorIndex).trim());
            const objectKey = typeof keyValue === "string" ? keyValue : String(keyValue);
            objectValue[objectKey] = parsePythonValue(entry.slice(separatorIndex + 1).trim());
        }
        return objectValue;
    }
    if (trimmed === "True")
        return true;
    if (trimmed === "False")
        return false;
    if (trimmed === "None")
        return null;
    const numberValue = Number(trimmed);
    if (!Number.isNaN(numberValue) && trimmed !== "") {
        return numberValue;
    }
    return trimmed;
}
function parsePythonicCalls(text) {
    const calls = [];
    let index = 0;
    while (index < text.length) {
        while (index < text.length && /[\s,]/u.test(text[index]))
            index += 1;
        if (index >= text.length)
            break;
        const nameStart = index;
        while (index < text.length && /\w/u.test(text[index]))
            index += 1;
        const name = text.slice(nameStart, index);
        if (!name) {
            index += 1;
            continue;
        }
        while (index < text.length && text[index] === " ")
            index += 1;
        if (index >= text.length || text[index] !== "(")
            continue;
        index += 1;
        const argsStart = index;
        let depth = 1;
        let inString = false;
        let stringChar = "";
        while (index < text.length && depth > 0) {
            const ch = text[index];
            if (inString) {
                if (ch === "\\" && index + 1 < text.length) {
                    index += 2;
                    continue;
                }
                if (ch === stringChar) {
                    inString = false;
                }
            }
            else if (ch === '"' || ch === "'") {
                inString = true;
                stringChar = ch;
            }
            else if (ch === "(") {
                depth += 1;
            }
            else if (ch === ")") {
                depth -= 1;
                if (depth === 0)
                    break;
            }
            index += 1;
        }
        const argsString = text.slice(argsStart, index);
        index += 1;
        const args = {};
        for (const part of splitTopLevel(argsString, ",")) {
            const separatorIndex = findTopLevelChar(part, "=");
            if (separatorIndex === -1)
                continue;
            const key = part.slice(0, separatorIndex).trim();
            if (key) {
                args[key] = parsePythonValue(part.slice(separatorIndex + 1).trim());
            }
        }
        calls.push({ name, args });
    }
    return calls;
}
function parseLfmResponse(content) {
    const toolCalls = [];
    let callIndex = 0;
    const blocks = content.match(/<\|tool_call_start\|>([\s\S]*?)<\|tool_call_end\|>/g) ?? [];
    for (const block of blocks) {
        const inner = block.replace(/<\|tool_call_start\|>/, "").replace(/<\|tool_call_end\|>/, "").trim();
        try {
            const parsed = JSON.parse(inner);
            for (const call of Array.isArray(parsed) ? parsed : [parsed]) {
                if (call?.name) {
                    toolCalls.push({
                        id: `tool_call_${++callIndex}`,
                        type: "function",
                        function: {
                            name: call.name,
                            arguments: typeof call.arguments === "string" ? call.arguments : JSON.stringify(call.arguments ?? {})
                        }
                    });
                }
            }
        }
        catch {
            // Some LFM-compatible servers emit pythonic calls instead of JSON.
        }
    }
    if (toolCalls.length > 0) {
        return {
            content: content.replace(/<\|tool_call_start\|>[\s\S]*?<\|tool_call_end\|>/g, "").trim(),
            toolCalls
        };
    }
    const trimmed = content.trim();
    if (trimmed.startsWith("[")) {
        try {
            const parsed = JSON.parse(trimmed);
            for (const call of Array.isArray(parsed) ? parsed : []) {
                if (call?.name) {
                    toolCalls.push({
                        id: `tool_call_${++callIndex}`,
                        type: "function",
                        function: {
                            name: call.name,
                            arguments: typeof call.arguments === "string" ? call.arguments : JSON.stringify(call.arguments ?? {})
                        }
                    });
                }
            }
            if (toolCalls.length > 0) {
                return { content: "", toolCalls };
            }
        }
        catch {
            const inner = trimmed.slice(1, -1).trim();
            for (const call of parsePythonicCalls(inner)) {
                toolCalls.push({
                    id: `tool_call_${++callIndex}`,
                    type: "function",
                    function: { name: call.name, arguments: JSON.stringify(call.args) }
                });
            }
            if (toolCalls.length > 0) {
                return { content: "", toolCalls };
            }
        }
    }
    return { content, toolCalls };
}
function isTimeoutError(error) {
    return error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError");
}
function isAbortError(error) {
    return error instanceof Error && error.name === "AbortError";
}
function resolveRequestTimeoutMs(params) {
    if (params?.request_timeout_seconds !== undefined) {
        const requested = Math.trunc(params.request_timeout_seconds);
        if (Number.isFinite(requested) && requested > 0) {
            return requested * 1000;
        }
    }
    const rawTimeout = process.env.MODEL_REQUEST_TIMEOUT_SECONDS?.trim();
    if (!rawTimeout) {
        return DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS * 1000;
    }
    const parsed = Number.parseInt(rawTimeout, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        return DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS * 1000;
    }
    return parsed * 1000;
}
function scenarioSystemPrompt(scenario) {
    return [benchmark_1.BASE_SYSTEM_PROMPT, scenario.systemPrompt].filter(Boolean).join("\n\nScenario-specific instructions:\n");
}
async function callModel(model, messages, params) {
    const baseUrl = normalizeBaseUrl(model.baseUrl);
    const requestTimeoutMs = resolveRequestTimeoutMs(params);
    const headers = {
        "Content-Type": "application/json"
    };
    if (model.apiKey) {
        headers.Authorization = `Bearer ${model.apiKey}`;
    }
    const useLfmFormat = (params?.tools_format ?? "default") === "lfm";
    const body = {
        model: model.model,
        messages: useLfmFormat ? buildLfmMessages(messages) : messages,
        ...(useLfmFormat ? {} : { parallel_tool_calls: true, tool_choice: "auto", tools: benchmark_1.UNIVERSAL_TOOLS })
    };
    if (params?.temperature !== undefined)
        body.temperature = params.temperature;
    if (params?.top_p !== undefined)
        body.top_p = params.top_p;
    if (params?.top_k !== undefined)
        body.top_k = params.top_k;
    if (params?.min_p !== undefined)
        body.min_p = params.min_p;
    if (params?.max_tokens !== undefined)
        body.max_tokens = params.max_tokens;
    if (params?.seed !== undefined)
        body.seed = params.seed;
    if (params?.stop !== undefined)
        body.stop = params.stop;
    if (params?.repetition_penalty !== undefined)
        body.repetition_penalty = params.repetition_penalty;
    if (params?.presence_penalty !== undefined)
        body.presence_penalty = params.presence_penalty;
    if (params?.frequency_penalty !== undefined)
        body.frequency_penalty = params.frequency_penalty;
    if (params?.reasoning !== undefined)
        body.reasoning = params.reasoning;
    if (params?.provider_options !== undefined)
        body.provider_options = params.provider_options;
    if (params?.extra_body !== undefined) {
        Object.assign(body, params.extra_body);
    }
    const timeoutController = new AbortController();
    const timeoutHandle = setTimeout(() => timeoutController.abort(), requestTimeoutMs);
    const parentSignal = params?.signal;
    const abortController = new AbortController();
    const abortFromParent = () => abortController.abort(parentSignal?.reason);
    const abortFromTimeout = () => abortController.abort(new DOMException("Request timed out.", "TimeoutError"));
    parentSignal?.addEventListener("abort", abortFromParent, { once: true });
    timeoutController.signal.addEventListener("abort", abortFromTimeout, { once: true });
    let response;
    let payload;
    try {
        response = await fetch(`${baseUrl}/chat/completions`, {
            method: "POST",
            headers,
            body: JSON.stringify(body),
            signal: abortController.signal
        });
        payload = (await response.json());
    }
    catch (error) {
        if (isAbortError(error) && parentSignal?.aborted) {
            throw new Error("Request aborted.");
        }
        if (isTimeoutError(error)) {
            throw new Error(`Request timed out after ${requestTimeoutMs / 1000}s.`);
        }
        throw error;
    }
    finally {
        clearTimeout(timeoutHandle);
        parentSignal?.removeEventListener("abort", abortFromParent);
    }
    if (!response.ok) {
        throw new Error(payload.error?.message || `Provider request failed with ${response.status}.`);
    }
    const message = payload.choices?.[0]?.message;
    if (!message) {
        throw new Error("Provider returned no assistant message.");
    }
    if (useLfmFormat) {
        const parsed = parseLfmResponse(normalizeContent(message.content));
        if (parsed.toolCalls.length === 0 && (message.tool_calls?.length ?? 0) > 0) {
            return { content: parsed.content, toolCalls: normalizeToolCalls(message) };
        }
        return { content: parsed.content, toolCalls: parsed.toolCalls };
    }
    return {
        content: normalizeContent(message.content),
        toolCalls: normalizeToolCalls(message),
        reasoning: message.reasoning_content ?? message.reasoning,
        reasoningContent: message.reasoning_content
    };
}
function createInitialMessages(scenario) {
    return [
        { role: "system", content: scenarioSystemPrompt(scenario) },
        { role: "user", content: scenario.userMessage }
    ];
}
