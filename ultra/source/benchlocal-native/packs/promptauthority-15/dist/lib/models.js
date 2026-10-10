"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getModelConfigGroups = getModelConfigGroups;
exports.getModelConfigs = getModelConfigs;
const PROVIDERS = new Set(["openrouter", "ollama", "llamacpp", "mlx", "lmstudio", "openai_compatible"]);
function normalizeHostBaseUrl(host, envName) {
    const trimmed = host.trim().replace(/\/+$/, "");
    if (!trimmed) {
        throw new Error(`${envName} is empty.`);
    }
    if (!/^https?:\/\//i.test(trimmed)) {
        throw new Error(`${envName} must start with http:// or https://.`);
    }
    const url = new URL(trimmed);
    const path = url.pathname.replace(/\/+$/, "");
    if (!path || path === "/") {
        url.pathname = "/v1";
        return url.toString().replace(/\/$/, "");
    }
    if (path.endsWith("/v1")) {
        url.pathname = path;
        return url.toString().replace(/\/$/, "");
    }
    if (path.endsWith("/api")) {
        url.pathname = `${path.slice(0, -4) || ""}/v1`;
        return url.toString().replace(/\/$/, "");
    }
    url.pathname = `${path}/v1`;
    return url.toString().replace(/\/$/, "");
}
function providerLabel(provider) {
    switch (provider) {
        case "openrouter":
            return "OpenRouter";
        case "ollama":
            return "Ollama";
        case "llamacpp":
            return "llama.cpp";
        case "mlx":
            return "mlx_lm";
        case "lmstudio":
            return "LM Studio";
        case "openai_compatible":
            return "OpenAI Compatible";
    }
}
function providerHostEnv(provider) {
    switch (provider) {
        case "ollama":
            return "OLLAMA_HOST";
        case "llamacpp":
            return "LLAMACPP_HOST";
        case "mlx":
            return "MLX_HOST";
        case "lmstudio":
            return "LMSTUDIO_HOST";
        case "openai_compatible":
            return "OPENAI_COMPATIBLE_HOST";
        case "openrouter":
            return "";
    }
}
function buildProviderBaseUrl(provider, envName) {
    if (provider === "openrouter") {
        return "https://openrouter.ai/api/v1";
    }
    const hostEnv = providerHostEnv(provider);
    const host = process.env[hostEnv]?.trim();
    if (!host) {
        throw new Error(`${hostEnv} is required when ${envName} includes a ${provider} model.`);
    }
    return normalizeHostBaseUrl(host, hostEnv);
}
function buildProviderApiKey(provider, envName) {
    if (provider === "openrouter") {
        const apiKey = process.env.OPENROUTER_API_KEY?.trim();
        if (!apiKey) {
            throw new Error(`OPENROUTER_API_KEY is required when ${envName} includes an openrouter model.`);
        }
        return apiKey;
    }
    if (provider === "openai_compatible") {
        return process.env.OPENAI_COMPATIBLE_API_KEY?.trim() || undefined;
    }
    return undefined;
}
function parseProvider(rawProvider, index, envName) {
    const normalized = rawProvider.trim().toLowerCase();
    if (!PROVIDERS.has(normalized)) {
        throw new Error(`${envName} entry ${index + 1} has unsupported provider "${rawProvider}". Use openrouter, ollama, llamacpp, mlx, lmstudio, or openai_compatible.`);
    }
    return normalized;
}
function parseModelEntry(entry, index, envName) {
    const trimmed = entry.trim();
    if (!trimmed) {
        throw new Error(`${envName} entry ${index + 1} is empty.`);
    }
    const separatorIndex = trimmed.indexOf(":");
    if (separatorIndex <= 0 || separatorIndex === trimmed.length - 1) {
        throw new Error(`${envName} entry ${index + 1} must use the format provider:model, for example openrouter:openai/gpt-4.1.`);
    }
    const provider = parseProvider(trimmed.slice(0, separatorIndex), index, envName);
    const model = trimmed.slice(separatorIndex + 1).trim();
    if (!model) {
        throw new Error(`${envName} entry ${index + 1} is missing the model name.`);
    }
    return {
        id: `${provider}:${model}`,
        label: `${model} via ${providerLabel(provider)}`,
        provider,
        model,
        baseUrl: buildProviderBaseUrl(provider, envName),
        apiKey: buildProviderApiKey(provider, envName)
    };
}
function parseModelConfigList(envName) {
    const raw = process.env[envName]?.trim() ?? "";
    if (!raw) {
        return [];
    }
    return raw
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean)
        .map((entry, index) => parseModelEntry(entry, index, envName));
}
function assertUniqueModelIds(models) {
    const seen = new Set();
    for (const model of models) {
        if (seen.has(model.id)) {
            throw new Error(`Duplicate model "${model.id}" found across LLM_MODELS and LLM_MODELS_2. Each configured provider:model must be unique.`);
        }
        seen.add(model.id);
    }
}
function getModelConfigGroups() {
    const primary = parseModelConfigList("LLM_MODELS");
    const secondary = parseModelConfigList("LLM_MODELS_2");
    const all = [...primary, ...secondary];
    assertUniqueModelIds(all);
    return {
        primary,
        secondary,
        all
    };
}
function getModelConfigs() {
    return getModelConfigGroups().all;
}
