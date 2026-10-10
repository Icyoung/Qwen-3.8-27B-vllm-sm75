"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.runScenarioForModel = runScenarioForModel;
exports.runBenchmark = runBenchmark;
const benchmark_1 = require("../lib/benchmark");
const llm_client_1 = require("../lib/llm-client");
const PROVIDER_ERROR_RETRY_PATTERN = /provider returned error/i;
const MAX_PROVIDER_ERROR_ATTEMPTS = 3;
function resolveScenarios(requestedScenarioIds) {
    if (!requestedScenarioIds || requestedScenarioIds.length === 0) {
        return benchmark_1.SCENARIOS;
    }
    const requested = new Set(requestedScenarioIds);
    const selected = benchmark_1.SCENARIOS.filter((scenario) => requested.has(scenario.id));
    if (selected.length !== requested.size) {
        const found = new Set(selected.map((scenario) => scenario.id));
        const missing = requestedScenarioIds.filter((scenarioId) => !found.has(scenarioId));
        throw new Error(`Unknown scenario id${missing.length > 1 ? "s" : ""}: ${missing.join(", ")}`);
    }
    return selected;
}
function sleep(milliseconds, signal) {
    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            signal?.removeEventListener("abort", abort);
            resolve();
        }, milliseconds);
        const abort = () => {
            clearTimeout(timeout);
            reject(new Error("Request aborted."));
        };
        if (signal) {
            signal.addEventListener("abort", abort, { once: true });
        }
    });
}
function parseToolArguments(rawArguments) {
    try {
        const parsed = JSON.parse(rawArguments);
        return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed) ? parsed : {};
    }
    catch {
        return {};
    }
}
function stringifyToolResult(result) {
    return JSON.stringify(result);
}
function toAssistantMessage(response) {
    return {
        role: "assistant",
        content: response.content,
        ...(response.reasoningContent !== undefined ? { reasoning_content: response.reasoningContent } : {}),
        tool_calls: response.toolCalls
    };
}
function formatScenarioTrace(model, scenario, evaluation, traceLines) {
    return [
        `model=${model.model}`,
        `scenario=${scenario.id} ${scenario.title}`,
        `prompt=${scenario.userMessage}`,
        "",
        ...traceLines,
        "",
        `verdict=${evaluation.status}`,
        `summary=${evaluation.summary}`,
        evaluation.note ? `note=${evaluation.note}` : ""
    ]
        .filter(Boolean)
        .join("\n");
}
function isCancellationError(error) {
    return error instanceof Error && /request aborted/i.test(error.message);
}
async function runScenarioForModel(model, scenario, emit, params) {
    const state = {
        toolCalls: [],
        toolResults: [],
        assistantMessages: [],
        finalAnswer: "",
        meta: {}
    };
    const messages = (0, llm_client_1.createInitialMessages)(scenario.userMessage);
    const maxTurns = 8;
    const traceLines = ["assistant=starting"];
    await emit({
        type: "model_progress",
        modelId: model.id,
        scenarioId: scenario.id,
        message: "Calling model"
    });
    try {
        for (let turn = 1; turn <= maxTurns; turn += 1) {
            let response = null;
            let lastError = null;
            for (let attempt = 1; attempt <= MAX_PROVIDER_ERROR_ATTEMPTS; attempt += 1) {
                try {
                    response = await (0, llm_client_1.callModel)(model, messages, params);
                    lastError = null;
                    break;
                }
                catch (error) {
                    lastError = error instanceof Error ? error : new Error("Unknown model execution error.");
                    if (!PROVIDER_ERROR_RETRY_PATTERN.test(lastError.message) || attempt === MAX_PROVIDER_ERROR_ATTEMPTS) {
                        throw lastError;
                    }
                    traceLines.push(`retry_attempt_${attempt}=${lastError.message}`);
                    await emit({
                        type: "model_progress",
                        modelId: model.id,
                        scenarioId: scenario.id,
                        message: `Provider returned error, retrying (${attempt + 1}/${MAX_PROVIDER_ERROR_ATTEMPTS})`
                    });
                    await sleep(750 * attempt, params?.signal);
                }
            }
            if (!response) {
                throw lastError ?? new Error("Unknown model execution error.");
            }
            state.assistantMessages.push(response.content);
            messages.push(toAssistantMessage(response));
            traceLines.push(`assistant_turn_${turn}=${response.content || "[tool_calls_only]"}`);
            if (response.reasoning) {
                traceLines.push(`assistant_reasoning_${turn}=${response.reasoning}`);
            }
            if (response.toolCalls.length === 0) {
                state.finalAnswer = response.content;
                break;
            }
            await emit({
                type: "model_progress",
                modelId: model.id,
                scenarioId: scenario.id,
                message: `Requested ${response.toolCalls.map((call) => call.function.name).join(", ")}`
            });
            for (const toolCall of response.toolCalls) {
                const record = {
                    id: toolCall.id,
                    name: toolCall.function.name,
                    rawArguments: toolCall.function.arguments,
                    arguments: parseToolArguments(toolCall.function.arguments),
                    turn
                };
                state.toolCalls.push(record);
                traceLines.push(`tool_call=${record.name} ${record.rawArguments}`);
                const result = await scenario.handleToolCall(state, record);
                state.toolResults.push({
                    callId: record.id,
                    name: record.name,
                    result
                });
                traceLines.push(`tool_result=${JSON.stringify(result)}`);
                messages.push({
                    role: "tool",
                    tool_call_id: toolCall.id,
                    content: stringifyToolResult(result)
                });
            }
        }
    }
    catch (error) {
        if (isCancellationError(error)) {
            throw error;
        }
        const summary = error instanceof Error ? error.message : "Unknown model execution error.";
        traceLines.push(`error=${summary}`);
        return {
            scenarioId: scenario.id,
            status: "fail",
            points: 0,
            summary,
            rawLog: formatScenarioTrace(model, scenario, { status: "fail", summary }, traceLines)
        };
    }
    if (!state.finalAnswer) {
        state.finalAnswer = state.assistantMessages.at(-1) ?? "Model did not return a final answer.";
    }
    traceLines.push(`final_answer=${state.finalAnswer}`);
    const evaluation = scenario.evaluate(state);
    return {
        scenarioId: scenario.id,
        status: evaluation.status,
        points: evaluation.points,
        summary: evaluation.summary,
        note: evaluation.note,
        rawLog: formatScenarioTrace(model, scenario, evaluation, traceLines)
    };
}
async function runBenchmark(models, emit, requestedScenarioIds, params) {
    const scenarios = resolveScenarios(requestedScenarioIds);
    const resultsByModel = Object.fromEntries(models.map((model) => [model.id, []]));
    await emit({
        type: "run_started",
        models: models.map((model) => ({ id: model.id, label: model.label })),
        totalScenarios: scenarios.length
    });
    try {
        const runScenario = async (model, scenario) => {
            const result = await runScenarioForModel(model, scenario, emit, params);
            return { modelId: model.id, scenarioId: scenario.id, result };
        };
        const emitResult = async (modelId, scenarioId, result) => {
            resultsByModel[modelId].push(result);
            await emit({ type: "scenario_result", modelId, scenarioId, result });
        };
        for (const [index, scenario] of scenarios.entries()) {
            await emit({
                type: "scenario_started",
                scenarioId: scenario.id,
                title: scenario.title,
                index: index + 1,
                total: scenarios.length
            });
            const results = await Promise.all(models.map((model) => runScenario(model, scenario)));
            for (const { modelId, scenarioId, result } of results) {
                await emitResult(modelId, scenarioId, result);
            }
            await emit({
                type: "scenario_finished",
                scenarioId: scenario.id
            });
        }
        const scores = Object.fromEntries(Object.entries(resultsByModel).map(([modelId, results]) => [modelId, (0, benchmark_1.scoreModelResults)(results)]));
        await emit({
            type: "run_finished",
            scores
        });
    }
    catch (error) {
        await emit({
            type: "run_error",
            message: error instanceof Error ? error.message : "Unknown benchmark error."
        });
    }
}
