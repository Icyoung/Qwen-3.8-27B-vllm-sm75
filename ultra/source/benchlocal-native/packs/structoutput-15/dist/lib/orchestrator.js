"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.runScenarioForModel = runScenarioForModel;
exports.runBenchmark = runBenchmark;
const benchmark_1 = require("../lib/benchmark");
const llm_client_1 = require("../lib/llm-client");
const sandbox_client_1 = require("../lib/sandbox-client");
const PROVIDER_ERROR_RETRY_PATTERN = /provider returned error/i;
const TIMEOUT_RETRY_PATTERN = /request timed out|aborted due to timeout|timeouterror|aborterror/i;
const MAX_MODEL_ATTEMPTS = 3;
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
function isRetryableModelError(error) {
    return PROVIDER_ERROR_RETRY_PATTERN.test(error.message) || TIMEOUT_RETRY_PATTERN.test(error.message);
}
function formatScenarioTrace(model, scenario, evaluation, traceLines) {
    return [
        `model=${model.model}`,
        `scenario=${scenario.id} ${scenario.title}`,
        "",
        ...traceLines,
        "",
        `verdict=${evaluation.status}`,
        evaluation.score !== undefined ? `score=${evaluation.score}` : "",
        `summary=${evaluation.summary}`,
        evaluation.note ? `note=${evaluation.note}` : ""
    ]
        .filter(Boolean)
        .join("\n");
}
async function callModelWithRetry(model, userMessage, params, traceLines, emit, scenarioId) {
    let response = null;
    let lastError = null;
    for (let attempt = 1; attempt <= MAX_MODEL_ATTEMPTS; attempt += 1) {
        try {
            response = await (0, llm_client_1.callModel)(model, (0, llm_client_1.createInitialMessages)(userMessage), params);
            lastError = null;
            break;
        }
        catch (error) {
            lastError = error instanceof Error ? error : new Error("Unknown model execution error.");
            if (!isRetryableModelError(lastError) || attempt === MAX_MODEL_ATTEMPTS) {
                throw lastError;
            }
            traceLines.push(`retry_attempt_${attempt}=${lastError.message}`);
            const isTimeout = TIMEOUT_RETRY_PATTERN.test(lastError.message);
            await emit({
                type: "model_progress",
                modelId: model.id,
                scenarioId,
                message: isTimeout
                    ? `Model request timed out, retrying (${attempt + 1}/${MAX_MODEL_ATTEMPTS})`
                    : `Provider returned error, retrying (${attempt + 1}/${MAX_MODEL_ATTEMPTS})`
            });
            await sleep(750 * attempt, params?.signal);
        }
    }
    if (!response) {
        throw lastError ?? new Error("Unknown model execution error.");
    }
    return response;
}
function isCancellationError(error) {
    return error instanceof Error && /request aborted|validator request aborted/i.test(error.message);
}
async function runScenarioForModel(model, scenario, emit, params, validationOptions) {
    const state = {
        assistantMessages: [],
        finalAnswer: "",
        meta: {}
    };
    const traceLines = [`user_turn_1=${scenario.userMessage}`];
    await emit({
        type: "model_progress",
        modelId: model.id,
        scenarioId: scenario.id,
        message: "Calling model"
    });
    try {
        const response = await callModelWithRetry(model, scenario.userMessage, params, traceLines, emit, scenario.id);
        const content = response.content || "[empty response]";
        state.assistantMessages.push(content);
        state.finalAnswer = content;
        traceLines.push(`assistant_turn_1=${content}`);
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
            score: 0,
            summary,
            rawLog: formatScenarioTrace(model, scenario, { status: "fail", score: 0, summary }, traceLines)
        };
    }
    traceLines.push(`final_answer=${state.finalAnswer}`);
    await emit({
        type: "model_progress",
        modelId: model.id,
        scenarioId: scenario.id,
        message: "Validating artifact in container"
    });
    const validation = await (0, sandbox_client_1.verifyAnswerInSandbox)(scenario.id, state.finalAnswer, {
        ...validationOptions,
        signal: params?.signal
    });
    state.meta.validationResult = validation;
    traceLines.push(`validator_status=${validation.status}`);
    traceLines.push(`validator_summary=${validation.summary}`);
    const evaluation = scenario.evaluate(state);
    return {
        scenarioId: scenario.id,
        status: evaluation.status,
        score: evaluation.score,
        summary: evaluation.summary,
        note: evaluation.note,
        rawLog: formatScenarioTrace(model, scenario, evaluation, traceLines)
    };
}
async function runBenchmark(models, emit, requestedScenarioIds, params, validationOptions) {
    const scenarios = resolveScenarios(requestedScenarioIds);
    const resultsByModel = Object.fromEntries(models.map((model) => [model.id, []]));
    await emit({
        type: "run_started",
        models: models.map((model) => ({ id: model.id, label: model.label })),
        totalScenarios: scenarios.length
    });
    try {
        const runScenario = async (model, scenario) => {
            const result = await runScenarioForModel(model, scenario, emit, params, validationOptions);
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
