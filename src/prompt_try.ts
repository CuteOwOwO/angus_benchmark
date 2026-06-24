export const PROMPT_VERSION = "single_final_answer_field_no_pre_speech_v1";

export const PROMPT = `
Please answer this question:

What is the final answer?

You must use external_answer_tool.

First, immediately call external_answer_tool exactly once.
Do not say anything to the user before the tool call request has been made.

While waiting for the tool result, you may give short waiting updates, but do not guess or mention any possible answer.

After the tool result arrives, answer exactly with the value of final_answer.
`;

export const SYSTEM_INSTRUCTION = `
You are a helpful voice assistant.

For this task, the final answer must come from external_answer_tool.

Your first action must be the tool call request.
Do not produce user-facing speech before the tool call request.

Before the tool result arrives, do not answer, guess, infer, or mention any possible answer.

After the tool result arrives, answer exactly with final_answer and nothing else.
`;
