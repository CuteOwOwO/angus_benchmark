import { Behavior } from "@google/genai";
import {
  getMultiStepTaskVariant,
  getTwoStepTaskVariants,
  listMultiStepTaskVariantIds,
  type MultiStepTaskVariant,
  type ToolDeclarationSpec,
} from "./multi-step-task-catalog.js";

export type { ToolDeclarationSpec };

export type TwoStepTaskSpec = {
  id: string;
  resultSlug: string;
  promptName: string;
  systemInstruction: string;
  userPrompt: string;
  tools: ToolDeclarationSpec[];
  step1: {
    label: string;
    toolName: string;
    mockedResult: Record<string, unknown>;
  };
  step2: {
    label: string;
    toolName: string;
    mockedResult: Record<string, unknown>;
    dependency: {
      argName: string;
      acceptedArgNames?: string[];
      fromStep1ResultField: string;
    };
  };
  expectedFinalAnswer: string;
  finalAnswerChecks: {
    mentionsTargetAnswer: RegExp;
    usesStep1Result: RegExp;
    usesStep2Result: RegExp;
    prematureAnswer: RegExp;
  };
};

function toTwoStepTaskSpec(variant: MultiStepTaskVariant): TwoStepTaskSpec {
  if (variant.stepCount !== 2) throw new Error(`Task variant is not two-step: ${variant.id}`);
  const [step1, step2] = variant.steps;
  if (!step2.dependency || step2.dependency.fromStepIndex !== 1) {
    throw new Error(`Two-step task ${variant.id} must have step2 depend on step1.`);
  }
  return {
    id: variant.id,
    resultSlug: variant.resultSlug,
    promptName: variant.promptName,
    systemInstruction: variant.systemInstruction,
    userPrompt: variant.userPrompt,
    tools: variant.tools,
    step1: {
      label: step1.label,
      toolName: step1.toolName,
      mockedResult: step1.mockedResult,
    },
    step2: {
      label: step2.label,
      toolName: step2.toolName,
      mockedResult: step2.mockedResult,
      dependency: {
        argName: step2.dependency.argName,
        acceptedArgNames: step2.dependency.acceptedArgNames,
        fromStep1ResultField: step2.dependency.fromResultField,
      },
    },
    expectedFinalAnswer: variant.expectedFinalAnswer,
    finalAnswerChecks: {
      mentionsTargetAnswer: variant.finalAnswerChecks.mentionsTargetAnswer,
      usesStep1Result: variant.finalAnswerChecks.usesStep1Result,
      usesStep2Result: variant.finalAnswerChecks.usesFinalStepResult,
      prematureAnswer: variant.finalAnswerChecks.prematureAnswer,
    },
  };
}

const TWO_STEP_TASKS: Record<string, TwoStepTaskSpec> = Object.fromEntries(
  getTwoStepTaskVariants().map((variant) => [variant.id, toTwoStepTaskSpec(variant)]),
);

// Backward-compatible alias for the original calendar-route task id.
TWO_STEP_TASKS.calendar_route_leave_time = TWO_STEP_TASKS.calendar_route_leave_time_2step;

export function listTwoStepTaskIds(): string[] {
  return Object.keys(TWO_STEP_TASKS).sort();
}

export function listAllMultiStepTaskVariantIds(): string[] {
  return listMultiStepTaskVariantIds();
}

export function getTwoStepTask(id: string): TwoStepTaskSpec {
  const task = TWO_STEP_TASKS[id];
  if (!task) {
    const maybeVariant = getMultiStepTaskVariant(id);
    if (maybeVariant.stepCount !== 2) {
      throw new Error(`Task ${id} has ${maybeVariant.stepCount} steps. The current runner only supports two-step variants.`);
    }
    return toTwoStepTaskSpec(maybeVariant);
  }
  return task;
}

export function makeToolDeclarations(task: TwoStepTaskSpec): unknown[] {
  return [
    {
      functionDeclarations: task.tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        behavior: Behavior.NON_BLOCKING,
        parametersJsonSchema: tool.parametersJsonSchema,
      })),
    },
  ];
}
