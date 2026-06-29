import {
  MULTI_STEP_TASK_VARIANTS,
  getTwoStepTaskVariants,
  listMultiStepTaskVariantIds,
} from "./tasks/multi-step-task-catalog.js";
import { getTwoStepTask, listTwoStepTaskIds } from "./tasks/two-step-tasks.js";

function assert(condition: unknown, message: string): void {
  if (!condition) throw new Error(message);
}

function main(): void {
  const variants = MULTI_STEP_TASK_VARIANTS;
  const baseIds = Array.from(new Set(variants.map((variant) => variant.baseId))).sort();
  assert(baseIds.length === 10, `Expected 10 base tasks, got ${baseIds.length}`);
  assert(variants.length === 40, `Expected 40 variants, got ${variants.length}`);

  for (const baseId of baseIds) {
    const stepCounts = variants
      .filter((variant) => variant.baseId === baseId)
      .map((variant) => variant.stepCount)
      .sort((left, right) => left - right);
    assert(stepCounts.join(",") === "2,3,4,5", `Base task ${baseId} missing variants: ${stepCounts.join(",")}`);
  }

  for (const variant of variants) {
    assert(variant.id === `${variant.baseId}_${variant.stepCount}step`, `Unexpected variant id: ${variant.id}`);
    assert(variant.steps.length === variant.stepCount, `${variant.id} has ${variant.steps.length} steps`);
    assert(variant.tools.length === variant.stepCount, `${variant.id} has ${variant.tools.length} tools`);
    assert(variant.userPrompt.length > 20, `${variant.id} user prompt is too short`);
    assert(variant.expectedFinalAnswer.length > 5, `${variant.id} expected final answer is too short`);
    const toolNames = new Set(variant.tools.map((tool) => tool.name));
    for (let index = 0; index < variant.steps.length; index += 1) {
      const step = variant.steps[index];
      assert(toolNames.has(step.toolName), `${variant.id} step ${index + 1} tool missing declaration: ${step.toolName}`);
      assert(Object.keys(step.mockedResult).length > 0, `${variant.id} step ${index + 1} has empty mocked result`);
      if (index > 0) {
        assert(step.dependency, `${variant.id} step ${index + 1} missing dependency`);
        assert(step.dependency!.fromStepIndex >= 1, `${variant.id} step ${index + 1} dependency fromStepIndex too low`);
        assert(step.dependency!.fromStepIndex < index + 1, `${variant.id} step ${index + 1} dependency must point to an earlier step`);
        const sourceStep = variant.steps[step.dependency!.fromStepIndex - 1];
        assert(
          Object.prototype.hasOwnProperty.call(sourceStep.mockedResult, step.dependency!.fromResultField),
          `${variant.id} step ${index + 1} dependency field not found: ${step.dependency!.fromResultField}`,
        );
      }
    }
  }

  const twoStepVariants = getTwoStepTaskVariants();
  assert(twoStepVariants.length === 10, `Expected 10 two-step variants, got ${twoStepVariants.length}`);
  for (const variant of twoStepVariants) {
    const task = getTwoStepTask(variant.id);
    assert(task.step1.toolName === variant.steps[0].toolName, `${variant.id} step1 mismatch`);
    assert(task.step2.toolName === variant.steps[1].toolName, `${variant.id} step2 mismatch`);
  }
  assert(listTwoStepTaskIds().includes("calendar_route_leave_time"), "Missing backward-compatible calendar_route_leave_time alias");

  console.log(
    JSON.stringify(
      {
        ok: true,
        base_task_count: baseIds.length,
        variant_count: variants.length,
        two_step_variant_count: twoStepVariants.length,
        two_step_runner_task_count_including_alias: listTwoStepTaskIds().length,
        sample_task_ids: listMultiStepTaskVariantIds().slice(0, 8),
      },
      null,
      2,
    ),
  );
}

main();
