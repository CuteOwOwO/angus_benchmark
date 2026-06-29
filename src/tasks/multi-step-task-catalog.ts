import { Behavior } from "@google/genai";

export type ToolDeclarationSpec = {
  name: string;
  description: string;
  parametersJsonSchema: Record<string, unknown>;
};

export type StepDependencySpec = {
  argName: string;
  acceptedArgNames?: string[];
  fromStepIndex: number;
  fromResultField: string;
};

export type MultiStepTaskStepSpec = {
  label: string;
  toolName: string;
  mockedResult: Record<string, unknown>;
  dependency?: StepDependencySpec;
};

export type MultiStepTaskVariant = {
  id: string;
  baseId: string;
  stepCount: 2 | 3 | 4 | 5;
  resultSlug: string;
  promptName: string;
  systemInstruction: string;
  userPrompt: string;
  tools: ToolDeclarationSpec[];
  steps: MultiStepTaskStepSpec[];
  expectedFinalAnswer: string;
  finalAnswerChecks: {
    mentionsTargetAnswer: RegExp;
    usesStep1Result: RegExp;
    usesFinalStepResult: RegExp;
    prematureAnswer: RegExp;
  };
};

type MultiStepTaskBase = {
  baseId: string;
  resultSlug: string;
  userPrompt: string;
  tools: ToolDeclarationSpec[];
  steps: [MultiStepTaskStepSpec, MultiStepTaskStepSpec, MultiStepTaskStepSpec, MultiStepTaskStepSpec, MultiStepTaskStepSpec];
  expectedFinalAnswers: Record<2 | 3 | 4 | 5, string>;
  finalAnswerChecks: MultiStepTaskVariant["finalAnswerChecks"];
};

export const MULTI_STEP_PROMPT_NAME = "tow-multistep-native-no-tick";

export const MULTI_STEP_SYSTEM_INSTRUCTION = `You are a helpful voice assistant.

This is a native tool-call setting, not a controlled stage-event setting.
Use the provided tools whenever the user asks for information that is not already known.
Do not invent tool results, intermediate facts, calculations, or final answers.

If a later tool requires information that the user did not provide, first obtain that missing information using an appropriate earlier tool.
Use earlier tool results as grounded inputs for later tool calls.
Before each required tool result arrives, do not guess, invent, or claim the concrete result of that step.

No_tick means the runtime will not send pending tick messages. It does not mean you must stay silent.
While a tool call is pending, you may give brief, natural, task-aware waiting responses.

Do not answer the final question until all relevant tool results are available.
The final answer must integrate the relevant tool results.`;

const emptySchema = {
  type: "object",
  properties: {},
};

function objectSchema(properties: Record<string, unknown>, required: string[] = []): Record<string, unknown> {
  return {
    type: "object",
    properties,
    ...(required.length ? { required } : {}),
  };
}

function stringProp(description: string): Record<string, unknown> {
  return { type: "string", description };
}

function numberProp(description: string): Record<string, unknown> {
  return { type: "number", description };
}

function dep(argName: string, fromStepIndex: number, fromResultField: string, acceptedArgNames?: string[]): StepDependencySpec {
  return { argName, acceptedArgNames, fromStepIndex, fromResultField };
}

function variants(base: MultiStepTaskBase): MultiStepTaskVariant[] {
  return ([2, 3, 4, 5] as const).map((stepCount) => ({
    id: `${base.baseId}_${stepCount}step`,
    baseId: base.baseId,
    stepCount,
    resultSlug: `${base.resultSlug}_${stepCount}step`,
    promptName: MULTI_STEP_PROMPT_NAME,
    systemInstruction: MULTI_STEP_SYSTEM_INSTRUCTION,
    userPrompt: base.userPrompt,
    tools: base.tools.slice(0, stepCount),
    steps: base.steps.slice(0, stepCount),
    expectedFinalAnswer: base.expectedFinalAnswers[stepCount],
    finalAnswerChecks: base.finalAnswerChecks,
  }));
}

const BASE_TASKS: MultiStepTaskBase[] = [
  {
    baseId: "calendar_route_leave_time",
    resultSlug: "calendar_route_two_step_no_tick",
    userPrompt: "Can you check when I should leave for my next meeting this afternoon? I want to arrive about 10 minutes early.",
    tools: [
      { name: "get_next_calendar_event", description: "Retrieve the user's next calendar event this afternoon.", parametersJsonSchema: emptySchema },
      { name: "get_route_eta", description: "Estimate travel time to a destination.", parametersJsonSchema: objectSchema({ destination_address: stringProp("Destination from a calendar event."), destination: stringProp("Destination from a calendar event.") }) },
      { name: "get_transit_status", description: "Check transit status for a route.", parametersJsonSchema: objectSchema({ destination: stringProp("Destination to check.") }, ["destination"]) },
      { name: "get_weather_brief", description: "Get a short weather brief for the destination area.", parametersJsonSchema: objectSchema({ area: stringProp("Area or destination.") }, ["area"]) },
      { name: "make_leave_time_plan", description: "Combine meeting, route, transit, and weather details into a leave-time plan.", parametersJsonSchema: objectSchema({ meeting_time: stringProp("Meeting start time."), travel_minutes: numberProp("Travel time in minutes.") }, ["meeting_time", "travel_minutes"]) },
    ],
    steps: [
      { label: "calendar", toolName: "get_next_calendar_event", mockedResult: { start_time: "3:00 PM", short_location: "NTU EE Building", destination_address: "NTU EE Building" } },
      { label: "route", toolName: "get_route_eta", dependency: dep("destination_address", 1, "destination_address", ["destination_address", "destination"]), mockedResult: { destination_address: "NTU EE Building", travel_time_minutes: 35, recommended_transport: "MRT and walking" } },
      { label: "transit", toolName: "get_transit_status", dependency: dep("destination", 2, "destination_address"), mockedResult: { route_status: "normal", extra_delay_minutes: 0 } },
      { label: "weather", toolName: "get_weather_brief", dependency: dep("area", 1, "short_location", ["area", "destination"]), mockedResult: { weather: "light rain", buffer_minutes: 5 } },
      { label: "plan", toolName: "make_leave_time_plan", dependency: dep("meeting_time", 1, "start_time"), mockedResult: { target_arrival_time: "2:50 PM", recommended_leave_time: "2:10 PM", reason: "35 minutes travel plus 10 minutes early arrival and 5 minutes rain buffer" } },
    ],
    expectedFinalAnswers: {
      2: "leave around 2:15 PM",
      3: "leave around 2:15 PM; transit is normal",
      4: "leave around 2:10 PM with a 5-minute rain buffer",
      5: "leave at 2:10 PM",
    },
    finalAnswerChecks: {
      mentionsTargetAnswer: /\b2\s*:\s*1[05]\b|\btwo\s*(ten|fifteen)\b/i,
      usesStep1Result: /\b3\s*:\s*00\b|3\s*pm|three/i,
      usesFinalStepResult: /\b35\b|rain|normal|2\s*:\s*10|2\s*:\s*15/i,
      prematureAnswer: /\b2\s*:\s*1[05]\b|leave around|should leave|leave at/i,
    },
  },
  {
    baseId: "retail_return_deadline",
    resultSlug: "retail_return_deadline_no_tick",
    userPrompt: "Can I still return the headphones from order R481? If yes, tell me the deadline and refund method.",
    tools: [
      { name: "get_order_details", description: "Retrieve order details.", parametersJsonSchema: objectSchema({ order_id: stringProp("Order id.") }, ["order_id"]) },
      { name: "get_return_policy", description: "Retrieve return policy for an item category.", parametersJsonSchema: objectSchema({ category: stringProp("Item category.") }, ["category"]) },
      { name: "check_item_condition", description: "Check whether the item condition allows return.", parametersJsonSchema: objectSchema({ item_id: stringProp("Item id.") }, ["item_id"]) },
      { name: "get_refund_method", description: "Get refund method for an order payment type.", parametersJsonSchema: objectSchema({ payment_method: stringProp("Payment method.") }, ["payment_method"]) },
      { name: "create_return_preview", description: "Create a return preview.", parametersJsonSchema: objectSchema({ order_id: stringProp("Order id."), item_id: stringProp("Item id.") }, ["order_id", "item_id"]) },
    ],
    steps: [
      { label: "order", toolName: "get_order_details", mockedResult: { order_id: "R481", item_id: "HP-22", category: "electronics", delivered_date: "June 10", payment_method: "Visa ending 4242" } },
      { label: "policy", toolName: "get_return_policy", dependency: dep("category", 1, "category"), mockedResult: { category: "electronics", return_window_days: 30, return_deadline: "July 10" } },
      { label: "condition", toolName: "check_item_condition", dependency: dep("item_id", 1, "item_id"), mockedResult: { item_id: "HP-22", condition_ok: true, packaging_required: true } },
      { label: "refund", toolName: "get_refund_method", dependency: dep("payment_method", 1, "payment_method"), mockedResult: { refund_method: "original Visa ending 4242", refund_eta_days: 5 } },
      { label: "preview", toolName: "create_return_preview", dependency: dep("order_id", 1, "order_id"), mockedResult: { return_eligible: true, deadline: "July 10", refund_method: "original Visa ending 4242", label_ready: true } },
    ],
    expectedFinalAnswers: {
      2: "yes, return by July 10",
      3: "yes, return by July 10 if packaging is included",
      4: "yes, return by July 10; refund goes to original Visa ending 4242",
      5: "yes, label is ready; return by July 10 with refund to original Visa ending 4242",
    },
    finalAnswerChecks: {
      mentionsTargetAnswer: /yes|eligible|return/i,
      usesStep1Result: /R481|headphones|electronics|June 10/i,
      usesFinalStepResult: /July 10|Visa|label|packaging/i,
      prematureAnswer: /return by|eligible|refund/i,
    },
  },
  {
    baseId: "airline_rebook_option",
    resultSlug: "airline_rebook_option_no_tick",
    userPrompt: "My flight reservation ZX92Q may be delayed. Can you find the best rebooking option and say whether there is a fee?",
    tools: [
      { name: "get_reservation_details", description: "Retrieve airline reservation details.", parametersJsonSchema: objectSchema({ reservation_id: stringProp("Reservation id.") }, ["reservation_id"]) },
      { name: "check_flight_status", description: "Check flight status.", parametersJsonSchema: objectSchema({ flight_number: stringProp("Flight number.") }, ["flight_number"]) },
      { name: "search_alternative_flights", description: "Search alternative flights for a route and date.", parametersJsonSchema: objectSchema({ route: stringProp("Route."), travel_date: stringProp("Travel date.") }, ["route", "travel_date"]) },
      { name: "check_seat_availability", description: "Check seat availability for a flight.", parametersJsonSchema: objectSchema({ flight_number: stringProp("Flight number.") }, ["flight_number"]) },
      { name: "quote_rebooking_fee", description: "Quote rebooking fee.", parametersJsonSchema: objectSchema({ fare_class: stringProp("Fare class."), disruption_status: stringProp("Disruption status.") }, ["fare_class", "disruption_status"]) },
    ],
    steps: [
      { label: "reservation", toolName: "get_reservation_details", mockedResult: { reservation_id: "ZX92Q", flight_number: "TA402", route: "SFO -> SEA", travel_date: "July 8", fare_class: "economy" } },
      { label: "status", toolName: "check_flight_status", dependency: dep("flight_number", 1, "flight_number"), mockedResult: { flight_number: "TA402", disruption_status: "delayed", delay_minutes: 95 } },
      { label: "alternatives", toolName: "search_alternative_flights", dependency: dep("route", 1, "route"), mockedResult: { best_alternative_flight: "TA418", departure_time: "6:20 PM", arrival_time: "8:30 PM" } },
      { label: "seats", toolName: "check_seat_availability", dependency: dep("flight_number", 3, "best_alternative_flight"), mockedResult: { flight_number: "TA418", seats_available: 4, cabin: "economy" } },
      { label: "fee", toolName: "quote_rebooking_fee", dependency: dep("fare_class", 1, "fare_class"), mockedResult: { rebooking_fee_usd: 0, reason: "delay waiver applies" } },
    ],
    expectedFinalAnswers: {
      2: "current flight TA402 is delayed by 95 minutes",
      3: "best alternative is TA418 at 6:20 PM",
      4: "TA418 has economy seats available",
      5: "rebook to TA418 with no fee",
    },
    finalAnswerChecks: {
      mentionsTargetAnswer: /TA418|no fee|0/i,
      usesStep1Result: /TA402|SFO|SEA|ZX92Q/i,
      usesFinalStepResult: /95|6:20|available|waiver|0/i,
      prematureAnswer: /TA418|no fee|rebook|delayed by/i,
    },
  },
  {
    baseId: "pharmacy_refill_pickup",
    resultSlug: "pharmacy_refill_pickup_no_tick",
    userPrompt: "Can I refill my allergy medicine today and pick it up at my usual pharmacy?",
    tools: [
      { name: "get_active_prescription", description: "Retrieve active prescription.", parametersJsonSchema: objectSchema({ medication: stringProp("Medication name.") }, ["medication"]) },
      { name: "check_refill_eligibility", description: "Check refill eligibility.", parametersJsonSchema: objectSchema({ prescription_id: stringProp("Prescription id.") }, ["prescription_id"]) },
      { name: "check_pharmacy_inventory", description: "Check local pharmacy inventory.", parametersJsonSchema: objectSchema({ pharmacy_id: stringProp("Pharmacy id."), ndc: stringProp("Medication code.") }, ["pharmacy_id", "ndc"]) },
      { name: "estimate_ready_time", description: "Estimate pickup ready time.", parametersJsonSchema: objectSchema({ pharmacy_id: stringProp("Pharmacy id.") }, ["pharmacy_id"]) },
      { name: "check_copay", description: "Check medication copay.", parametersJsonSchema: objectSchema({ prescription_id: stringProp("Prescription id.") }, ["prescription_id"]) },
    ],
    steps: [
      { label: "prescription", toolName: "get_active_prescription", mockedResult: { prescription_id: "RX-77", medication: "cetirizine", ndc: "CET10", usual_pharmacy_id: "PH-NTU" } },
      { label: "eligibility", toolName: "check_refill_eligibility", dependency: dep("prescription_id", 1, "prescription_id"), mockedResult: { refill_allowed: true, refill_date: "today" } },
      { label: "inventory", toolName: "check_pharmacy_inventory", dependency: dep("ndc", 1, "ndc"), mockedResult: { pharmacy_id: "PH-NTU", in_stock: true, quantity_available: 48 } },
      { label: "ready_time", toolName: "estimate_ready_time", dependency: dep("pharmacy_id", 1, "usual_pharmacy_id"), mockedResult: { ready_time: "5:30 PM" } },
      { label: "copay", toolName: "check_copay", dependency: dep("prescription_id", 1, "prescription_id"), mockedResult: { copay_usd: 8 } },
    ],
    expectedFinalAnswers: {
      2: "yes, refill is allowed today",
      3: "yes, refill is allowed and in stock",
      4: "yes, pickup should be ready at 5:30 PM",
      5: "yes, ready at 5:30 PM with an $8 copay",
    },
    finalAnswerChecks: {
      mentionsTargetAnswer: /yes|allowed|ready/i,
      usesStep1Result: /cetirizine|RX-77|CET10/i,
      usesFinalStepResult: /in stock|5:30|8|copay/i,
      prematureAnswer: /allowed|in stock|ready|copay/i,
    },
  },
  {
    baseId: "restaurant_arrival_plan",
    resultSlug: "restaurant_arrival_plan_no_tick",
    userPrompt: "I have dinner tonight. Can you tell me when to leave and whether parking might be an issue?",
    tools: [
      { name: "get_dinner_reservation", description: "Retrieve dinner reservation.", parametersJsonSchema: emptySchema },
      { name: "get_route_eta", description: "Estimate route ETA.", parametersJsonSchema: objectSchema({ destination: stringProp("Destination.") }, ["destination"]) },
      { name: "check_parking_status", description: "Check parking status near restaurant.", parametersJsonSchema: objectSchema({ venue: stringProp("Venue.") }, ["venue"]) },
      { name: "check_weather_brief", description: "Check local weather.", parametersJsonSchema: objectSchema({ area: stringProp("Area.") }, ["area"]) },
      { name: "create_arrival_plan", description: "Create arrival plan.", parametersJsonSchema: objectSchema({ reservation_time: stringProp("Reservation time."), travel_minutes: numberProp("Travel minutes.") }, ["reservation_time", "travel_minutes"]) },
    ],
    steps: [
      { label: "reservation", toolName: "get_dinner_reservation", mockedResult: { venue: "Luna Bistro", reservation_time: "7:00 PM", destination: "Luna Bistro" } },
      { label: "route", toolName: "get_route_eta", dependency: dep("destination", 1, "destination"), mockedResult: { travel_time_minutes: 28 } },
      { label: "parking", toolName: "check_parking_status", dependency: dep("venue", 1, "venue"), mockedResult: { parking_status: "limited", extra_buffer_minutes: 10 } },
      { label: "weather", toolName: "check_weather_brief", dependency: dep("area", 1, "destination"), mockedResult: { weather: "clear", weather_buffer_minutes: 0 } },
      { label: "plan", toolName: "create_arrival_plan", dependency: dep("reservation_time", 1, "reservation_time"), mockedResult: { recommended_leave_time: "6:12 PM", reason: "28 minutes travel plus 10 minutes parking buffer and 10 minutes early arrival" } },
    ],
    expectedFinalAnswers: {
      2: "leave around 6:22 PM",
      3: "leave around 6:12 PM because parking is limited",
      4: "leave around 6:12 PM; weather is clear",
      5: "leave at 6:12 PM",
    },
    finalAnswerChecks: {
      mentionsTargetAnswer: /6\s*:\s*(12|22)|six\s*(twelve|twenty-two)/i,
      usesStep1Result: /7\s*:\s*00|Luna/i,
      usesFinalStepResult: /28|limited|clear|6\s*:\s*12/i,
      prematureAnswer: /leave|parking is|6\s*:\s*/i,
    },
  },
  {
    baseId: "subscription_plan_change",
    resultSlug: "subscription_plan_change_no_tick",
    userPrompt: "Can you check whether I should switch my subscription to the Plus plan this month?",
    tools: [
      { name: "get_subscription", description: "Retrieve subscription.", parametersJsonSchema: emptySchema },
      { name: "get_usage_summary", description: "Retrieve monthly usage.", parametersJsonSchema: objectSchema({ account_id: stringProp("Account id.") }, ["account_id"]) },
      { name: "compare_plans", description: "Compare subscription plans.", parametersJsonSchema: objectSchema({ current_plan: stringProp("Current plan."), target_plan: stringProp("Target plan.") }, ["current_plan", "target_plan"]) },
      { name: "calculate_proration", description: "Calculate prorated charge.", parametersJsonSchema: objectSchema({ account_id: stringProp("Account id."), target_plan: stringProp("Target plan.") }, ["account_id", "target_plan"]) },
      { name: "prepare_plan_change_quote", description: "Prepare plan-change quote.", parametersJsonSchema: objectSchema({ account_id: stringProp("Account id.") }, ["account_id"]) },
    ],
    steps: [
      { label: "subscription", toolName: "get_subscription", mockedResult: { account_id: "AC-91", current_plan: "Basic", target_plan: "Plus" } },
      { label: "usage", toolName: "get_usage_summary", dependency: dep("account_id", 1, "account_id"), mockedResult: { monthly_usage_gb: 92, basic_limit_gb: 50 } },
      { label: "compare", toolName: "compare_plans", dependency: dep("current_plan", 1, "current_plan"), mockedResult: { plus_limit_gb: 200, monthly_price_delta_usd: 12 } },
      { label: "proration", toolName: "calculate_proration", dependency: dep("account_id", 1, "account_id"), mockedResult: { prorated_charge_usd: 7 } },
      { label: "quote", toolName: "prepare_plan_change_quote", dependency: dep("account_id", 1, "account_id"), mockedResult: { recommendation: "switch to Plus", due_today_usd: 7, new_monthly_delta_usd: 12 } },
    ],
    expectedFinalAnswers: {
      2: "usage is over the Basic limit",
      3: "Plus fits usage and costs $12 more monthly",
      4: "switching now costs $7 prorated",
      5: "switch to Plus; due today is $7 and monthly increase is $12",
    },
    finalAnswerChecks: {
      mentionsTargetAnswer: /switch|Plus|over/i,
      usesStep1Result: /Basic|AC-91/i,
      usesFinalStepResult: /92|200|12|7/i,
      prematureAnswer: /switch|Plus|cost|due today/i,
    },
  },
  {
    baseId: "device_repair_appointment",
    resultSlug: "device_repair_appointment_no_tick",
    userPrompt: "My laptop screen is flickering. Can you check repair options and the earliest appointment?",
    tools: [
      { name: "get_device_warranty", description: "Retrieve device warranty.", parametersJsonSchema: objectSchema({ device: stringProp("Device.") }, ["device"]) },
      { name: "diagnose_issue", description: "Diagnose issue category.", parametersJsonSchema: objectSchema({ symptom: stringProp("Symptom.") }, ["symptom"]) },
      { name: "find_service_center", description: "Find service center.", parametersJsonSchema: objectSchema({ repair_type: stringProp("Repair type.") }, ["repair_type"]) },
      { name: "check_appointment_slots", description: "Check appointment slots.", parametersJsonSchema: objectSchema({ center_id: stringProp("Center id.") }, ["center_id"]) },
      { name: "estimate_repair_cost", description: "Estimate repair cost.", parametersJsonSchema: objectSchema({ warranty_status: stringProp("Warranty status."), repair_type: stringProp("Repair type.") }, ["warranty_status", "repair_type"]) },
    ],
    steps: [
      { label: "warranty", toolName: "get_device_warranty", mockedResult: { device_id: "LTP-44", warranty_status: "active" } },
      { label: "diagnosis", toolName: "diagnose_issue", dependency: dep("symptom", 1, "device_id"), mockedResult: { repair_type: "display cable check", severity: "moderate" } },
      { label: "center", toolName: "find_service_center", dependency: dep("repair_type", 2, "repair_type"), mockedResult: { center_id: "SC-12", center_name: "Downtown Service" } },
      { label: "slots", toolName: "check_appointment_slots", dependency: dep("center_id", 3, "center_id"), mockedResult: { earliest_slot: "tomorrow 10:30 AM" } },
      { label: "cost", toolName: "estimate_repair_cost", dependency: dep("warranty_status", 1, "warranty_status"), mockedResult: { estimated_cost_usd: 0, reason: "covered by active warranty" } },
    ],
    expectedFinalAnswers: {
      2: "warranty is active and issue looks like a display cable check",
      3: "Downtown Service can handle it",
      4: "earliest appointment is tomorrow at 10:30 AM",
      5: "earliest appointment is tomorrow at 10:30 AM and estimated cost is $0",
    },
    finalAnswerChecks: {
      mentionsTargetAnswer: /tomorrow|10:30|covered|0/i,
      usesStep1Result: /active|LTP-44/i,
      usesFinalStepResult: /display cable|Downtown|10:30|0/i,
      prematureAnswer: /tomorrow|appointment|covered|cost/i,
    },
  },
  {
    baseId: "event_ticket_transfer",
    resultSlug: "event_ticket_transfer_no_tick",
    userPrompt: "Can I transfer my concert ticket to Maya and confirm how she will receive it?",
    tools: [
      { name: "get_ticket_order", description: "Retrieve ticket order.", parametersJsonSchema: objectSchema({ event: stringProp("Event.") }, ["event"]) },
      { name: "verify_recipient", description: "Verify transfer recipient.", parametersJsonSchema: objectSchema({ recipient_name: stringProp("Recipient name.") }, ["recipient_name"]) },
      { name: "check_transfer_policy", description: "Check ticket transfer policy.", parametersJsonSchema: objectSchema({ event_id: stringProp("Event id.") }, ["event_id"]) },
      { name: "create_transfer_preview", description: "Create transfer preview.", parametersJsonSchema: objectSchema({ ticket_id: stringProp("Ticket id."), recipient_id: stringProp("Recipient id.") }, ["ticket_id", "recipient_id"]) },
      { name: "confirm_delivery_method", description: "Confirm recipient delivery method.", parametersJsonSchema: objectSchema({ recipient_id: stringProp("Recipient id.") }, ["recipient_id"]) },
    ],
    steps: [
      { label: "ticket", toolName: "get_ticket_order", mockedResult: { event_id: "EV-8", ticket_id: "TCK-19", event_name: "Indie Night" } },
      { label: "recipient", toolName: "verify_recipient", dependency: dep("recipient_name", 1, "event_name"), mockedResult: { recipient_name: "Maya", recipient_id: "USR-MAYA", verified: true } },
      { label: "policy", toolName: "check_transfer_policy", dependency: dep("event_id", 1, "event_id"), mockedResult: { transferable: true, deadline: "6:00 PM today" } },
      { label: "preview", toolName: "create_transfer_preview", dependency: dep("ticket_id", 1, "ticket_id"), mockedResult: { preview_ready: true, recipient_name: "Maya" } },
      { label: "delivery", toolName: "confirm_delivery_method", dependency: dep("recipient_id", 2, "recipient_id"), mockedResult: { delivery_method: "mobile ticket link by email" } },
    ],
    expectedFinalAnswers: {
      2: "ticket and Maya are verified",
      3: "yes, transferable until 6:00 PM today",
      4: "transfer preview is ready for Maya",
      5: "yes, Maya will receive a mobile ticket link by email",
    },
    finalAnswerChecks: {
      mentionsTargetAnswer: /yes|transfer|Maya/i,
      usesStep1Result: /TCK-19|Indie Night|EV-8/i,
      usesFinalStepResult: /verified|6:00|preview|email|mobile/i,
      prematureAnswer: /transferable|preview|receive|email/i,
    },
  },
  {
    baseId: "grocery_recipe_shopping",
    resultSlug: "grocery_recipe_shopping_no_tick",
    userPrompt: "Can you check what I need to buy for pesto pasta tonight and whether pickup is available?",
    tools: [
      { name: "get_recipe_ingredients", description: "Retrieve recipe ingredients.", parametersJsonSchema: objectSchema({ recipe: stringProp("Recipe.") }, ["recipe"]) },
      { name: "check_pantry_items", description: "Check pantry items.", parametersJsonSchema: objectSchema({ ingredients: stringProp("Ingredient list.") }, ["ingredients"]) },
      { name: "search_grocery_inventory", description: "Search grocery inventory.", parametersJsonSchema: objectSchema({ missing_items: stringProp("Missing item list.") }, ["missing_items"]) },
      { name: "estimate_cart_total", description: "Estimate cart total.", parametersJsonSchema: objectSchema({ cart_items: stringProp("Cart item list.") }, ["cart_items"]) },
      { name: "choose_pickup_slot", description: "Choose pickup slot.", parametersJsonSchema: objectSchema({ store_id: stringProp("Store id.") }, ["store_id"]) },
    ],
    steps: [
      { label: "recipe", toolName: "get_recipe_ingredients", mockedResult: { recipe: "pesto pasta", ingredients: "basil, pasta, parmesan, pine nuts", serving_count: 2 } },
      { label: "pantry", toolName: "check_pantry_items", dependency: dep("ingredients", 1, "ingredients"), mockedResult: { have: "pasta", missing_items: "basil, parmesan, pine nuts" } },
      { label: "inventory", toolName: "search_grocery_inventory", dependency: dep("missing_items", 2, "missing_items"), mockedResult: { store_id: "G-5", available_items: "basil, parmesan, pine nuts" } },
      { label: "cart", toolName: "estimate_cart_total", dependency: dep("cart_items", 3, "available_items"), mockedResult: { estimated_total_usd: 18.5 } },
      { label: "pickup", toolName: "choose_pickup_slot", dependency: dep("store_id", 3, "store_id"), mockedResult: { pickup_available: true, pickup_slot: "6:30 PM" } },
    ],
    expectedFinalAnswers: {
      2: "you need basil, parmesan, and pine nuts",
      3: "missing items are available at store G-5",
      4: "cart total is about $18.50",
      5: "pickup is available at 6:30 PM for about $18.50",
    },
    finalAnswerChecks: {
      mentionsTargetAnswer: /basil|parmesan|pine nuts|pickup/i,
      usesStep1Result: /pesto|pasta|2/i,
      usesFinalStepResult: /18\.?5|6:30|available|G-5/i,
      prematureAnswer: /need|available|total|pickup/i,
    },
  },
  {
    baseId: "banking_bill_pay",
    resultSlug: "banking_bill_pay_no_tick",
    userPrompt: "Can you check whether I can pay my electricity bill today without risking an overdraft?",
    tools: [
      { name: "get_upcoming_bill", description: "Retrieve upcoming bill.", parametersJsonSchema: objectSchema({ biller: stringProp("Biller.") }, ["biller"]) },
      { name: "get_account_balance", description: "Retrieve account balance.", parametersJsonSchema: objectSchema({ account_id: stringProp("Account id.") }, ["account_id"]) },
      { name: "check_payment_policy", description: "Check payment policy.", parametersJsonSchema: objectSchema({ biller_id: stringProp("Biller id.") }, ["biller_id"]) },
      { name: "estimate_processing_time", description: "Estimate bill payment processing time.", parametersJsonSchema: objectSchema({ biller_id: stringProp("Biller id.") }, ["biller_id"]) },
      { name: "create_payment_plan", description: "Create safe payment plan.", parametersJsonSchema: objectSchema({ amount_due: numberProp("Amount due."), available_balance: numberProp("Available balance.") }, ["amount_due", "available_balance"]) },
    ],
    steps: [
      { label: "bill", toolName: "get_upcoming_bill", mockedResult: { biller_id: "ELEC", amount_due: 86, due_date: "Friday", account_id: "CHK-2" } },
      { label: "balance", toolName: "get_account_balance", dependency: dep("account_id", 1, "account_id"), mockedResult: { available_balance: 240, minimum_buffer: 100 } },
      { label: "policy", toolName: "check_payment_policy", dependency: dep("biller_id", 1, "biller_id"), mockedResult: { same_day_allowed: true, fee_usd: 0 } },
      { label: "processing", toolName: "estimate_processing_time", dependency: dep("biller_id", 1, "biller_id"), mockedResult: { processing_time: "same day" } },
      { label: "plan", toolName: "create_payment_plan", dependency: dep("amount_due", 1, "amount_due"), mockedResult: { safe_to_pay_today: true, remaining_balance_after_payment: 154 } },
    ],
    expectedFinalAnswers: {
      2: "yes, balance covers the $86 bill while keeping the $100 buffer",
      3: "yes, same-day payment has no fee",
      4: "yes, payment can process same day",
      5: "yes, safe to pay today; remaining balance would be $154",
    },
    finalAnswerChecks: {
      mentionsTargetAnswer: /yes|safe|pay today/i,
      usesStep1Result: /86|Friday|ELEC/i,
      usesFinalStepResult: /240|100|same day|154|0/i,
      prematureAnswer: /safe|pay today|same day|remaining/i,
    },
  },
];

export const MULTI_STEP_TASK_VARIANTS: MultiStepTaskVariant[] = BASE_TASKS.flatMap(variants);

const VARIANTS_BY_ID: Record<string, MultiStepTaskVariant> = Object.fromEntries(
  MULTI_STEP_TASK_VARIANTS.map((variant) => [variant.id, variant]),
);

// Backward-compatible alias for the original calendar-route two-step runner.
VARIANTS_BY_ID.calendar_route_leave_time = VARIANTS_BY_ID.calendar_route_leave_time_2step;

export function listMultiStepTaskVariantIds(): string[] {
  return Object.keys(VARIANTS_BY_ID).sort();
}

export function getMultiStepTaskVariant(id: string): MultiStepTaskVariant {
  const task = VARIANTS_BY_ID[id];
  if (!task) throw new Error(`Unknown multi-step task variant: ${id}. Available tasks: ${listMultiStepTaskVariantIds().join(", ")}`);
  return task;
}

export function getTwoStepTaskVariants(): MultiStepTaskVariant[] {
  return MULTI_STEP_TASK_VARIANTS.filter((variant) => variant.stepCount === 2);
}

export function makeMultiStepToolDeclarations(task: MultiStepTaskVariant): unknown[] {
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
