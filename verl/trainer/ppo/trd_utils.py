"""Science and coding refinement prompts for Trajectory-Refined Distillation (TRD)."""


SDPO_SCIENCE_REFINEMENT_PROMPT = (
    "Your task is to rewrite your scientific solution using the reference solution as guidance.\n"
    r"""Problem:
{PROBLEM}
Reference Solution:
{EXPERT SOLUTION}
Your Initial Solution:
{INITIAL RESPONSE}
Instructions:
1. Review the reference solution to understand the target reasoning and method
2. Rewrite your solution so it is consistent with the reference solution
3. Keep useful parts of your original structure and style when appropriate
4. Output ONLY the rewritten solution
"""
)


SDPO_CODING_SOLUTION_SECTION = "Correct solution:\n\n{successful_previous_attempt}\n\n"
SDPO_CODING_FEEDBACK_SECTION = (
    "The following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n"
)


SDPO_CODING_REFINEMENT_PROMPT = """Your task is to rewrite Your Initial Solution using the available correct solution and execution feedback as guidance.
Problem:
{PROBLEM}

{SOLUTION_SECTION}Your Initial Solution:
{INITIAL RESPONSE}

{FEEDBACK_SECTION}Instructions:
1. Use Correct solution as guidance when provided. Any execution feedback refers to Your Initial Solution.
2. Fix the underlying algorithm, edge cases, runtime errors, and performance issues; do not hard-code test cases.
3. Preserve the required function signature or standard input/output interface and respect the problem constraints.
4. Output ONLY the complete corrected solution inside a ```python ... ``` code block, with no explanations or test output.
"""


def build_sdpo_science_refinement_prompt(
    problem: str,
    expert_solution: str | None,
    initial_response: str,
    feedback: str | None = None,
) -> str:
    """Render the science prompt, optionally using feedback with or without a reference."""
    template = SDPO_SCIENCE_REFINEMENT_PROMPT
    if expert_solution is None:
        template = template.replace("Reference Solution:\n{EXPERT SOLUTION}\n", "")
    if feedback:
        template = template.replace("the reference solution", "the available guidance")
        template = template.replace("Instructions:\n", "Environment Feedback:\n{FEEDBACK}\nInstructions:\n")
    return template.format_map(
        {
            "PROBLEM": problem,
            "EXPERT SOLUTION": expert_solution or "",
            "INITIAL RESPONSE": initial_response,
            "FEEDBACK": feedback or "",
        }
    )


def build_sdpo_coding_refinement_prompt(
    problem: str,
    expert_solution: str | None,
    initial_response: str,
    feedback: str | None = None,
) -> str:
    """Render code refinement using a successful sibling, execution feedback, or both."""
    return SDPO_CODING_REFINEMENT_PROMPT.format_map(
        {
            "PROBLEM": problem,
            "SOLUTION_SECTION": (
                SDPO_CODING_SOLUTION_SECTION.format(successful_previous_attempt=expert_solution)
                if expert_solution is not None else ""
            ),
            "INITIAL RESPONSE": initial_response,
            "FEEDBACK_SECTION": SDPO_CODING_FEEDBACK_SECTION.format(feedback_raw=feedback) if feedback else "",
        }
    )
