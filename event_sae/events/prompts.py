"""Gemini VLM prompts for event-cluster annotation.

`PHASE_LABELS` is the closed phase vocabulary used in the paper. The prompt
asks the model to assign one canonical phrase + one phase to a cluster of
visually-similar 5-frame bundles.
"""

from __future__ import annotations

PROMPT_VERSION = "gemini_task_local_event_cluster_png_v3"

PHASE_LABELS = (
    "pre_grasp",
    "immobilization",
    "contact",
    "detach",
    "post_grasp",
    "transition",
)

PHASE_DESCRIPTIONS = {
    "pre_grasp": "gripper is open and the robot has not yet firmly engaged the target object",
    "immobilization": "gripper is closing around the object but firm contact is not yet clearly established",
    "contact": "gripper has firm contact with the object and the object appears secured",
    "detach": "gripper is opening to release the object but is not yet fully open",
    "post_grasp": "gripper is fully open after the main grasp or release interaction has completed",
    "transition": "unclear, temporary, or ambiguous transition that does not cleanly fit the other labels",
}


def build_cluster_annotation_prompt(
    *,
    task_description: str,
    cluster_id: str,
    num_sequences: int,
    num_frames_per_sequence: int,
    episode_coverage: float,
    relative_progress_percents: list[float] | None = None,
) -> str:
    phase_lines = "\n".join(
        f'- "{phase}": {PHASE_DESCRIPTIONS[phase]}' for phase in PHASE_LABELS
    )
    progress_lines = ""
    if relative_progress_percents:
        formatted_progress = "\n".join(
            f"- Clip {clip_idx}: about {100.0 * progress_percent:.1f}% of its source trajectory"
            for clip_idx, progress_percent in enumerate(relative_progress_percents, start=1)
        )
        progress_lines = (
            "Representative relative progress by clip:\n"
            f"{formatted_progress}\n"
            "Use relative progress only as a weak cue for early/mid/late-stage disambiguation.\n"
            "Do not let progress override clear visual evidence.\n\n"
        )
    return f"""You are labeling a recurring event in a robot manipulation task.

You will be shown {num_sequences} image sequences from different rollout episodes of the same task.
Each sequence contains {num_frames_per_sequence} images in chronological order.
All sequences were clustered automatically and are intended to depict the same recurring event type.
Treat each sequence as one short clip. The model input is therefore a small batch of clips:
clip 1 = consecutive frames from one episode,
clip 2 = consecutive frames from another episode,
and so on.

Task instruction: "{task_description}"
Cluster id: {cluster_id}
Episode coverage: {episode_coverage:.3f}
{progress_lines}Your job is to assign one canonical phrase describing the shared event across the sequences.

Focus on the common event across clips, not small differences between episodes.
Do not interpret the entire image list as one long timeline; instead, reason about each clip separately and then summarize the shared event.
The phrase should describe the main action or transition happening within each short clip, especially the change from earlier frames to later frames.
Prefer a dynamic event description over a static state description.
When a clip clearly shows placing, releasing, lifting, grasping, approaching, or withdrawing, use that event in the phrase rather than a generic state like "holding the object".
If the later frames make the event clearer than the earlier frames, prioritize the later frames when choosing the phrase and phase.

Choose exactly one phase label from this closed set:
{phase_lines}

Return JSON with exactly two keys:
{{
  "phrase": "short canonical event phrase",
  "phase": "one label from the closed set above"
}}

Requirements:
- The top-level JSON value must be a single object, not a list or array.
- Output exactly one short human-readable phrase describing the common event across the sequences.
- The phrase should read naturally to a person.
- Prefer a brief action-focused phrase, not a full sentence.
- Prefer an event verb phrase such as "releasing the object into the basket" or "approaching the object", not a generic state phrase such as "holding the object", unless the clip truly shows no clearer transition.
- If the object is already supported by the basket or table and the gripper is opening or withdrawing, describe that as placing/releasing rather than holding.
- The phase must be exactly one of the allowed labels above.
- Do not output multiple tags or a long explanation.
- Do not include any keys other than "phrase" and "phase".
- Do not include markdown fences.
"""
