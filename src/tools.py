"""Stage 3 - Tools: structured helpers (Pydantic schemas + model calls)."""
import random

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float = Field(ge=0, le=1, description="Left edge as a fraction of image width")
    y: float = Field(ge=0, le=1, description="Top edge as a fraction of image height")
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class UnalterableVisual(BaseModel):
    label: str = Field(description="Short neutral label, e.g. 'Gaussian heatmap'")
    kind: str = Field(description="graph, heatmap, diagram, formula, photo, table")
    bbox: BoundingBox


class SlideExtraction(BaseModel):
    is_lecture_slide: bool
    scope_reason: str = Field(description="One sentence on why this is or is not a lecture slide")
    title: str = Field(description="Slide title verbatim; empty string if none")
    unalterable_visuals: list[UnalterableVisual]
    core_terminology: list[str] = Field(
        description="Terms, formulas and exact definitions copied verbatim from the slide text"
    )
    tweakable_content: list[str] = Field(
        description="Explanatory sentences outside the visuals that may be paraphrased"
    )


class MaskQuestion(BaseModel):
    sentence: str = Field(description="Sentence containing exactly one [MASK] placeholder")
    answer: str = Field(description="The word or phrase hidden by [MASK]")
    distractors: list[str] = Field(description="Exactly three plausible wrong options")


class MaskSet(BaseModel):
    questions: list[MaskQuestion]


def parse_structured(client, model: str, messages: list[dict], schema: type[BaseModel]):
    completion = client.chat.completions.parse(
        model=model, messages=messages, response_format=schema, temperature=0.3
    )
    message = completion.choices[0].message
    if message.refusal or message.parsed is None:
        raise RuntimeError(message.refusal or "The model returned no structured output.")
    return message.parsed


def extract_and_categorize_slide(client, model: str, system_prompt: str, image_b64: str) -> SlideExtraction:
    """Vision call: sort slide contents into unalterable visuals, core terminology and tweakable content."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Categorize this slide. Give tight bounding boxes (fractions of the image) for every "
                        "graph, heatmap, diagram, formula image, table or photo. Copy terminology verbatim. "
                        "If the image is not a lecture slide, set is_lecture_slide to false and leave lists empty."
                    ),
                },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"}},
            ],
        },
    ]
    return parse_structured(client, model, messages, SlideExtraction)


def _valid_question(question: MaskQuestion) -> bool:
    options = {question.answer.strip().lower(), *(d.strip().lower() for d in question.distractors)}
    return (
        question.sentence.count("[MASK]") == 1
        and bool(question.answer.strip())
        and len(question.distractors) >= 2
        and len(options) == len(question.distractors) + 1
    )


def generate_active_recall_masks(
    client, model: str, system_prompt: str, text: str, core_terms: list[str], max_questions: int = 3
) -> list[dict]:
    """Turn paraphrased text into fill-in-the-blank questions with multiple-choice options."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Create up to {max_questions} fill-in-the-blank questions from the text below. Each sentence must "
                "contain exactly one [MASK]. Prefer masking a core term or key concept from this list when it appears: "
                f"{core_terms}. Keep every other core term verbatim. Give exactly three plausible distractors from the "
                "same subject area.\n\nTEXT:\n" + text
            ),
        },
    ]
    result = parse_structured(client, model, messages, MaskSet)
    questions = []
    for index, question in enumerate(q for q in result.questions if _valid_question(q)):
        options = [question.answer.strip(), *(d.strip() for d in question.distractors)]
        random.Random(f"{question.sentence}{index}").shuffle(options)
        questions.append({"sentence": question.sentence, "answer": question.answer.strip(), "options": options})
    return questions[:max_questions]
