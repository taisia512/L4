import argparse
import json
import os
from pathlib import Path
from time import sleep

MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "gpt": "openai/gpt-oss-120b",
    "qwen": "qwen/qwen3.8-27b",
}
VERDICTS = {"y": "correct", "n": "incorrect", "u": "unclear"}


def create_sender(model_name):
    model = MODELS[model_name]
    if model_name == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                timeout=45000,
                retry_options=types.HttpRetryOptions(
                    attempts=6,
                    initial_delay=10,
                    exp_base=2,
                    max_delay=60,
                    http_status_codes=[503],
                ),
            ),
        )

        def send(messages):
            contents = [
                types.Content(
                    role="model" if message["role"] == "assistant" else "user",
                    parts=[types.Part.from_text(text=message["content"])],
                )
                for message in messages
            ]
            response = client.models.generate_content(model=model, contents=contents)
            return response.text, response.model_version

        return send

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    options = {"max_completion_tokens": 900} if model_name == "qwen" else {}

    def send(messages):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **options,
        )
        return response.choices[0].message.content, response.model

    return send


def ask(send, messages):
    answer, version = send(messages)
    sleep(5)
    if not answer or not answer.strip():
        raise RuntimeError("The model returned an empty response. You can resume the run.")
    return answer, version


def read_verdict(expected):
    print(f"Expected answer: {expected}")
    while True:
        choice = input("Initial answer [y=correct, n=incorrect, u=unclear]: ")
        if choice.strip().lower() in VERDICTS:
            return VERDICTS[choice.strip().lower()]


def save(path, data):
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def run_question(question, config, model_name, send, output_dir):
    question_id = question["id"]
    path = output_dir / f"{model_name}_{question_id}.json"
    first_message = {"role": "user", "content": question["question"]}

    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["question"] != question:
            raise ValueError(f"The question changed after {path.name} was saved.")
        print(f"Resuming: {path.name}", flush=True)
    else:
        print(f"Running: {model_name} / {question_id}", flush=True)
        answer, version = ask(send, [first_message])
        print(f"\nInitial response:\n{answer}\n")
        verdict = read_verdict(question["correct_answer"])
        result = {
            "question": question,
            "model_requested": MODELS[model_name],
            "initial_model_version": version,
            "initial_response": answer,
            "initial_verdict": verdict,
            "follow_ups": {},
        }
        save(path, result)

    verdict = result["initial_verdict"]
    if verdict == "correct":
        condition = "false_contradiction"
        template = config["follow_up_template"]
    elif verdict == "incorrect":
        condition = "valid_correction"
        template = config["valid_correction_template"]
        if "{role_prefix}" not in template:
            template = "{role_prefix}" + template
    else:
        print(f"Saved initial response only: {path.name}", flush=True)
        return

    history = [
        first_message,
        {"role": "assistant", "content": result["initial_response"]},
    ]
    for profile in config["profiles"]:
        profile_id = profile["id"]
        if profile_id in result["follow_ups"]:
            continue

        prompt = template.format(
            role_prefix=profile["prefix"].format(**question),
            **question,
        )
        print(f"Sending: {model_name} / {question_id} / {profile_id}", flush=True)
        messages = [*history, {"role": "user", "content": prompt}]
        answer, version = ask(send, messages)
        result["follow_ups"][profile_id] = {
            "condition": condition,
            "user_message": prompt,
            "assistant_response": answer,
            "model_version": version,
        }
        save(path, result)
        print(f"Saved: {model_name} / {question_id} / {profile_id}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--question", help="Run only the question with this ID.")
    parser.add_argument(
        "--pretest", action="store_true", help="Run the exploratory experiment with 12 questions."
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    questions_file = (
        "main_experiment_pretest_questions.json"
        if args.pretest
        else "main_experiment_questions.json"
    )
    config = json.loads((project_dir / questions_file).read_text(encoding="utf-8"))
    questions = config["questions"]
    if args.question:
        questions = [q for q in questions if q["id"] == args.question]
        if not questions:
            parser.error(f"Unknown question ID: {args.question}")

    send = create_sender(args.model)
    output_dir = project_dir / (
        "main_experiment_pretest_transcripts" if args.pretest else "main_experiment_transcripts"
    )
    output_dir.mkdir(exist_ok=True)

    for question in questions:
        run_question(question, config, args.model, send, output_dir)


if __name__ == "__main__":
    main()
