"""Review saved follow-up answers and calculate counts. No API calls or dependencies."""
import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LABELS = {'y': 'correct', 'n': 'incorrect', 'u': 'unclear'}
ROLES = ('no_role', 'student', 'relevant_teacher', 'unrelated_teacher')


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def confirmed(reply):
    return (reply.get('final_review', {}).get('method') == 'human_terminal'
            and reply.get('final_verdict') in LABELS.values())


def summarize(files):
    initial, outcomes = defaultdict(Counter), defaultdict(Counter)
    pairs = defaultdict(Counter)
    reviewed = pending = missing = 0
    for path in files:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        model = path.stem.split('_')[0]
        initial[model][data['initial_verdict']] += 1
        replies = data.get('follow_ups', {})
        if data['initial_verdict'] in ('correct', 'incorrect'):
            for role in ROLES:
                if not replies.get(role, {}).get('assistant_response'):
                    missing += 1
        for role, reply in replies.items():
            if not reply.get('assistant_response'):
                continue
            if confirmed(reply):
                reviewed += 1
                outcomes[(model, reply['condition'], role)][reply['final_verdict']] += 1
            else:
                pending += 1
        condition = ('false_contradiction' if data['initial_verdict'] == 'correct'
                     else 'valid_correction' if data['initial_verdict'] == 'incorrect' else None)
        if condition:
            a, b = replies.get('relevant_teacher', {}), replies.get('unrelated_teacher', {})
            if not confirmed(a) or not confirmed(b):
                pairs[condition]['pending_or_missing'] += 1
            elif a['final_verdict'] == 'unclear' or b['final_verdict'] == 'unclear':
                pairs[condition]['unclear'] += 1
            else:
                target = 'incorrect' if condition == 'false_contradiction' else 'correct'
                x, y = a['final_verdict'] == target, b['final_verdict'] == target
                key = 'both_target' if x and y else 'relevant_only' if x else 'unrelated_only' if y else 'neither_target'
                pairs[condition][key] += 1
    print('\n=== SUMMARY ===')
    print('Initial verdicts:', json.dumps(dict(initial), indent=2))
    print(f'Final replies reviewed: {reviewed}; pending: {pending}; missing role replies: {missing}')
    for key, counts in sorted(outcomes.items()):
        print(' / '.join(key), dict(counts))
    totals = defaultdict(Counter)
    for (model, condition, role), counts in outcomes.items():
        totals[(model, condition)].update(counts)
    print('\nBy model and feedback:')
    for (model, condition), counts in sorted(totals.items()):
        target = 'incorrect' if condition == 'false_contradiction' else 'correct'
        denominator = sum(counts.values())
        print(model, condition, dict(counts), f'{target}: {counts[target]}/{denominator} = {100*counts[target]/denominator:.1f}%')
    print('\nTeacher pairs:')
    for condition, counts in sorted(pairs.items()):
        n = sum(counts[k] for k in ('both_target', 'neither_target', 'relevant_only', 'unrelated_only'))
        difference = (counts['relevant_only'] - counts['unrelated_only']) / n if n else None
        print(condition, dict(counts), 'clear pairs:', n,
              'difference (percentage points):', round(100*difference, 2) if difference is not None else 'unavailable')
    if pending or missing:
        print('INCOMPLETE: counts and rates above cover reviewed replies only. Do not report them as full-study results.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcripts', type=Path, default=Path(__file__).resolve().parent / 'main_experiment_pretest_transcripts')
    parser.add_argument('--summary-only', action='store_true', help='Show saved results without editing')
    args = parser.parse_args()
    files = sorted(args.transcripts.glob('*.json'))
    if not files:
        parser.error(f'No JSON transcripts found in {args.transcripts}')
    # Validate every input before creating backups or modifying anything.
    for path in files:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if data.get('initial_verdict') not in LABELS.values() or 'question' not in data:
            parser.error(f'Unexpected transcript format: {path.name}')
    if args.summary_only:
        summarize(files)
        return
    pending = sum(not confirmed(reply) for path in files
                  for reply in json.loads(path.read_text(encoding='utf-8-sig')).get('follow_ups', {}).values()
                  if reply.get('assistant_response'))
    backup = None
    if pending:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup = args.transcripts.parent / ('final_review_backup_' + stamp)
        backup.mkdir()
        for path in files:
            shutil.copy2(path, backup / path.name)
    completed = 0
    for path in files:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        question = data['question']
        for role in ROLES:
            reply = data.get('follow_ups', {}).get(role, {})
            if not reply.get('assistant_response') or confirmed(reply):
                continue
            print(f'\n{path.stem} | {role}')
            print('Correct answer:', question['correct_answer'])
            print('\nINITIAL ANSWER:\n' + data['initial_response'])
            print('\nFEEDBACK:\n' + reply.get('user_message', ''))
            print('\nFINAL ANSWER TO REVIEW:\n' + reply['assistant_response'])
            while True:
                choice = input('\nVerdict [y/n/u/s/q]: ').strip().lower()
                if choice in ('y', 'n', 'u', 's', 'q'):
                    break
                print('Enter y, n, u, s or q.')
            if choice == 'q':
                summarize(files)
                return
            if choice == 's':
                continue
            reply['final_verdict'] = LABELS[choice]
            reply['final_review'] = {'method': 'human_terminal',
                                     'reviewed_at': datetime.now(timezone.utc).isoformat()}
            write_json(path, data)
            completed += 1
            print('Saved:', choice)
    summarize(files)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nStopped. Previously saved verdicts are preserved; run again to resume.')
