#!/usr/bin/env python3
"""Deterministic validation for output.csv against the Buy or Wait? contract.

Run from repository root:
    python3 code/evaluation/main.py

This checks structural/rule compliance only (it cannot verify that
amount_safe_to_pay is *optimal*, only that the submission is internally
consistent and legal per problem_statement.md / AGENTS.md section 6).
"""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dataset'
OUT = ROOT / 'output.csv'

COLS = ['request_id', 'amount_safe_to_pay', 'affordability_status',
        'recommended_payment_method', 'payment_plan',
        'earliest_date_for_full_payment', 'spending_changes_needed',
        'decision_explanation']

STATUSES = {'affordable_now', 'affordable_with_plan', 'affordable_later', 'not_affordable'}
METHODS = {'full_payment', 'partial_payment', 'installments', 'wait', 'not_recommended'}
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
PLAN_ENTRY_RE = re.compile(r'^(\d{4}-\d{2}-\d{2}):(-?\d+(?:\.\d+)?)$')
CHANGE_RE = re.compile(r'^(stop:([^:|]+)|reduce_to:([^:|]+):(-?\d+(?:\.\d+)?))$')


def parse_plan(s):
    """Returns (list[(date_str, amount_float)], errors:list[str])."""
    errs = []
    if s is None or str(s).strip() == '' or str(s).strip().lower() == 'none':
        return [], errs
    parts = str(s).split('|')
    entries = []
    for p in parts:
        m = PLAN_ENTRY_RE.match(p.strip())
        if not m:
            errs.append(f'malformed payment_plan entry: {p!r}')
            continue
        entries.append((m.group(1), float(m.group(2))))
    dates = [pd.Timestamp(d) for d, _ in entries]
    if dates != sorted(dates):
        errs.append('payment_plan entries are not in chronological order')
    return entries, errs


def parse_changes(s):
    errs = []
    if s is None or str(s).strip() == '' or str(s).strip().lower() == 'none':
        return [], errs
    parts = str(s).split('|')
    if len(parts) > 3:
        errs.append('spending_changes_needed has more than 3 entries')
    out = []
    for p in parts:
        m = CHANGE_RE.match(p.strip())
        if not m:
            errs.append(f'malformed spending_changes_needed entry: {p!r}')
            continue
        if m.group(2):
            out.append(('stop', m.group(2)))
        else:
            out.append(('reduce_to', m.group(3), float(m.group(4))))
    stop_ids = {e[1] for e in out if e[0] == 'stop'}
    reduce_ids = {e[1] for e in out if e[0] == 'reduce_to'}
    if stop_ids & reduce_ids:
        errs.append('same event_id both stopped and reduced (mutually exclusive)')
    return out, errs


def main():
    req = pd.read_csv(DATA / 'requests.csv')
    out = pd.read_csv(OUT, keep_default_na=False)
    try:
        options = pd.read_csv(DATA / 'request_payment_options.csv')
    except FileNotFoundError:
        options = pd.DataFrame(columns=['request_id'])

    errors = []

    if list(out.columns) != COLS:
        errors.append(f'wrong output columns: {list(out.columns)}')
    if len(out) != len(req):
        errors.append(f'row count {len(out)} != {len(req)}')
    if set(out.request_id) != set(req.request_id):
        errors.append('request_id set mismatch')
    if out.request_id.duplicated().any():
        errors.append('duplicate request_id rows in output')

    req_by_id = req.set_index('request_id')
    opts_by_req = {rid: g for rid, g in options.groupby('request_id')} if 'request_id' in options.columns else {}

    for _, r in out.iterrows():
        rid = r.request_id
        if rid not in req_by_id.index:
            continue  # already reported as a set mismatch
        request_row = req_by_id.loc[rid]
        requested_amount = float(request_row.requested_amount)
        request_date = str(request_row.request_date)
        desired_completion = pd.Timestamp(request_row.desired_completion_date)
        allows_partial = str(request_row.allows_partial_payment).strip().lower() in {'true', '1', 'yes'}

        # amount_safe_to_pay bounds
        try:
            a = float(r.amount_safe_to_pay)
            if a < -1e-6 or a > requested_amount + 1e-6:
                errors.append(f'{rid}: amount_safe_to_pay {a} out of [0, {requested_amount}]')
        except Exception:
            errors.append(f'{rid}: amount_safe_to_pay not numeric: {r.amount_safe_to_pay!r}')
            a = None

        status = r.affordability_status
        method = r.recommended_payment_method
        if status not in STATUSES:
            errors.append(f'{rid}: invalid affordability_status {status!r}')
        if method not in METHODS:
            errors.append(f'{rid}: invalid recommended_payment_method {method!r}')

        # earliest_date_for_full_payment format / affordable_now rule
        edfp = str(r.earliest_date_for_full_payment).strip()
        if edfp and not DATE_RE.match(edfp):
            errors.append(f'{rid}: earliest_date_for_full_payment not YYYY-MM-DD: {edfp!r}')
        if status == 'affordable_now':
            if edfp != request_date:
                errors.append(f'{rid}: affordable_now requires earliest_date_for_full_payment == request_date '
                               f'({edfp!r} != {request_date!r})')

        # payment_plan structure
        entries, plan_errs = parse_plan(r.payment_plan)
        for e in plan_errs:
            errors.append(f'{rid}: {e}')
        if method == 'not_recommended' and str(r.payment_plan).strip().lower() != 'none':
            errors.append(f'{rid}: not_recommended should have payment_plan "none"')
        if method in {'full_payment', 'partial_payment', 'installments'} and not entries:
            errors.append(f'{rid}: {method} has an empty/none payment_plan')

        # partial_payment specific rules
        if method == 'partial_payment':
            if status != 'affordable_with_plan':
                errors.append(f'{rid}: partial_payment must have affordability_status affordable_with_plan')
            if not allows_partial:
                errors.append(f'{rid}: partial_payment recommended but request does not allow_partial_payment')
            if a is not None and not (0 < a < requested_amount):
                errors.append(f'{rid}: partial_payment requires 0 < amount_safe_to_pay < requested_amount')
            if len(entries) != 2:
                errors.append(f'{rid}: partial_payment must have exactly 2 payments, got {len(entries)}')
            elif a is not None:
                first_amt, second_amt = entries[0][1], entries[1][1]
                if abs(first_amt - a) > 0.01:
                    errors.append(f'{rid}: partial_payment first payment {first_amt} != amount_safe_to_pay {a}')
                if abs((first_amt + second_amt) - requested_amount) > 0.01:
                    errors.append(f'{rid}: partial_payment payments sum to {first_amt + second_amt} != requested_amount {requested_amount}')
                if entries[0][0] != request_date:
                    errors.append(f'{rid}: partial_payment first payment date {entries[0][0]} != request_date {request_date}')
                if edfp and entries[1][0] != edfp:
                    errors.append(f'{rid}: partial_payment second payment date {entries[1][0]} != earliest_date_for_full_payment {edfp}')
                if edfp and pd.Timestamp(edfp) > desired_completion:
                    errors.append(f'{rid}: partial_payment earliest_date_for_full_payment after desired_completion_date')

        # installments must exactly match a supplied option
        if method == 'installments':
            plan_str = str(r.payment_plan).strip()
            row_opts = opts_by_req.get(rid, pd.DataFrame())
            matched = False
            for _, o in row_opts.iterrows():
                if str(o.get('payment_method', '')) != 'installments':
                    continue
                try:
                    n = int(o.number_of_payments)
                    freq = int(o.payment_frequency_days)
                    first = pd.Timestamp(o.first_payment_date)
                    amt = float(o.payment_amount)
                    expected = '|'.join(
                        f"{(first + pd.Timedelta(days=i*freq)).strftime('%Y-%m-%d')}:{amt:.2f}".rstrip('0').rstrip('.')
                        for i in range(n)
                    )
                except Exception:
                    continue
                if plan_str == expected or plan_str.replace('.00', '') == expected.replace('.00', ''):
                    matched = True
                    break
            if not matched:
                errors.append(f'{rid}: installments payment_plan does not match any supplied payment option')

        # spending_changes_needed structure
        changes, change_errs = parse_changes(r.spending_changes_needed)
        for e in change_errs:
            errors.append(f'{rid}: {e}')

        # decision_explanation presence
        if not str(r.decision_explanation).strip():
            errors.append(f'{rid}: empty decision_explanation')

    print(f'rows={len(out)} errors={len(errors)}')
    for e in errors[:50]:
        print('ERROR', e)
    if len(errors) > 50:
        print(f'... and {len(errors) - 50} more errors')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())