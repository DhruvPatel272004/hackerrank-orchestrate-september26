#!/usr/bin/env python
"""Buy or Wait? deterministic financial decision agent.

Run from repository root:
    python code/main.py

The implementation intentionally keeps arithmetic and safety decisions
outside the language model: evidence extraction is heuristic/OCR based and
all final decisions are verified by a deterministic 90-day cash-flow engine.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from itertools import combinations
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import pytesseract
    from PIL import Image
except Exception:  # OCR is optional; structured data remains usable.
    pytesseract = None
    Image = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
MEDIA = DATA / "media" / "images"
OUT = ROOT / "output.csv"

OUTPUT_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
]

ACTIVE_STATUSES = {"pending", "scheduled", "settled"}
BAD_STATUSES = {"cancelled", "failed", "unrealized"}

# Status priority used when collapsing an event lifecycle chain
# (linked_event_id) down to a single canonical row. Higher wins.
LIFECYCLE_PRIORITY = {
    "settled": 5,
    "pending": 4,
    "scheduled": 3,
    "estimate": 2,
    "forecast": 1,
    "cancelled": 0,
    "failed": 0,
}


def clean(x):
    if pd.isna(x):
        return None
    return x


def money(x: float) -> float:
    # Stable CSV representation; avoid scientific notation.
    return round(float(x) + 1e-9, 2)


def money_text(x: float) -> str:
    return f"{money(x):.2f}".rstrip("0").rstrip(".")


def parse_bool(x) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def split_pipe(x) -> set[str]:
    if x is None or pd.isna(x) or not str(x).strip():
        return set()
    return {p.strip() for p in str(x).split("|") if p.strip()}


def parse_date(x) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def add_months(d: pd.Timestamp, n: int = 1) -> pd.Timestamp:
    return d + pd.DateOffset(months=n)


@dataclass
class CashFlow:
    dt: pd.Timestamp
    amount: float
    direction: str
    category: str
    event_id: str
    source: str
    flexibility: str = "fixed"
    minimum_allowed_amount: Optional[float] = None


@dataclass
class Stream:
    user_id: str
    category: str
    description: str
    event_type: str
    direction: str
    flexibility: str
    minimum_allowed_amount: Optional[float]
    interval_days: int
    amount: float
    last_date: pd.Timestamp
    representative_event_id: str
    dates: List[pd.Timestamp]


class Agent:
    def __init__(self):
        self.requests = pd.read_csv(DATA / "requests.csv")
        self.profiles = pd.read_csv(DATA / "financial_profiles.csv")
        self.events = pd.read_csv(DATA / "financial_events.csv")
        self.options = pd.read_csv(DATA / "request_payment_options.csv")
        self.messages = pd.read_csv(DATA / "messages.csv")
        self.images = pd.read_csv(DATA / "images.csv")
        self.rates = pd.read_csv(DATA / "exchange_rates.csv")

        self.events["event_date"] = pd.to_datetime(self.events["event_date"], errors="coerce")
        self.events["settlement_date"] = pd.to_datetime(self.events["settlement_date"], errors="coerce")
        self.rates["rate_date"] = pd.to_datetime(self.rates["rate_date"], errors="coerce")
        self.messages["sent_at_dt"] = pd.to_datetime(self.messages["sent_at"], errors="coerce")

        self.profile_by_user = self.profiles.set_index("user_id").to_dict("index")
        self.events_by_user = {u: g.copy() for u, g in self.events.groupby("user_id")}
        self.options_by_request = {u: g.copy() for u, g in self.options.groupby("request_id")}
        self.messages_by_user = {u: g.copy() for u, g in self.messages.groupby("user_id")}

        # Request-linked evidence. Guarded by column presence rather than
        # assumed, since the exact messages.csv/images.csv schema wasn't
        # confirmed -- if `request_id` isn't a column, these are just empty
        # and every other code path is unaffected.
        self.messages_by_request: Dict[str, List[dict]] = {}
        if "request_id" in self.messages.columns:
            for _, row in self.messages.iterrows():
                rid = clean(row.get("request_id"))
                if rid is not None:
                    self.messages_by_request.setdefault(str(rid), []).append(row.to_dict())

        self.image_by_event = {}
        self.image_by_request: Dict[str, List[str]] = {}
        has_image_request_id = "request_id" in self.images.columns
        for _, row in self.images.iterrows():
            if pd.notna(row.get("related_event_id")):
                self.image_by_event[str(row.related_event_id)] = str(row.image_id)
            if has_image_request_id:
                rid = clean(row.get("request_id"))
                if rid is not None:
                    self.image_by_request.setdefault(str(rid), []).append(str(row.image_id))

        self.ocr_cache: Dict[str, Optional[float]] = {}
        self.stream_cache: Dict[Tuple[str, str], List[Stream]] = {}
        self.flow_cache: Dict[Tuple[str, str, str], List[CashFlow]] = {}
        self.income_stream_cache: Dict[Tuple[str, str], List[Stream]] = {}
        self.canonical_events_cache: Dict[str, pd.DataFrame] = {}

    # ---------- Lifecycle canonicalization ----------
    def canonicalize_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Collapse an event lifecycle chain (linked via `linked_event_id`)
        down to a single canonical row per real-world transaction, so a
        pending record and its later settled counterpart are never both
        counted as separate cash flows.

        Rules (per problem statement's conflict-resolution order):
        - records linked (directly or transitively) by linked_event_id are
          one group, regardless of which row points to which
        - a group that is entirely cancelled/failed is dropped
        - unrealized-investment rows are excluded from consideration
        - within the remaining rows, settled beats pending/scheduled beats
          estimate/forecast; ties broken by the later settlement/event date
        - rows with no event_id, or that never link to anything, pass
          through untouched as singleton groups
        """
        if events.empty or "event_id" not in events.columns:
            return events

        records = events.to_dict("records")
        by_id = {
            str(r["event_id"]): r
            for r in records
            if pd.notna(r.get("event_id"))
        }
        if not by_id:
            return events

        # Union-find over the linked_event_id graph so that transitively
        # linked records (A -> B, C -> B) end up in one group together.
        parent = {eid: eid for eid in by_id}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for eid, r in by_id.items():
            linked = r.get("linked_event_id")
            if pd.notna(linked) and str(linked) in by_id:
                union(eid, str(linked))

        groups: Dict[str, List[dict]] = {}
        for eid in by_id:
            groups.setdefault(find(eid), []).append(by_id[eid])

        def sort_key(r: dict):
            status = str(r.get("status", "")).lower()
            settle = r.get("settlement_date")
            evdate = r.get("event_date")
            dt = settle if pd.notna(settle) else (evdate if pd.notna(evdate) else pd.Timestamp.min)
            return (LIFECYCLE_PRIORITY.get(status, 0), dt, str(r.get("event_id")))

        canonical_rows: List[dict] = []
        for rows in groups.values():
            statuses = {str(r.get("status", "")).lower() for r in rows}
            if statuses and statuses.issubset({"cancelled", "failed"}):
                continue
            valid = [
                r for r in rows
                if str(r.get("status", "")).lower() not in {"cancelled", "failed", "unrealized"}
            ]
            if not valid:
                continue
            valid.sort(key=sort_key, reverse=True)
            canonical_rows.append(valid[0])

        # Rows without an event_id can't be lifecycle-linked to anything;
        # keep them as-is rather than silently dropping them.
        orphans = [r for r in records if pd.isna(r.get("event_id"))]
        canonical_rows.extend(orphans)

        if not canonical_rows:
            return events.iloc[0:0]
        return pd.DataFrame(canonical_rows, columns=events.columns)

    def canonical_events_for(self, user: str) -> pd.DataFrame:
        if user in self.canonical_events_cache:
            return self.canonical_events_cache[user]
        result = self.canonicalize_events(self.events_by_user.get(user, pd.DataFrame()))
        self.canonical_events_cache[user] = result
        return result

    # ---------- Request-linked evidence ----------
    def request_messages(self, request_id) -> List[dict]:
        return self.messages_by_request.get(str(request_id), [])

    def request_message_text(self, request_id) -> str:
        """Concatenated text of every message tied directly to this request
        (not to a specific event). Available as an evidence source; the
        problem statement requires messages to only clarify/amend/confirm
        supplied financial facts, not to freely override the arithmetic, so
        this is exposed for narrow, explicit-pattern extraction rather than
        wired into amount/date logic without a confirmed extraction rule."""
        return " ".join(
            str(m.get("message_text", "")).strip()
            for m in self.request_messages(request_id)
            if str(m.get("message_text", "")).strip()
        )

    def request_images(self, request_id) -> List[str]:
        return self.image_by_request.get(str(request_id), [])

    def request_image_amount(self, request_id, currency: str) -> Optional[float]:
        """Largest OCR amount found across images linked directly to this
        request. Exposed as an evidence source only -- NOT used to override
        `requested_amount`, since the problem statement treats the CSV
        request amount as authoritative and only calls for image OCR to
        fill in a blank *event* amount, not to amend a request amount."""
        amounts = [
            a for iid in self.request_images(request_id)
            if (a := self.ocr_amount(iid, currency)) is not None and a > 0
        ]
        return max(amounts) if amounts else None

    # ---------- Evidence / normalization ----------
    def rate(self, dt: pd.Timestamp, src: str, dst: str) -> float:
        if src == dst:
            return 1.0
        r = self.rates[
            (self.rates.from_currency == src)
            & (self.rates.to_currency == dst)
            & (self.rates.rate_date == dt.normalize())
        ]
        if not r.empty:
            return float(r.iloc[0].rate)
        # The dataset normally supplies the exact date. For a missing direct
        # quote, try a deterministic same-date two-leg route through USD/EUR.
        for mid in ("USD", "EUR"):
            if mid in (src, dst):
                continue
            a = self.rates[(self.rates.from_currency == src) & (self.rates.to_currency == mid) & (self.rates.rate_date == dt.normalize())]
            b = self.rates[(self.rates.from_currency == mid) & (self.rates.to_currency == dst) & (self.rates.rate_date == dt.normalize())]
            if not a.empty and not b.empty:
                return float(a.iloc[0].rate) * float(b.iloc[0].rate)
        raise ValueError(f"No exchange rate for {src}->{dst} on {dt.date()}")

    def ocr_amount(self, image_id: str, currency: str) -> Optional[float]:
        if image_id in self.ocr_cache:
            return self.ocr_cache[image_id]
        if pytesseract is None or Image is None:
            self.ocr_cache[image_id] = None
            return None
        path = MEDIA / f"{image_id}.png"
        if not path.exists():
            self.ocr_cache[image_id] = None
            return None
        try:
            txt = pytesseract.image_to_string(Image.open(path))
        except Exception:
            self.ocr_cache[image_id] = None
            return None

        # Priority 1: an explicit "amount in words" line. Receipts/invoices in
        # this dataset sometimes spell the total out in words (e.g. "Indian
        # Rupee Seventy-Nine Thousand Six Hundred Seventy-Nine and Twenty-Six
        # Paise Only"). This is far more reliable than picking a number out of
        # a noisy OCR'd table, because it survives column/line misalignment.
        words_val = self.words_amount(txt)
        if words_val is not None:
            self.ocr_cache[image_id] = words_val
            return words_val

        # Priority 2: explicit totals/net pay/amount due/amount received on a
        # single line (label and number are on the same OCR line).
        lines = [x.strip() for x in txt.splitlines() if x.strip()]
        priority = [
            r"(?:grand\s+total|total\s+amount|amount\s+due|amount\s+received|net\s+pay|cash\s+paid|total\s*:)[^0-9]{0,30}([0-9][0-9,\. ]*)",
            r"(?:transferred\s+to|sum\s+of)[^0-9]{0,40}([0-9][0-9,\. ]*)",
        ]
        candidates: List[float] = []
        for line in lines:
            low = line.lower()
            for pat in priority:
                m = re.search(pat, low, re.I)
                if m:
                    v = self.parse_numeric(m.group(1))
                    if v is not None:
                        candidates.append(v)
        # Priority 3 (last resort, low confidence): a label keyword
        # ("total"/"balance due"/"amount") appears on its own line with the
        # number on a nearby line below it (common OCR table-splitting
        # artifact). Search only a small window after the keyword rather than
        # the whole document, and take the number closest to the keyword
        # rather than the single largest number anywhere in the receipt --
        # the largest raw number in a multi-item receipt is frequently a
        # mis-merged duplicate of a subtotal/tax column, not the true total.
        if not candidates:
            label_idx = [i for i, l in enumerate(lines) if re.search(r"\b(total|balance due|amount due)\b", l, re.I)]
            for i in label_idx:
                for j in range(i, min(i + 4, len(lines))):
                    v = self.parse_numeric(lines[j])
                    if v is not None and v > 0:
                        candidates.append(v)
                        break  # nearest number to this label only
        if not candidates:
            nums = []
            for m in re.finditer(r"(?<![A-Za-z])(?:\d{1,3}(?:[,\.]\d{3})+|\d+(?:\.\d+)?)", txt):
                v = self.parse_numeric(m.group(0))
                if v is not None and v > 0:
                    nums.append(v)
            if nums:
                candidates = nums
        val = max(candidates) if candidates else None
        self.ocr_cache[image_id] = val
        return val

    _ONES = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
             "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
             "seventeen":17,"eighteen":18,"nineteen":19}
    _TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
    _SCALES = {"hundred":100,"thousand":1000,"lakh":100000,"lac":100000,"million":1000000,"crore":10000000}

    @classmethod
    def _words_to_int(cls, phrase: str) -> Optional[int]:
        tokens = re.findall(r"[a-z]+", phrase.lower())
        if not tokens:
            return None
        total, current = 0, 0
        seen = False
        for tok in tokens:
            if tok in cls._ONES:
                current += cls._ONES[tok]; seen = True
            elif tok in cls._TENS:
                current += cls._TENS[tok]; seen = True
            elif tok == "hundred":
                current = (current or 1) * 100; seen = True
            elif tok in cls._SCALES:
                scale = cls._SCALES[tok]
                total += (current or 1) * scale
                current = 0
                seen = True
            elif tok == "and":
                continue
            else:
                return None  # unrecognized word: bail out, don't guess
        total += current
        return total if seen else None

    @classmethod
    def words_amount(cls, txt: str) -> Optional[float]:
        m = re.search(r"(?:amount|total)\s+in\s+words[:\s]*([A-Za-z ,\-]+?)(?:only|\n\n|$)", txt, re.I)
        if not m:
            return None
        phrase = m.group(1)
        # Split off a trailing "... and <sub-unit> paise/cents" fraction.
        frac_m = re.search(r"\band\s+([A-Za-z\- ]+?)\s+(?:paise|cents)\b", phrase, re.I)
        major_phrase = phrase[: frac_m.start()] if frac_m else phrase
        # Strip a leading currency name (e.g. "Indian Rupee", "US Dollar").
        major_phrase = re.sub(r"^\s*[A-Za-z]+(?:\s+[A-Za-z]+)?\s+(?=[A-Za-z]+\s)", "", major_phrase, count=1)
        major = cls._words_to_int(major_phrase)
        if major is None:
            return None
        minor = cls._words_to_int(frac_m.group(1)) if frac_m else 0
        return float(major) + float(minor or 0) / 100.0

    @staticmethod
    def parse_numeric(s: str) -> Optional[float]:
        s = s.replace(" ", "").strip()
        if not s:
            return None
        # Indian/international thousands separators and decimal commas.
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif s.count(",") >= 1 and s.count(".") == 0:
            parts = s.split(",")
            if all(len(p) == 3 for p in parts[1:]):
                s = "".join(parts)
            else:
                s = ".".join(parts)
        elif s.count(".") > 1:
            s = s.replace(".", "")
        try:
            return float(s)
        except ValueError:
            return None

    def event_amount_home(self, row: pd.Series, home: str) -> Optional[float]:
        if pd.notna(row.amount):
            val = float(row.amount)
        else:
            eid = str(row.event_id)
            iid = self.image_by_event.get(eid)
            if not iid:
                return None
            val = self.ocr_amount(iid, str(row.currency))
            if val is None:
                return None
        return val * self.rate(parse_date(row.settlement_date), str(row.currency), home)

    def apply_message_overrides(self, user: str, row: pd.Series) -> Tuple[Optional[float], pd.Timestamp]:
        """Apply only messages that explicitly amend a supplied event.

        Unattached messages can affect confirmed salary information, but they
        never create an income event unless a date and amount are explicit.

        Returns (amount, date). `amount` is None when the underlying event
        amount could not be determined at all (blank amount, no usable
        image/OCR) -- this is missing evidence, not a zero-value transaction,
        and callers must treat it as "skip", never as "$0 happened".
        """
        amt = self.event_amount_home(row, self.profile_by_user[user]["home_currency"])
        dt = parse_date(row.settlement_date)
        if amt is None:
            return None, dt
        msgs = self.messages_by_user.get(user)
        if msgs is None:
            return amt, dt
        related = msgs[msgs.related_event_id.astype(str) == str(row.event_id)]
        for _, m in related.sort_values("sent_at_dt").iterrows():
            text = str(m.message_text)
            # Explicit date amendments such as "expected on 2024-09-23".
            dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
            if dates and any(k in text.lower() for k in ["date", "payroll", "salary", "payment"]):
                try:
                    dt = parse_date(dates[-1])
                except Exception:
                    pass
            # Explicit numeric salary/amount amendment.
            if row.event_type == "income":
                nums = re.findall(r"(?<![A-Za-z])\d[\d,\.]{3,}", text)
                vals = [self.parse_numeric(x) for x in nums]
                vals = [v for v in vals if v and v > 100]
                if vals:
                    # Message amounts are generally in the event currency.
                    # Convert at the *effective* date `dt` (which may already
                    # have been shifted by a date-amendment above), not the
                    # original settlement_date -- the exchange rate is
                    # date-specific, so using a stale date here can silently
                    # apply the wrong day's rate after a date correction.
                    amt = max(vals) * self.rate(dt, str(row.currency), self.profile_by_user[user]["home_currency"])
        return amt, dt

    # ---------- Recurrence ----------
    def build_streams(self, user: str, asof: pd.Timestamp) -> List[Stream]:
        cache_key = (user, asof.strftime("%Y-%m-%d"))
        if cache_key in self.stream_cache:
            return self.stream_cache[cache_key]
        # Use canonical (lifecycle-deduplicated) events: without this, a
        # settled record that amends/corrects an earlier settled record for
        # the same event chain would be counted as a second, separate
        # occurrence and could inflate recurrence detection.
        d = self.canonical_events_for(user).copy()
        if d.empty:
            return []
        d = d[(d.event_date < asof) & (d.status == "settled") & (d.direction == "debit") & d.amount.notna()]
        streams: List[Stream] = []
        for (cat, desc, etype, direction), g in d.groupby(["category", "description", "event_type", "direction"]):
            g = g.sort_values("event_date")
            if len(g) < 3:
                continue
            dates = list(g.event_date.drop_duplicates())
            if len(dates) < 3:
                continue
            diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            med = float(pd.Series(diffs).median())
            close = sum(abs(x - med) <= max(2, med * 0.08) for x in diffs) / len(diffs)
            if med < 7 or med > 45 or close < 0.70:
                continue
            # Strong recurrence signal: repeated stream or explicit subscription/debt.
            if len(g) < 4 and etype not in {"subscription", "debt_payment"}:
                continue
            flex = str(g.flexibility.dropna().iloc[-1]) if g.flexibility.notna().any() else "fixed"
            mins = g.minimum_allowed_amount.dropna()
            min_allowed = float(mins.iloc[-1]) if not mins.empty else None
            amount = float(g.amount.tail(min(5, len(g))).median())
            streams.append(Stream(
                user_id=user, category=str(cat), description=str(desc), event_type=str(etype),
                direction=direction, flexibility=flex, minimum_allowed_amount=min_allowed,
                interval_days=max(1, int(round(med))), amount=amount, last_date=dates[-1],
                representative_event_id=str(g.iloc[-1].event_id), dates=dates,
            ))
        self.stream_cache[cache_key] = streams
        return streams

    def build_income_streams(self, user: str, asof: pd.Timestamp) -> List[Stream]:
        key = (user, asof.strftime("%Y-%m-%d"))
        if key in self.income_stream_cache:
            return self.income_stream_cache[key]
        d = self.canonical_events_for(user).copy()
        streams: List[Stream] = []
        if d.empty:
            return streams
        d = d[(d.event_date < asof) & (d.status == "settled") & (d.direction == "credit")]
        home = str(self.profile_by_user[user]["home_currency"])
        # Group salaries/recurring credits by description. Fill blank amounts
        # from the linked image when necessary.
        for (cat, desc, etype, direction), g in d.groupby(["category", "description", "event_type", "direction"]):
            if len(g) < 3:
                continue
            dates = list(g.event_date.drop_duplicates())
            if len(dates) < 3:
                continue
            diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            med = float(pd.Series(diffs).median())
            close = sum(abs(x - med) <= max(2, med * 0.08) for x in diffs) / len(diffs)
            if med < 7 or med > 45 or close < 0.70:
                continue
            vals = []
            for _, row in g.tail(5).iterrows():
                a = self.event_amount_home(row, home)
                if a is not None and a > 0:
                    vals.append(a)
            if not vals:
                continue
            amount = float(pd.Series(vals).median())
            streams.append(Stream(
                user_id=user, category=str(cat), description=str(desc), event_type=str(etype),
                direction="credit", flexibility="fixed", minimum_allowed_amount=None,
                interval_days=max(1, int(round(med))), amount=amount, last_date=dates[-1],
                representative_event_id=str(g.iloc[-1].event_id), dates=dates,
            ))
        self.income_stream_cache[key] = streams
        return streams

    def message_salary_override(self, user: str, dt: pd.Timestamp, default: float) -> float:
        msgs = self.messages_by_user.get(user)
        if msgs is None:
            return default
        best = default
        best_sent = None
        # NOTE: bare "pay" is intentionally excluded -- it false-positives on
        # any unrelated message like "I need to pay $500 for X".
        salary_keywords = [
            "salary", "payroll", "monthly salary", "monthly pay",
            "salary received", "salary credited", "salary payment",
            "wage", "wages", "gaji",
        ]
        for _, m in msgs.iterrows():
            text = str(m.message_text)
            low = text.lower()
            if not any(k in low for k in salary_keywords):
                continue
            nums = [self.parse_numeric(x) for x in re.findall(r"(?<![A-Za-z])\d[\d,\.]{3,}", text)]
            nums = [x for x in nums if x is not None and x > 100]
            if not nums:
                continue
            # Payroll messages usually contain one salient amount; use the
            # largest amount rather than reference numbers such as EMP-0001.
            val = max(nums)
            date_matches = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
            effective = None
            if date_matches:
                try:
                    effective = parse_date(date_matches[-1])
                except Exception:
                    pass
            if effective is not None and dt < effective:
                continue
            sent = m.sent_at_dt
            if best_sent is None or sent >= best_sent:
                # Message currency is normally the user's home currency for
                # payroll. Convert only when the text explicitly names USD/EUR/INR/ZAR/IDR.
                cur = next((c for c in ["USD", "EUR", "INR", "ZAR", "IDR"] if c in text.upper()), str(self.profile_by_user[user]["home_currency"]))
                try:
                    val = val * self.rate(dt, cur, str(self.profile_by_user[user]["home_currency"]))
                except Exception:
                    pass
                best, best_sent = val, sent
        return best

    def build_variable_streams(self, user: str, asof: pd.Timestamp) -> List[Stream]:
        key=(user,asof.strftime("%Y-%m-%d"))
        d=self.canonical_events_for(user).copy()
        out=[]
        if d.empty: return out
        d=d[(d.event_date<asof)&(d.status=="settled")&(d.direction=="debit")&d.amount.notna()&d.category.isin(["groceries","transport"])]
        for cat,g in d.groupby("category"):
            dates=sorted(g.event_date.drop_duplicates())
            if len(dates)<6: continue
            diffs=[(dates[i]-dates[i-1]).days for i in range(1,len(dates))]
            med=float(pd.Series(diffs).median())
            close=sum(abs(x-med)<=max(2,med*.08) for x in diffs)/len(diffs)
            if not (5<=med<=35 and close>=.75): continue
            amount=float(g.amount.tail(min(12,len(g))).median())
            out.append(Stream(user,cat,"__variable_category__","expense","debit","fixed",None,max(1,int(round(med))),amount,dates[-1],f"__var__{cat}",dates))
        return out

    def next_dates(self, stream: Stream, start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
        cur = stream.last_date
        # Monthly-looking cadence should preserve calendar day rather than
        # drift by 30/31-day arithmetic.
        monthly = 26 <= stream.interval_days <= 35
        while True:
            nxt = add_months(cur) if monthly else cur + pd.Timedelta(days=stream.interval_days)
            if nxt > end:
                break
            cur = nxt
            if cur >= start:
                yield cur

    def baseline_flows(self, user: str, start: pd.Timestamp, end: pd.Timestamp) -> List[CashFlow]:
        home = str(self.profile_by_user[user]["home_currency"])
        out: List[CashFlow] = []
        # Collapse lifecycle chains (pending -> settled, etc.) before doing
        # any cash-flow accounting, so the same real-world transaction is
        # never counted twice under two different event_id rows.
        d = self.canonical_events_for(user)
        if not d.empty:
            f = d[d.settlement_date.notna() & (d.settlement_date >= start) & (d.settlement_date <= end)]
            for _, row in f.iterrows():
                status = str(row.status)
                if status not in ACTIVE_STATUSES:
                    continue
                # Pending credits are explicitly ignored. Pending debits reserve cash.
                if str(row.direction) == "credit" and status == "pending":
                    continue
                # Unrealized investments never reach this branch due to status filter.
                amt, dt = self.apply_message_overrides(user, row)
                if amt is None or amt <= 0:
                    continue
                out.append(CashFlow(dt, amt, str(row.direction), str(row.category), str(row.event_id), "event", str(row.flexibility), clean(row.minimum_allowed_amount)))

        # Forecast recurring income streams (especially monthly salary).
        for s in self.build_income_streams(user, start):
            for dt in self.next_dates(s, start, end):
                amount = self.message_salary_override(user, dt, s.amount) if s.category == "salary" else s.amount
                out.append(CashFlow(dt, amount, "credit", s.category, s.representative_event_id, "recurring_income", "fixed", None))

        # Forecast recurring expense streams only when history strongly supports them.
        for s in self.build_streams(user, start):
            for dt in self.next_dates(s, start, end):
                out.append(CashFlow(dt, s.amount, "debit", s.category, s.representative_event_id, "recurring", s.flexibility, s.minimum_allowed_amount))
        # Essential variable spending is forecast conservatively at the
        # category level rather than treating every grocery/transport receipt
        # as a separate recurring subscription.
        for s in self.build_variable_streams(user, start):
            for dt in self.next_dates(s, start, end):
                out.append(CashFlow(dt, s.amount, "debit", s.category, s.representative_event_id, "variable_recurring", s.flexibility, None))
        return out

    # ---------- Forecast / safety ----------
    def apply_flows(self, profile: dict, start: pd.Timestamp, end: pd.Timestamp, extra: List[CashFlow] | None = None,
                    changes: Optional[List[Tuple[str, str, float]]] = None) -> Tuple[float, Dict[pd.Timestamp, float], List[CashFlow]]:
        user = profile["user_id"]
        key = (user, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if extra is None and not changes and key in self.flow_cache:
            flows = list(self.flow_cache[key])
        else:
            flows = self.baseline_flows(user, start, end)
            if extra:
                flows.extend(extra)
            changes = changes or []
            changed_ids = {eid: (kind, value) for eid, kind, value in changes}
            adjusted = []
            for f in flows:
                if f.source == "recurring" and f.event_id in changed_ids:
                    kind, value = changed_ids[f.event_id]
                    if kind == "stop":
                        continue
                    if kind == "reduce":
                        f = CashFlow(f.dt, min(f.amount, value), f.direction, f.category, f.event_id, f.source, f.flexibility, f.minimum_allowed_amount)
                adjusted.append(f)
            flows = adjusted
            if extra is None and not changes:
                self.flow_cache[key] = list(flows)
        flows.sort(key=lambda x: (x.dt, 0 if x.direction == "debit" else 1))
        balance = float(profile["current_available_balance"])
        minimum = float(profile["minimum_balance_to_keep"])
        min_balance = balance
        by_day: Dict[pd.Timestamp, float] = {}
        for f in flows:
            if f.dt < start or f.dt > end:
                continue
            balance += f.amount if f.direction == "credit" else -f.amount
            by_day[f.dt] = balance
            min_balance = min(min_balance, balance)
        return min_balance, by_day, flows

    def forecast_state(self, profile: dict, request: pd.Series, changes=None):
        start = parse_date(request.request_date); end = start + pd.Timedelta(days=90)
        _, by_day, flows = self.apply_flows(profile, start, end, changes=changes)
        # Balance immediately after each day's baseline events. Also build a
        # suffix minimum, allowing O(days) earliest-date checks.
        days = [start + pd.Timedelta(days=i) for i in range(91)]
        bal = float(profile["current_available_balance"])
        daily = {}
        grouped = {}
        for f in flows:
            grouped.setdefault(f.dt, []).append(f)
        for d in days:
            for f in grouped.get(d, []):
                bal += f.amount if f.direction == "credit" else -f.amount
            daily[d] = bal
        suffix = {}
        m = float("inf")
        for d in reversed(days):
            m = min(m, daily[d])
            suffix[d] = m
        return daily, suffix

    def safe_today(self, profile: dict, request: pd.Series, changes=None) -> float:
        start = parse_date(request.request_date)
        daily, suffix = self.forecast_state(profile, request, changes)
        minimum = float(profile["minimum_balance_to_keep"])
        # Payment today is made against the post-request-date baseline balance.
        headroom = suffix[start] - minimum
        return max(0.0, min(float(request.requested_amount), headroom))

    def full_payment_safe(self, profile: dict, request: pd.Series, pay_date: pd.Timestamp, changes=None) -> bool:
        start = parse_date(request.request_date)
        amount = float(request.requested_amount)
        daily, suffix = self.forecast_state(profile, request, changes)
        minimum = float(profile["minimum_balance_to_keep"])
        # Paying after the day's baseline events is conservative and avoids
        # treating a same-day scheduled credit as unavailable.
        return pay_date in daily and daily[pay_date] - amount >= minimum - 1e-7 and suffix[pay_date] - amount >= minimum - 1e-7

    def earliest_full(self, profile: dict, request: pd.Series, changes=None) -> Optional[pd.Timestamp]:
        start = parse_date(request.request_date)
        deadline = min(parse_date(request.desired_completion_date), start + pd.Timedelta(days=90))
        daily, suffix = self.forecast_state(profile, request, changes)
        minimum = float(profile["minimum_balance_to_keep"])
        amount = float(request.requested_amount)
        d = start
        while d <= deadline:
            if daily[d] - amount >= minimum - 1e-7 and suffix[d] - amount >= minimum - 1e-7:
                return d
            d += pd.Timedelta(days=1)
        return None

    # ---------- Payment plans ----------
    def option_plan(self, profile: dict, request: pd.Series, option: pd.Series) -> List[CashFlow]:
        # Payment option amounts are already in home currency.
        first = parse_date(option.first_payment_date)
        n = int(option.number_of_payments)
        if n <= 0:
            return []
        freq_valid = pd.notna(option.payment_frequency_days) and int(option.payment_frequency_days) > 0
        if n > 1 and not freq_valid:
            # A multi-payment schedule with no usable frequency would stack
            # every payment on the same day -- that's malformed option data,
            # not a real schedule, so treat this option as unusable rather
            # than silently fabricating a same-day plan.
            return []
        freq = int(option.payment_frequency_days) if pd.notna(option.payment_frequency_days) else 0
        amount = float(option.payment_amount)
        flows = []
        for i in range(n):
            dt = first + pd.Timedelta(days=i * freq)
            flows.append(CashFlow(dt, amount, "debit", str(request.request_type), str(option.payment_option_id), "payment_option"))
        return flows

    def plan_safe(self, profile: dict, request: pd.Series, flows: List[CashFlow], changes=None) -> bool:
        start = parse_date(request.request_date); end = start + pd.Timedelta(days=90)
        deadline = parse_date(request.desired_completion_date)
        if any(f.dt < start or f.dt > deadline for f in flows):
            return False
        minb, _, _ = self.apply_flows(profile, start, end, extra=flows, changes=changes)
        return minb + 1e-7 >= float(profile["minimum_balance_to_keep"])

    def flexible_changes(self, profile: dict, request: pd.Series) -> List[Tuple[str, str, float]]:
        """Find all legal changes to flexible recurring streams, ranked by
        estimated monthly cash impact. The caller decides how many of these
        (0-3) to actually combine -- this must NOT pre-truncate to 3, since
        the best <=3-change combination is not necessarily the 3 highest-
        impact candidates individually."""
        user = profile["user_id"]
        protected = split_pipe(profile.get("expense_categories_to_protect"))
        reduce_cats = split_pipe(profile.get("expense_categories_user_is_willing_to_reduce"))
        stop_cats = split_pipe(profile.get("expense_categories_user_is_willing_to_stop"))
        streams = self.build_streams(user, parse_date(request.request_date))
        candidates = []
        for s in streams:
            if s.category in protected:
                continue
            flex = s.flexibility
            if flex in {"stoppable", "reducible_or_stoppable"} and s.category in stop_cats:
                candidates.append((s.amount * 4, (s.representative_event_id, "stop", 0.0)))
            if flex in {"reducible", "reducible_or_stoppable"} and s.category in reduce_cats and s.minimum_allowed_amount is not None and s.minimum_allowed_amount < s.amount:
                candidates.append(((s.amount - s.minimum_allowed_amount) * 4, (s.representative_event_id, "reduce", float(s.minimum_allowed_amount))))
        candidates.sort(reverse=True, key=lambda x: (x[0], x[1][0]))
        # Mutually exclusive stop/reduce per event; return every legal
        # candidate -- do NOT cap here, the combination search needs the
        # full set to find the true best <=3-change plan.
        return [x[1] for x in candidates]

    @staticmethod
    def _valid_change_set(changes: List[Tuple[str, str, float]]) -> bool:
        """Reject any combination that both stops and reduces the same
        event_id. flexible_changes() can legally emit both a stop candidate
        and a reduce candidate for the same stream (e.g. a category that's
        in both the user's willing-to-reduce and willing-to-stop lists), so
        this must be enforced at combination-selection time, not assumed
        away by candidate construction."""
        event_ids = [c[0] for c in changes]
        return len(event_ids) == len(set(event_ids))

    def search_changes_for_full(self, profile: dict, request: pd.Series) -> Tuple[Optional[List[Tuple[str, str, float]]], Optional[pd.Timestamp]]:
        cands = self.flexible_changes(profile, request)
        tests: List[Tuple[Tuple[str, str, float], ...]] = [()]
        for size in range(1, min(3, len(cands)) + 1):
            tests.extend(combinations(cands, size))
        best = None
        best_date = None
        for combo in tests:
            changes = list(combo)
            if not self._valid_change_set(changes):
                continue
            d = self.earliest_full(profile, request, changes=changes)
            if d is not None:
                if best is None or len(changes) < len(best):
                    best, best_date = changes, d
                elif len(changes) == len(best) and d < best_date:
                    best, best_date = changes, d
        return best, best_date

    def user_methods(self, profile: dict) -> set[str]:
        return split_pipe(profile.get("payment_methods_user_will_consider"))

    def format_plan(self, flows: List[CashFlow]) -> str:
        flows = sorted(flows, key=lambda x: x.dt)
        return "|".join(f"{f.dt.strftime('%Y-%m-%d')}:{money_text(f.amount)}" for f in flows)

    def format_changes(self, changes: Optional[List[Tuple[str, str, float]]]) -> str:
        if not changes:
            return "none"
        out = []
        for eid, kind, val in changes:
            out.append(f"stop:{eid}" if kind == "stop" else f"reduce_to:{eid}:{money_text(val)}")
        return "|".join(out)

    # ---------- Decision ----------
    def decide(self, request: pd.Series) -> dict:
        uid = str(request.user_id)
        profile = dict(self.profile_by_user[uid])
        profile["user_id"] = uid
        amount = float(request.requested_amount)
        start = parse_date(request.request_date)
        deadline = parse_date(request.desired_completion_date)
        methods = self.user_methods(profile)
        partial_allowed = parse_bool(request.allows_partial_payment) and "partial_payment" in methods

        safe_now = money(self.safe_today(profile, request))
        # `earliest` is the no-spending-change earliest full-payment-safe
        # date. Per the output contract this is always what gets reported
        # in `earliest_date_for_full_payment`, regardless of which method
        # ends up being recommended.
        earliest = self.earliest_full(profile, request)
        changes, earliest_with_changes = self.search_changes_for_full(profile, request)

        # Build every eligible, individually-safe candidate plan. Each is
        # already guaranteed (by its own construction / plan_safe check) to
        # complete by `deadline` -- so ranking rule 1 ("completes by
        # deadline") is satisfied by inclusion in this list, not by a sort
        # key. Rules 2-6 are then applied as a single sort below.
        candidates = []

        # (a) Full payment today, no spending changes.
        if "full_payment" in methods and safe_now + 1e-7 >= amount and self.full_payment_safe(profile, request, start):
            plan = [CashFlow(start, amount, "debit", str(request.request_type), "__purchase__", "purchase")]
            candidates.append({
                "requires_changes": False,
                "total_paid": amount,
                "start_date": start,
                "num_payments": 1,
                "option_rank": 0,
                "status": "affordable_now",
                "method": "full_payment",
                "plan": plan,
                "changes": None,
                "explanation": (
                    f"Pay {self.currency(profile)} {money_text(amount)} today. This keeps the "
                    f"{self.currency(profile)} {money_text(profile['minimum_balance_to_keep'])} minimum "
                    f"protected over the 90-day forecast."
                ),
            })

        # (b) Wait for full payment later, no spending changes.
        wait_eligible = "full_payment" in methods and earliest is not None and earliest <= deadline
        if wait_eligible and earliest > start:
            plan = [CashFlow(earliest, amount, "debit", str(request.request_type), "__purchase__", "purchase")]
            candidates.append({
                "requires_changes": False,
                "total_paid": amount,
                "start_date": earliest,
                "num_payments": 1,
                "option_rank": 0,
                "status": "affordable_later",
                "method": "wait",
                "plan": plan,
                "changes": None,
                "explanation": (
                    f"Wait until {earliest.strftime('%Y-%m-%d')} to pay {self.currency(profile)} "
                    f"{money_text(amount)} in full; paying earlier would risk the {self.currency(profile)} "
                    f"{money_text(profile['minimum_balance_to_keep'])} minimum."
                ),
            })

        # (c) Supplied installment options that are safe and within any
        # user-imposed max-duration limit. `plan_safe` already rejects any
        # option whose payments fall outside [request_date, deadline].
        if "installments" in methods:
            option_rows = self.options_by_request.get(str(request.request_id), pd.DataFrame())
            max_months = profile.get("max_installment_months")
            for _, o in option_rows.iterrows():
                if str(o.payment_method) != "installments":
                    continue
                if pd.notna(max_months):
                    duration_days = (int(o.number_of_payments) - 1) * int(o.payment_frequency_days or 0)
                    if duration_days > float(max_months) * 31.0:
                        continue
                flows = self.option_plan(profile, request, o)
                if not flows:
                    continue  # malformed option (see option_plan)
                # Sanity check only: the scheduled payments should never sum
                # to *more* than the option's claimed total_payable_amount --
                # that would be an internally inconsistent option row. We do
                # NOT require exact equality, since total_payable_amount may
                # legitimately include a financing fee on top of the raw
                # payment_amount * n (the schema documents "explicit
                # financing fees" as a separate concept from the schedule).
                expected_total = float(o.payment_amount) * int(o.number_of_payments)
                if expected_total > float(o.total_payable_amount) + 0.01:
                    continue
                if not self.plan_safe(profile, request, flows):
                    continue
                option_rank = int(re.sub(r"\D", "", str(o.payment_option_id)) or 10 ** 9)
                candidates.append({
                    "requires_changes": False,
                    "total_paid": float(o.total_payable_amount),
                    "start_date": parse_date(o.first_payment_date),
                    "num_payments": int(o.number_of_payments),
                    "option_rank": option_rank,
                    "status": "affordable_with_plan",
                    "method": "installments",
                    "plan": flows,
                    "changes": None,
                    "explanation": (
                        f"Use {int(o.number_of_payments)} installments of {self.currency(profile)} "
                        f"{money_text(o.payment_amount)}. The full schedule stays above the "
                        f"{self.currency(profile)} {money_text(profile['minimum_balance_to_keep'])} minimum "
                        f"and completes by the requested date."
                    ),
                })

        # (d) Full payment after permitted flexible-spending changes.
        if "full_payment" in methods and changes and earliest_with_changes is not None and earliest_with_changes <= deadline:
            plan = [CashFlow(earliest_with_changes, amount, "debit", str(request.request_type), "__purchase__", "purchase")]
            candidates.append({
                "requires_changes": True,
                "total_paid": amount,
                "start_date": earliest_with_changes,
                "num_payments": 1,
                "option_rank": 0,
                "status": "affordable_with_plan",
                "method": "full_payment",
                "plan": plan,
                "changes": changes,
                "explanation": (
                    f"Pay the full {self.currency(profile)} {money_text(amount)} on "
                    f"{earliest_with_changes.strftime('%Y-%m-%d')} after the permitted flexible-spending "
                    f"changes. This keeps the minimum balance protected."
                ),
            })

        # (e) Partial payment today + remainder once full payment is safe.
        if partial_allowed and 0 < safe_now < amount and earliest is not None and earliest <= deadline:
            second = money(amount - safe_now)
            plan = [
                CashFlow(start, safe_now, "debit", str(request.request_type), "__partial1__", "purchase"),
                CashFlow(earliest, second, "debit", str(request.request_type), "__partial2__", "purchase"),
            ]
            if self.plan_safe(profile, request, plan):
                candidates.append({
                    "requires_changes": False,
                    "total_paid": amount,
                    "start_date": start,
                    "num_payments": 2,
                    "option_rank": 0,
                    "status": "affordable_with_plan",
                    "method": "partial_payment",
                    "plan": plan,
                    "changes": None,
                    "explanation": (
                        f"Pay {self.currency(profile)} {money_text(safe_now)} today and the remaining "
                        f"{self.currency(profile)} {money_text(second)} on {earliest.strftime('%Y-%m-%d')}."
                    ),
                })

        if candidates:
            # Ranking rules 2-6, applied in one pass rather than as ad hoc
            # pairwise comparisons:
            #   2. no spending changes beats requiring changes
            #   3. minimize total amount paid
            #   4. start payment earlier
            #   5. fewer payments
            #   6. lowest payment_option_id (installments only; 0 elsewhere)
            candidates.sort(key=lambda c: (
                int(c["requires_changes"]),
                c["total_paid"],
                c["start_date"],
                c["num_payments"],
                c["option_rank"],
            ))
            best = candidates[0]
            return self.row(
                request, safe_now, best["status"], best["method"],
                self.format_plan(best["plan"]), earliest, best["changes"], best["explanation"],
            )

        # No safe, deadline-completing plan exists under any eligible method.
        explanation = (
            f"Do not make this payment by {deadline.strftime('%d %B %Y')}. No eligible payment option "
            f"keeps the {self.currency(profile)} {money_text(profile['minimum_balance_to_keep'])} minimum "
            f"protected through the 90-day forecast."
        )
        return self.row(request, safe_now, "not_affordable", "not_recommended", "none", earliest, None, explanation)

    def currency(self, profile):
        return str(profile["home_currency"])

    def row(self, request, safe, status, method, plan, earliest, changes, explanation):
        return {
            "request_id": request.request_id,
            "amount_safe_to_pay": money(max(0, min(float(request.requested_amount), float(safe)))),
            "affordability_status": status,
            "recommended_payment_method": method,
            "payment_plan": plan,
            "earliest_date_for_full_payment": "" if earliest is None else parse_date(earliest).strftime("%Y-%m-%d"),
            "spending_changes_needed": self.format_changes(changes),
            "decision_explanation": explanation,
        }

    def run(self) -> pd.DataFrame:
        rows = [self.decide(r) for _, r in self.requests.iterrows()]
        out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        # Hard validation required by the challenge contract.
        req = self.requests.set_index("request_id")
        for _, r in out.iterrows():
            request_row = req.loc[r.request_id]
            a = float(r.amount_safe_to_pay)
            assert 0 <= a <= float(request_row.requested_amount) + 1e-6
            assert r.affordability_status in {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
            assert r.recommended_payment_method in {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
            # Semantic invariants that decide()'s candidate construction
            # already guarantees -- kept as asserts to catch a future
            # regression before submission, not because they're expected to
            # ever fail against the current logic.
            if r.affordability_status == "affordable_now":
                assert r.recommended_payment_method == "full_payment"
                assert r.earliest_date_for_full_payment == parse_date(request_row.request_date).strftime("%Y-%m-%d")
            if r.affordability_status == "affordable_later":
                assert r.recommended_payment_method == "wait"
            if r.affordability_status == "not_affordable":
                assert r.recommended_payment_method == "not_recommended"
        out.to_csv(OUT, index=False)
        return out


if __name__ == "__main__":
    agent = Agent()
    result = agent.run()
    print(f"Wrote {len(result)} predictions to {OUT}")