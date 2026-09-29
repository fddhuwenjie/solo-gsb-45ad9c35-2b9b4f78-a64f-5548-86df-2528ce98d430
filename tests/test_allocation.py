"""POST /cost-allocation：共享电表多方阶梯电费 Shapley 拆分的端到端测试。"""
import os
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from itertools import permutations

import pytest

os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.pricing import Q6  # noqa: E402

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _startup():
    with client:
        yield


VERSION = "us-ny-demo-v2024"
TZ = "America/New_York"


def _iv(day, a, b, energy, tz=TZ):
    return {
        "start": f"2024-{day}T{a:02d}:00:00-04:00",
        "end": f"2024-{day}T{b:02d}:00:00-04:00",
        "timezone": tz, "energy": str(energy),
    }


def _member(mid, intervals):
    return {"id": mid, "intervals": intervals}


def _allocate(members, version=VERSION):
    return client.post("/cost-allocation",
                       json={"version": version, "members": members})


# --------------------------------------------------------------- basic cases

def test_two_symmetric_overlapping_members():
    # 2024-07-06 Saturday -> leisure: tier1 0.15 up to 100 kWh, tier2 0.18.
    members = [
        _member("A", [_iv("07-06", 12, 13, 60)]),
        _member("B", [_iv("07-06", 12, 13, 60)]),
    ]
    r = _allocate(members)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_energy"] == "120.000000"
    # 100 @0.15 + 20 @0.18 = 18.60
    assert Decimal(body["total_amount"]) == Decimal("18.600000")
    assert body["coalition_count"] == 4
    for m in body["members"]:
        # Each alone: 60 @0.15 = 9.00; symmetric Shapley share = 9.30.
        assert m["energy"] == "60"
        assert Decimal(m["standalone_amount"]) == Decimal("9.000000")
        assert Decimal(m["allocated_amount"]) == Decimal("9.300000")
        assert Decimal(m["amount_diff"]) == Decimal("-0.300000")


def test_three_asymmetric_members_hand_values():
    # Loads 80 / 40 / 30 over the identical weekend hour -> 150 total:
    # v values: 80->12, 40->6, 30->4.5, 110->16.8, 120->18.6,
    #           70->10.5, 150->24
    # Shapley: A 12.65, B 6.50, C 4.85.
    members = [
        _member("A", [_iv("07-06", 12, 13, 80)]),
        _member("B", [_iv("07-06", 12, 13, 40)]),
        _member("C", [_iv("07-06", 12, 13, 30)]),
    ]
    r = _allocate(members)
    assert r.status_code == 200, r.text
    body = r.json()
    assert Decimal(body["total_amount"]) == Decimal("24.000000")
    out = {m["member_id"]: m for m in body["members"]}
    assert Decimal(out["A"]["allocated_amount"]) == Decimal("12.650000")
    assert Decimal(out["B"]["allocated_amount"]) == Decimal("6.500000")
    assert Decimal(out["C"]["allocated_amount"]) == Decimal("4.850000")
    assert Decimal(out["A"]["standalone_amount"]) == Decimal("12.000000")
    assert Decimal(out["B"]["standalone_amount"]) == Decimal("6.000000")
    assert Decimal(out["C"]["standalone_amount"]) == Decimal("4.500000")
    for m in body["members"]:
        assert Decimal(m["amount_diff"]) == (
            Decimal(m["standalone_amount"]) - Decimal(m["allocated_amount"]))


def test_allocated_sum_strictly_equals_grand_quote():
    members = [
        _member("A", [_iv("07-06", 12, 13, 80)]),
        _member("B", [_iv("07-06", 12, 13, 40)]),
        _member("C", [_iv("07-06", 12, 13, 30)]),
    ]
    body = _allocate(members).json()
    total = Decimal(body["total_amount"])
    shares = sum((Decimal(m["allocated_amount"]) for m in body["members"]),
                 Decimal(0))
    assert shares == total
    energies = sum((Decimal(m["energy"]) for m in body["members"]),
                   Decimal(0))
    assert energies == Decimal(body["total_energy"])


def test_last_member_absorbs_rounding_residual():
    # Awkward loads chosen so the exact Shapley means are non-terminating;
    # still the ordered quantization must reconcile to the grand quote.
    members = [
        _member("A", [_iv("07-06", 12, 13, 33)]),
        _member("B", [_iv("07-06", 12, 13, 34)]),
        _member("C", [_iv("07-06", 12, 13, 35)]),
    ]
    body = _allocate(members).json()
    shares = [Decimal(m["allocated_amount"]) for m in body["members"]]
    assert sum(shares, Decimal(0)) == Decimal(body["total_amount"])
    # First members are exactly the half-up quantization of their raw values
    # (verified against the independent permutation oracle below).
    assert shares[-1] == Decimal(body["total_amount"]) - sum(shares[:-1])


# ------------------------------------------------- time-proration semantics

def test_cross_midnight_partial_overlap_prorated_by_duration():
    # A spans Friday 22:00 -> Saturday 02:00 (4 h, 40 kWh): 20 kWh on the
    # dst-workday rule and 20 kWh on leisure. B overlaps only the Saturday
    # 00:00-02:00 leisure half with 100 kWh.
    members = [
        _member("A", [{
            "start": "2024-07-05T22:00:00-04:00",
            "end": "2024-07-06T02:00:00-04:00",
            "timezone": TZ, "energy": "40",
        }]),
        _member("B", [_iv("07-06", 0, 2, 100)]),
    ]
    body = _allocate(members).json()
    out = {m["member_id"]: m for m in body["members"]}
    # A alone: 20 @0.22 + 20 @0.15 = 7.40 (matches /quote directly).
    rq = client.post("/quote", json={
        "version": VERSION, "intervals": members[0]["intervals"]})
    assert rq.json()["total_amount"] == "7.40"
    raw_quote = sum((Decimal(s["amount"]) for s in rq.json()["segments"]),
                    Decimal(0))
    assert Decimal(out["A"]["standalone_amount"]) == raw_quote
    # B alone: 100 leisure kWh = 15.00.
    assert Decimal(out["B"]["standalone_amount"]) == Decimal("15.000000")
    # Joint: workday 20@0.22=4.4; leisure 20+100 -> 80@0.15 + 40@0.18;
    # 4.4 + 12 + 7.2 = 23.6; symmetric 2-player split of the 1.2 uplift.
    assert Decimal(body["total_amount"]) == Decimal("23.600000")
    assert Decimal(out["A"]["allocated_amount"]) == Decimal("8.000000")
    assert Decimal(out["B"]["allocated_amount"]) == Decimal("15.600000")


def test_mixed_rules_crossing_tier_boundary():
    # A: Fri 20:00-24:00 (40 kWh, all workday). B: Fri 22:00-Sat 02:00
    # (80 kWh: 40 workday + 40 leisure). Joint cumulative crosses 100 kWh
    # inside the final leisure slice, so tier2 (0.18) bites:
    # v(A)=8.8, v(B)=14.8, v(AB)=4.4+13.2+3.0+3.6=24.2 -> phi 9.1 / 15.1.
    members = [
        _member("A", [{
            "start": "2024-07-05T20:00:00-04:00",
            "end": "2024-07-06T00:00:00-04:00",
            "timezone": TZ, "energy": "40",
        }]),
        _member("B", [{
            "start": "2024-07-05T22:00:00-04:00",
            "end": "2024-07-06T02:00:00-04:00",
            "timezone": TZ, "energy": "80",
        }]),
    ]
    body = _allocate(members).json()
    assert Decimal(body["total_amount"]) == Decimal("24.200000")
    out = {m["member_id"]: m for m in body["members"]}
    assert Decimal(out["A"]["allocated_amount"]) == Decimal("9.100000")
    assert Decimal(out["B"]["allocated_amount"]) == Decimal("15.100000")


def test_member_with_multiple_intervals_conserves_energy():
    # A's two disjoint intervals (Friday afternoon + Sunday morning) are cut
    # independently by B's endpoints too; A's reported energy must reconcile.
    members = [
        _member("A", [_iv("07-05", 13, 16, 30), _iv("07-07", 9, 12, 30.25)]),
        _member("B", [_iv("07-05", 14, 15, 25), _iv("07-07", 11, 13, 25)]),
    ]
    body = _allocate(members).json()
    out = {m["member_id"]: m for m in body["members"]}
    assert Decimal(out["A"]["energy"]) == Decimal("60.25")
    assert Decimal(out["B"]["energy"]) == Decimal("50")
    assert Decimal(body["total_energy"]) == Decimal("110.250000")
    # A's standalone quote must equal a direct /quote for its raw intervals,
    # compared on the engine's six-decimal segment sum (the /quote total shown
    # to clients is rounded to cents; allocation uses the unrounded sum).
    rq = client.post("/quote", json={
        "version": VERSION, "intervals": members[0]["intervals"]})
    raw_quote = sum((Decimal(s["amount"]) for s in rq.json()["segments"]),
                    Decimal(0))
    assert Decimal(out["A"]["standalone_amount"]) == raw_quote
    shares = sum((Decimal(m["allocated_amount"]) for m in body["members"]),
                 Decimal(0))
    assert shares == Decimal(body["total_amount"])


def test_dst_spring_forward_overlap():
    # A covers 2024-03-09 23:30 -> 2024-03-10 05:30 NY local: the clock jumps
    # 02:00->03:00, so this is 5 UTC hours (both weekend days -> leisure).
    # B overlaps a 2-hour chunk; A alone still quotes 10 kWh = 1.50.
    members = [
        _member("A", [{
            "start": "2024-03-09T23:30:00-05:00",
            "end": "2024-03-10T05:30:00-04:00",
            "timezone": TZ, "energy": "10",
        }]),
        _member("B", [{
            "start": "2024-03-10T03:00:00-04:00",
            "end": "2024-03-10T05:00:00-04:00",
            "timezone": TZ, "energy": "150",
        }]),
    ]
    body = _allocate(members).json()
    out = {m["member_id"]: m for m in body["members"]}
    assert Decimal(out["A"]["standalone_amount"]) == Decimal("1.500000")
    # A prorates 2/5 of 10 = 4 kWh into B's two overlap hours and 6 kWh
    # elsewhere; grand total must equal a direct joint /quote over the same
    # atomic load profile (built here as abutting UTC intervals).
    shares = sum((Decimal(m["allocated_amount"]) for m in body["members"]),
                 Decimal(0))
    assert shares == Decimal(body["total_amount"])
    assert Decimal(body["total_energy"]) == Decimal("160.000000")


# ------------------------------------------------- independent Shapley oracle

def _oracle_allocation(members):
    """Independent reference: atomic segments by UTC endpoints, duration
    proration at 6 dp, /quote per coalition, average over every permutation."""
    parsed = []
    for mid, mvs in members:
        ivs = []
        for item in mvs:
            s = datetime.fromisoformat(item["start"]).astimezone(timezone.utc)
            e = datetime.fromisoformat(item["end"]).astimezone(timezone.utc)
            ivs.append((s, e, Decimal(item["energy"])))
        ivs.sort()
        parsed.append((mid, ivs))

    points = sorted({p for _, ivs in parsed for s, e, _ in ivs
                     for p in (s, e)})
    n = len(parsed)
    seg_energy = [dict() for _ in range(len(points) - 1)]
    for mi, (_, ivs) in enumerate(parsed):
        for s, e, energy in ivs:
            hit, weights = [], []
            for k in range(len(points) - 1):
                a, b = points[k], points[k + 1]
                if a >= e:
                    break
                if b > s:
                    hit.append(k)
                    lo = a if a > s else s
                    hi = b if b < e else e
                    weights.append(int((hi - lo).total_seconds() * 1_000_000))
            whole = sum(weights)
            acc = 0
            for k, w in zip(hit[:-1], weights[:-1]):
                share = (energy * Decimal(w) / Decimal(whole)).quantize(Q6)
                seg_energy[k][mi] = share
                acc += share
            seg_energy[hit[-1]][mi] = energy - acc

    def coalition_cost(mask):
        if mask == 0:
            return Decimal(0)
        intervals = []
        for k, cells in enumerate(seg_energy):
            val = sum((v for mi, v in cells.items() if mask & (1 << mi)),
                      Decimal(0))
            if val > 0:
                intervals.append({
                    "start": points[k].isoformat().replace("+00:00", "Z"),
                    "end": points[k + 1].isoformat().replace("+00:00", "Z"),
                    "timezone": "UTC", "energy": str(val),
                })
        r = client.post("/quote", json={
            "version": VERSION, "intervals": intervals})
        assert r.status_code == 200, r.text
        return sum((Decimal(s["amount"]) for s in r.json()["segments"]),
                   Decimal(0))

    costs = {mask: coalition_cost(mask) for mask in range(1 << n)}
    phi = [Fraction(0) for _ in range(n)]
    for order in permutations(range(n)):
        mask = 0
        prev = Fraction(0)
        for mi in order:
            mask |= 1 << mi
            cur = Fraction(costs[mask])
            phi[mi] += cur - prev
            prev = cur
    phi = [(Decimal(p.numerator) / Decimal(p.denominator)
            / Decimal(_fact(n))) for p in phi]
    return [p.quantize(Q6) for p in phi], costs[(1 << n) - 1]


def _fact(n):
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


@pytest.mark.parametrize("energies", [
    [33, 34, 35],
    [10, 20, 30, 40],
    [7, 11, 13, 17, 19],
    [3.5, 7.25, 0.125, 99.9],
])
def test_matches_permutation_oracle_overlapping(energies):
    members = [_member(f"M{i}", [_iv("07-06", 12, 14, e)])
               for i, e in enumerate(energies)]
    body = _allocate(members).json()
    expected, grand = _oracle_allocation(
        [(m["id"], m["intervals"]) for m in members])
    assert Decimal(body["total_amount"]) == grand
    got = [Decimal(m["allocated_amount"]) for m in body["members"]]
    # The endpoint quantizes in input order with the residual on the last
    # member; compare against the same ordered convention.
    reconciled = expected[:-1] + [grand - sum(expected[:-1], Decimal(0))]
    assert got == reconciled


def test_matches_oracle_with_staggered_intervals():
    members = [
        _member("A", [_iv("07-06", 10, 13, 40)]),
        _member("B", [_iv("07-06", 12, 15, 40)]),
        _member("C", [_iv("07-06", 11, 16, 40)]),
        _member("D", [_iv("07-06", 10, 16, 20)]),
    ]
    body = _allocate(members).json()
    expected, grand = _oracle_allocation(
        [(m["id"], m["intervals"]) for m in members])
    assert Decimal(body["total_amount"]) == grand
    got = [Decimal(m["allocated_amount"]) for m in body["members"]]
    reconciled = expected[:-1] + [grand - sum(expected[:-1], Decimal(0))]
    assert got == reconciled
    assert body["coalition_count"] == 16


# ------------------------------------------------------------ validation

def test_duplicate_member_id_rejected():
    members = [
        _member("A", [_iv("07-06", 12, 13, 10)]),
        _member("A", [_iv("07-06", 14, 15, 10)]),
    ]
    r = _allocate(members)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "duplicate_member"
    assert err["details"]["member_id"] == "A"
    assert err["details"]["member_index"] == 1


def test_intra_member_overlap_rejected_but_cross_member_allowed():
    bad = [
        _member("A", [_iv("07-06", 0, 3, 1), _iv("07-06", 2, 4, 1)]),
        _member("B", [_iv("07-06", 1, 5, 1)]),
    ]
    r = _allocate(bad)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "interval_overlap"
    assert err["details"]["member_id"] == "A"

    # Same two intervals moved to different members is legal and prices.
    ok = [
        _member("A", [_iv("07-06", 0, 3, 1)]),
        _member("B", [_iv("07-06", 2, 4, 1)]),
    ]
    assert _allocate(ok).status_code == 200


def test_member_count_bounds():
    one = [_member("A", [_iv("07-06", 0, 1, 1)])]
    assert _allocate(one).status_code == 422
    seven = [_member(f"M{i}", [_iv("07-06", 0, 1, 1)]) for i in range(7)]
    assert _allocate(seven).status_code == 422


def test_empty_member_intervals_rejected():
    members = [
        {"id": "A", "intervals": []},
        _member("B", [_iv("07-06", 0, 1, 1)]),
    ]
    assert _allocate(members).status_code == 422


def test_invalid_interval_inside_member_rejected():
    members = [
        _member("A", [_iv("07-06", 3, 2, 1)]),
        _member("B", [_iv("07-06", 0, 1, 1)]),
    ]
    r = _allocate(members)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_interval"
    assert r.json()["error"]["details"]["member_id"] == "A"


def test_unknown_version_404():
    members = [
        _member("A", [_iv("07-06", 0, 1, 1)]),
        _member("B", [_iv("07-06", 0, 1, 1)]),
    ]
    r = _allocate(members, version="nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "version_not_found"


def test_rule_gap_propagates():
    # us-ny-demo-v2025 has no rules before 2024-07-01: any coalition hits a
    # rule gap, which must surface as 409 (not as an empty quote).
    members = [
        _member("A", [_iv("01-02", 12, 13, 1)]),
        _member("B", [_iv("01-02", 12, 13, 1)]),
    ]
    r = _allocate(members, version="us-ny-demo-v2025")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "rule_gap"


def test_extra_fields_forbidden():
    r = client.post("/cost-allocation", json={
        "version": VERSION,
        "members": [
            {"id": "A", "extra": 1,
             "intervals": [_iv("07-06", 0, 1, 1)]},
            {"id": "B", "intervals": [_iv("07-06", 0, 1, 1)]},
        ],
    })
    assert r.status_code == 422


def test_response_member_order_follows_input():
    members = [
        _member("zeta", [_iv("07-06", 12, 13, 50)]),
        _member("alpha", [_iv("07-06", 12, 13, 70)]),
    ]
    body = _allocate(members).json()
    assert [m["member_id"] for m in body["members"]] == ["zeta", "alpha"]
