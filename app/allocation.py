"""共享电表下多方用能的阶梯电费公平拆分（Shapley 值）。

背景
====

一块总表下挂 2～6 个用能方（成员），各方按现有 ``/quote`` 的区间格式
申报自己的用量区间；成员**内部**区间互不重叠，成员**之间**允许重叠
（重叠时段各方都在用能，总表负荷为各方之和）。分时阶梯费率的累计
阶梯由**总表全部负荷**共同穿越：任何成员的边际费用都取决于还有哪些
成员同时在网，因此费用按 **Shapley 值**拆分——每名成员在全部加入
顺序中的**边际增费平均值**。空联盟费用为 0。

切分与分摊
==========

* 收集全部成员全部区间的 UTC 端点（半开区间的 start/end），排序去重后
  得到一串**原子段** ``[p_k, p_{k+1})``；原子段内活跃成员集合不变。
* 每个成员在其覆盖的每个原子段内，按**实际 UTC 微秒时长**比例分摊该
  成员对应原区间的申报能量：非末段在六位小数上四舍五入，舍入残差并入
  该区间的最后一个原子段，因此逐成员逐区间**严格能量守恒**。
* 任一成员联盟的负荷即各原子段内**联盟成员分摊量之和**（重叠即叠加）；
  每个原子段作为一条首尾相接的输入交给 :func:`pricing.quote` 计价
  （段内活跃成员集合恒定、负荷率一致，不合段以免跨规则边界二次扭曲
  时间分摊），累计阶梯跨全部段共享；报价金额为引擎逐段六位小数金额之和
  （不再二次舍入）。

枚举与归属
==========

* 成员数 ≤ 6，精确枚举全部 ``2^n`` 个联盟（含空联盟），每个非空联盟
  计价一次；Shapley 均值以 :class:`fractions.Fraction` 在有理数上精确
  累积，最后一次四舍五入到六位小数。
* 归属金额按**输入顺序**逐项量化到六位小数，最后一名取全体合并报价
  减去前项之和，因此归属金额之和**严格等于**全体合并报价。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction

from .pricing import (
    Q6,
    ZERO,
    PricingError,
    ensure_valid,
    get_zone,
    parse_local_datetime,
    quote,
)
from .schemas import QuoteIntervalIn


@dataclass(frozen=True)
class ParsedInterval:
    start: datetime          # UTC
    end: datetime            # UTC
    energy: Decimal
    tz: object               # ZoneInfo


@dataclass
class AllocationMember:
    id: str
    index: int
    intervals: list[ParsedInterval]
    energy: Decimal                      # 申报总量（精确值）
    standalone_amount: Decimal = ZERO    # 六位小数独立报价
    allocated_amount: Decimal = ZERO     # 六位小数 Shapley 归属金额


@dataclass
class AllocationResult:
    version: object
    members: list[AllocationMember]
    total_energy: Decimal                # 全体负荷（== 各成员申报量之和）
    total_amount: Decimal                # 六位小数全体合并报价
    coalition_count: int
    segments: list = field(default_factory=list)  # list[AtomicSegment]


@dataclass
class AtomicSegment:
    start: datetime                      # UTC，半开
    end: datetime                        # UTC
    # member_index -> prorated energy（六位小数；残差并入末段后严格守恒）
    energy: dict[int, Decimal]

    def energy_for(self, mask: int) -> Decimal:
        total = ZERO
        for idx, value in self.energy.items():
            if mask & (1 << idx):
                total += value
        return total


def to_utc_z(dt: datetime) -> str:
    """Format an aware UTC datetime with a trailing ``Z``."""
    return dt.isoformat().replace("+00:00", "Z")


def _micros(delta: timedelta) -> int:
    """Exact length of a timedelta in whole microseconds."""
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _parse_member(member, index: int) -> AllocationMember:
    """Parse one member's intervals; intervals must be non-overlapping."""
    intervals: list[ParsedInterval] = []
    for j, item in enumerate(member.intervals):
        tz = get_zone(item.timezone)
        start = parse_local_datetime(item.start, tz)
        end = parse_local_datetime(item.end, tz)
        if end <= start:
            raise PricingError(
                f"成员 {member.id} 的第 {j} 段用量区间必须满足 end > start"
                "（半开区间）",
                code="invalid_interval", status_code=400,
                details={"member_id": member.id, "interval_index": j},
            )
        energy = item.energy_decimal()
        if energy <= 0:
            raise PricingError(
                f"成员 {member.id} 的第 {j} 段用量必须为正数",
                code="invalid_interval", status_code=400,
                details={"member_id": member.id, "interval_index": j},
            )
        intervals.append(ParsedInterval(start, end, energy, tz))

    intervals.sort(key=lambda x: (x.start, x.end))
    for k in range(1, len(intervals)):
        if intervals[k].start < intervals[k - 1].end:
            raise PricingError(
                f"成员 {member.id} 的用量区间相互重叠；同一成员的区间必须"
                "互不重叠（不同成员之间允许重叠）",
                code="interval_overlap", status_code=400,
                details={"member_id": member.id,
                         "intervals": [k - 1, k]},
            )

    total = sum((iv.energy for iv in intervals), ZERO)
    return AllocationMember(id=member.id, index=index,
                            intervals=intervals, energy=total)


def _build_segments(members: list[AllocationMember]
                    ) -> list[AtomicSegment]:
    """Cut the UTC timeline at every member interval endpoint, then prorate
    each member's energy over the atomic segments its intervals cover by
    exact elapsed microseconds (6 dp, residual on each interval's last cell).
    """
    points: set[datetime] = set()
    for m in members:
        for iv in m.intervals:
            points.add(iv.start)
            points.add(iv.end)
    cuts = sorted(points)

    # segment_energies[s] maps member_index -> its prorated 6-dp energy inside
    # atomic segment s. A member's own intervals are non-overlapping, so at
    # most one can cover any given atomic segment.
    segment_energies: list[dict[int, Decimal]] = [
        dict() for _ in range(len(cuts) - 1)]
    for m in members:
        for iv in m.intervals:
            weights: list[int] = []
            hit: list[int] = []
            for seg_idx in range(len(cuts) - 1):
                a, b = cuts[seg_idx], cuts[seg_idx + 1]
                if a >= iv.end:
                    break
                if b > iv.start:
                    lo = a if a > iv.start else iv.start
                    hi = b if b < iv.end else iv.end
                    hit.append(seg_idx)
                    weights.append(_micros(hi - lo))

            for seg_idx, share in zip(hit, _distribute(iv.energy, weights)):
                if share > 0:
                    segment_energies[seg_idx][m.index] = share

    segments: list[AtomicSegment] = []
    for seg_idx, energies in enumerate(segment_energies):
        if energies:
            segments.append(AtomicSegment(
                start=cuts[seg_idx], end=cuts[seg_idx + 1],
                energy=dict(energies)))
    return segments


def _distribute(total: Decimal, weights: list[int]) -> list[Decimal]:
    """Split ``total`` proportionally to ``weights`` at 6 dp; the residual
    lands on the final part so the parts sum exactly to ``total``."""
    whole = sum(weights)
    if whole <= 0:
        return [ZERO] * (len(weights) - 1) + [total]
    shares = [(total * Decimal(w) / Decimal(whole)).quantize(
        Q6, rounding=ROUND_HALF_UP) for w in weights[:-1]]
    shares.append(total - sum(shares, ZERO))
    return shares


def _coalition_payload(mask: int,
                       segments: list[AtomicSegment]
                       ) -> list[QuoteIntervalIn]:
    """Build the engine payload for one coalition: one interval per atomic
    segment carrying positive coalition energy (segments are abutting, which
    the engine accepts). Segments are never merged — the active member set is
    constant inside a segment, so its load rate is uniform; merging across a
    junction that is also a rule boundary would distort time-based proration.
    """
    payload: list[QuoteIntervalIn] = []
    for seg in segments:
        energy = seg.energy_for(mask)
        if energy > 0:
            payload.append(_interval(seg.start, seg.end, energy))
    return payload


def _interval(start: datetime, end: datetime,
              energy: Decimal) -> QuoteIntervalIn:
    # All boundaries are already UTC instants; quote them with an explicit
    # UTC zone so no input-member timezone participates in pricing.
    return QuoteIntervalIn(start=to_utc_z(start), end=to_utc_z(end),
                           timezone="UTC", energy=energy)


def allocate(version, req) -> AllocationResult:
    """Validate members, enumerate every coalition quote, split by Shapley."""
    # Structural rule errors (overlap / tier gap / invalid rule) are a hard
    # failure shared by every coalition quote.
    ensure_valid(version)

    seen: set[str] = set()
    members: list[AllocationMember] = []
    for idx, item in enumerate(req.members):
        if item.id in seen:
            raise PricingError(
                f"成员 ID 重复: {item.id}；各成员 ID 必须唯一",
                code="duplicate_member", status_code=400,
                details={"member_id": item.id, "member_index": idx},
            )
        seen.add(item.id)
        members.append(_parse_member(item, idx))

    n = len(members)
    segments = _build_segments(members)

    # Quote every non-empty coalition exactly once. The intervals handed to
    # the engine are already sorted, non-overlapping UTC pieces of the
    # coalition's merged load; cumulative tiers span the whole joint load.
    costs: dict[int, Decimal] = {0: ZERO}
    grand_result = None
    full_mask = (1 << n) - 1
    for mask in range(1, 1 << n):
        payload = _coalition_payload(mask, segments)
        result = quote(version, payload, validate=False)
        costs[mask] = result.total_amount
        if mask == full_mask:
            grand_result = result

    # Shapley value: average marginal cost over every joining order. The
    # costs are exact 6-dp Decimals; the average accumulates as a Fraction so
    # the only rounding is the final 6-dp quantization per member.
    factorial = _factorial(n)
    for i, m in enumerate(members):
        bit = 1 << i
        acc = Fraction(0)
        for mask in range(0, 1 << n):
            if mask & bit:
                continue
            k = bin(mask).count("1")
            weight = (Fraction(_factorial(k) * _factorial(n - k - 1),
                               factorial))
            acc += weight * Fraction(costs[mask | bit] - costs[mask])
        m.standalone_amount = costs[bit]
        m.allocated_amount = _fraction_to_decimal(acc).quantize(
            Q6, rounding=ROUND_HALF_UP)

    # Quantize in input order; the last member absorbs the rounding residual
    # so allocated amounts sum exactly to the grand-coalition quote.
    grand_total = costs[full_mask]
    accounted = sum((m.allocated_amount for m in members[:-1]), ZERO)
    members[-1].allocated_amount = grand_total - accounted

    total_energy = (grand_result.total_energy
                    if grand_result is not None else ZERO)
    return AllocationResult(
        version=version, members=members,
        total_energy=total_energy, total_amount=grand_total,
        coalition_count=1 << n, segments=segments,
    )


def _factorial(k: int) -> int:
    out = 1
    for v in range(2, k + 1):
        out *= v
    return out


def _fraction_to_decimal(value: Fraction) -> Decimal:
    """High-precision Decimal view of a Fraction (40 significant digits —
    far beyond the 6-dp quantization applied afterwards)."""
    with localcontext() as ctx:
        ctx.prec = 40
        return Decimal(value.numerator) / Decimal(value.denominator)
