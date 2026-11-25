"""四大选择算法、经验校准器及大规模蒙特卡洛仿真器单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from tts_theory.algorithms import (
    BaseSelector,
    CalibratedWeightedBoN,
    EmpiricalScoreCalibrator,
    MajorityVoting,
    OptimalNStarBoN,
    StandardBoN,
    TwoStageVerifiedMajority,
)
from tts_theory.simulator import (
    DiscreteAnswerTaskSpec,
    _sample_scores_and_labels,
    simulate_algorithms_comparison,
    simulate_bon,
)
from tts_theory.theorems import ContaminatedVerifierSpec, GaussianVerifierSpec


def test_standard_bon_selection_and_tie_breaking() -> None:
    """验证 StandardBoN 正确选取最高分且平局时严格均匀随机打破。"""
    selector = StandardBoN()
    scores = np.array([0.2, 1.8, 0.9, 1.5])
    answers = np.array([10, 20, 30, 40])

    assert selector.select(scores) == 1
    assert selector.select(scores, answers=answers) == 20

    # 平局随机性检验：下标 1 与 3 并列最大值 2.5
    rng = np.random.default_rng(123)
    tie_scores = np.array([0.5, 2.5, 1.0, 2.5])
    counts = {1: 0, 3: 0}
    for _ in range(2_000):
        idx = selector.select(tie_scores, rng=rng)
        assert idx in counts
        counts[idx] += 1
    assert 0.42 < counts[1] / 2_000 < 0.58

    # select 与 select_batch 一致性
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    batch_scores = np.array([[0.1, 3.0, 2.0], [5.0, 1.0, 4.0]])
    batch_ans = np.array([[7, 8, 9], [1, 2, 3]])
    res_batch = selector.select_batch(batch_scores, batch_ans, rng=rng1)
    res_single = np.array([selector.select(batch_scores[i], batch_ans[i], rng=rng2) for i in range(2)])
    assert np.array_equal(res_batch, res_single)


def test_optimal_n_star_bon_truncation_and_early_stop() -> None:
    """验证 OptimalNStarBoN 在安全区不截断、坍塌区截断至 N* 以及高置信早停行为。"""
    safe_spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0)
    sel_safe = OptimalNStarBoN(spec=safe_spec)
    assert not sel_safe.will_collapse
    assert sel_safe.n_star is None

    # 安全区不截断，能选出末尾的最高分
    scores = np.array([1.0, 2.0, 0.5, 4.0])
    assert sel_safe.select(scores) == 3

    # 坍塌区截断：设置 n_star=3，第 4 个样本（下标 3）分数即使很高也应被忽略
    het_spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.6)
    sel_het = OptimalNStarBoN(spec=het_spec)
    assert sel_het.will_collapse
    assert sel_het.n_star is not None and sel_het.n_star >= 1

    sel_fixed = OptimalNStarBoN(n_star=3)
    scores_long = np.array([0.5, 2.2, 1.1, 9.9, 10.5])
    answers_long = np.array([1, 0, 2, 3, 4])
    assert sel_fixed.select(scores_long) == 1
    assert sel_fixed.select(scores_long, answers=answers_long) == 0

    # 动态覆盖 n_star 应立即生效
    sel_fixed.n_star = 4
    assert sel_fixed.select(scores_long) == 3

    # 早停阈值测试：当样本 0 的后验已满足 early_stop_posterior 时立即早停
    peak_q = float(het_spec.posterior(1.2))
    sel_early = OptimalNStarBoN(spec=het_spec, n_star=10, early_stop_posterior=peak_q * 0.9)
    # 下标 0 分数 1.2 达到高后验，虽下标 1 分数 1.5 后验可能略高但也直接早停在 0
    assert sel_early.select(np.array([1.2, 1.3, 0.5])) == 0


def test_calibrated_weighted_bon_avoids_high_score_tail_trap() -> None:
    """验证 CalibratedWeightedBoN 能识别极右尾假高分陷阱并支持有界权重聚票。"""
    spec = ContaminatedVerifierSpec(
        p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5,
        epsilon=0.10, mu_hack=4.0, sigma_hack=1.2,
    )
    selector = CalibratedWeightedBoN(spec=spec, w_max=10.0)

    # 样本 0 分数 1.3（后验高），样本 1 分数 6.5（Reward Hacking 极右尾，后验极低）
    scores = np.array([1.3, 6.5, -1.0])
    q = spec.posterior(scores)
    assert q[0] > q[1]
    assert selector.select(scores) == 0

    # 离散答案空间聚票：即使某孤立恶意样本有高分，也无法压倒多个高后验一致样本（且权重受 w_max 上限限制）
    answers = np.array([0, 0, 0, 1, 2])
    scores_vote = np.array([1.2, 1.4, 1.1, 6.5, -0.5])
    assert selector.select(scores_vote, answers=answers) == 0

    # 检验 w_max 上限约束真正截断单个极高后验样本的权重
    homo = GaussianVerifierSpec(p=0.25, mu1=2.0, sigma1=1.0, mu0=0.0, sigma0=1.0)
    sel_capped = CalibratedWeightedBoN(spec=homo, w_max=2.0)
    # 样本 0 极高分后验近 1，截断后权重为 2.0；样本 1,2 分数为 1.5（单条权重 > 1.2，三条合计 > 2.0）
    s_cap = np.array([8.0, 1.5, 1.5, 1.5])
    a_cap = np.array([9, 1, 1, 1])
    assert sel_capped.select(s_cap, answers=a_cap) == 1


def test_empirical_score_calibrator_nonmonotone_and_isotonic() -> None:
    """验证经验校准器既能捕获非单调尾部坍塌，也支持单调等渗拟合。"""
    hack_spec = ContaminatedVerifierSpec(
        p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5,
        epsilon=0.10, mu_hack=3.8, sigma_hack=1.3,
    )
    rng = np.random.default_rng(77)
    s_cal, y_cal = _sample_scores_and_labels(hack_spec, (30_000,), rng)

    cal_hist = EmpiricalScoreCalibrator(n_bins=60, method="quantile_hist").fit(s_cal, y_cal)
    q_peak = float(cal_hist.predict_proba(1.4))
    q_tail = float(cal_hist.predict_proba(5.5))
    assert q_peak > 0.40
    assert q_tail < 0.15
    assert q_peak > q_tail

    # 用纯经验校准器（不传 spec）驱动的两大校准选择器在大 N 下显著胜过 StandardBoN
    task = DiscreteAnswerTaskSpec(verifier_spec=hack_spec, num_wrong_answers=5)
    algos = {
        "Standard-BoN": StandardBoN(),
        "Empirical-Calibrated": CalibratedWeightedBoN(calibrator=cal_hist),
        "Empirical-TwoStage": TwoStageVerifiedMajority(calibrator=cal_hist),
    }
    sim = simulate_algorithms_comparison(task, [1, 256], algorithms=algos, n_trials=2_000, seed=11, batch_size=500)
    assert sim["algorithms"]["Empirical-Calibrated"]["accuracy"][-1] > 0.85
    assert sim["algorithms"]["Empirical-TwoStage"]["accuracy"][-1] > 0.85
    assert sim["algorithms"]["Standard-BoN"]["accuracy"][-1] < 0.25

    # 等渗校准在单调分布上应严格非降
    homo = GaussianVerifierSpec(p=0.3, mu1=1.5, sigma1=1.0, mu0=0.0, sigma0=1.0)
    s_mono, y_mono = _sample_scores_and_labels(homo, (10_000,), rng)
    cal_iso = EmpiricalScoreCalibrator(method="isotonic").fit(s_mono, y_mono)
    grid = np.linspace(-3.0, 4.0, 50)
    preds = cal_iso.predict_proba(grid)
    assert np.all(np.diff(preds) >= -1e-12)


def test_two_stage_verified_majority_filtering_and_fallback() -> None:
    """验证 TwoStageVerifiedMajority 第一阶段阈值过滤、显式阈值回退及自适应低 SNR 门控。"""
    spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5)
    # 显式阈值 0.38：只有高质量候选（如分数 1.2 附近 q≈0.408）能进入投票
    sel_strict = TwoStageVerifiedMajority(spec=spec, threshold=0.38)

    # 样本 0,1,2 是噪声/假尾 (<0.38)，但样本 0,1,2 全投答案 9；只有样本 3,4 通过门控并投答案 0
    scores = np.array([-2.0, -1.5, 5.0, 1.2, 1.3])
    answers = np.array([9, 9, 9, 0, 0])
    assert float(spec.posterior(5.0)) < 0.1
    assert float(spec.posterior(1.2)) >= 0.38
    assert sel_strict.select(scores, answers=answers) == 0

    # 全被过滤时的回退机制：设极高阈值 0.99，应回退到最大后验样本（下标 4）对应答案
    sel_ultra = TwoStageVerifiedMajority(spec=spec, threshold=0.99)
    scores_fb = np.array([-3.0, 5.0, 1.1])
    answers_fb = np.array([1, 2, 7])
    assert sel_ultra.select(scores_fb, answers=answers_fb) == 7
    assert sel_ultra.select(scores_fb) == 2

    # 自适应模式（threshold=None）在低 SNR 场景（所有后验都 < 0.5）下仍保留 top 分位样本聚票
    low_snr = GaussianVerifierSpec(p=0.15, mu1=0.6, sigma1=1.0, mu0=0.0, sigma0=1.0)
    sel_adapt = TwoStageVerifiedMajority(spec=low_snr, threshold=None, min_keep_fraction=0.2)
    # 即使没有单点超过 0.5，前 20% 样本聚票仍选出 0
    s_low = np.array([1.1, 1.2, 1.15, -1.0, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7])
    a_low = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4])
    assert float(np.max(low_snr.posterior(s_low))) < 0.5
    assert sel_adapt.select(s_low, answers=a_low) == 0


def test_validation_and_custom_selector_fallback() -> None:
    """验证输入合法性校验、乱序 n_values 顺序保留及纯 select 子类回退兼容。"""
    with pytest.raises(ValueError):
        StandardBoN().select(np.array([]))
    with pytest.raises(ValueError):
        StandardBoN().select(np.array([1.0, 2.0]), answers=np.array([0]))
    with pytest.raises(ValueError):
        CalibratedWeightedBoN()
    with pytest.raises(ValueError):
        CalibratedWeightedBoN(calibrator=EmpiricalScoreCalibrator())
    with pytest.raises(ValueError):
        OptimalNStarBoN(n_star=0)
    with pytest.raises(ValueError):
        DiscreteAnswerTaskSpec(
            verifier_spec=GaussianVerifierSpec(p=0.2, mu1=1.0, sigma1=1.0),
            num_wrong_answers=3,
            wrong_answer_probs=(0.5, 0.5),
        )
    with pytest.raises(ValueError):
        DiscreteAnswerTaskSpec(
            verifier_spec=GaussianVerifierSpec(p=0.2, mu1=1.0, sigma1=1.0),
            num_wrong_answers=2,
            wrong_answer_probs=(0.6, 0.6),
        )

    # 验证 simulate_bon 保留乱序和重复的 n_values
    spec = GaussianVerifierSpec(p=0.25, mu1=1.0, sigma1=1.0, mu0=0.0, sigma0=1.0)
    out = simulate_bon(spec, [32, 1, 8, 32], n_trials=2_000, seed=5, batch_size=500)
    assert np.array_equal(out["n_values"], np.array([32, 1, 8, 32]))
    assert out["accuracy"][0] == out["accuracy"][3]
    assert out["accuracy"][0] > out["accuracy"][1]

    # 自定义只重写 select（未重写 select_batch）的 BaseSelector 子类也能在仿真器正常运转
    class _FirstIndexSelector(BaseSelector):
        def select(
            self,
            scores: np.ndarray,
            answers: np.ndarray | None = None,
            rng: np.random.Generator | None = None,
        ) -> int:
            return 0 if answers is None else int(answers[0])

    task = DiscreteAnswerTaskSpec(verifier_spec=spec, num_wrong_answers=4)
    comp = simulate_algorithms_comparison(
        task_spec=task,
        n_values=[1, 10],
        algorithms={"First": _FirstIndexSelector(), "Majority": MajorityVoting()},
        n_trials=1_000,
        seed=42,
        batch_size=300,
    )
    # 永远挑第 0 个样本，准确期望等于 p=0.25
    assert comp["algorithms"]["First"]["accuracy"][1] == pytest.approx(0.25, abs=0.04)
